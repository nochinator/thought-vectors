"""ThinkerTrainer: train the thinker against a frozen (or partially unfrozen)
codec.

Loss modes (ablated via thinker.* config weights):
  w_thought  — MSE + (1-cosine) between predicted and frozen-encoder response
               thoughts (cheap; no decoder in the loop).
  w_decoder  — teacher-forced CE of the response text through the (frozen)
               decoder with predicted thoughts as memory (optimizes what we
               actually ship; gradients flow through frozen decoder weights
               into the prediction).
  w_reverse  — aux task, annealed to 0: from (context minus its last turn +
               the true response), predict the dropped turn's thoughts
               ("what was said to get this reply").
  w_cycle    — on cycle_frac of steps: greedy-decode the prediction, re-encode
               the text (both no-grad), pull the prediction toward what it
               actually decodes to (self-consistency).

Phase 2 (unfreeze="decoder"/"codec"): codec params join the optimizer at
codec_lr_scale * lr, and compress_frac of steps run a pure compression batch
from compress_shard so the thought space cannot drift away from
reconstructability.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, from_dict, to_dict
from .data import DialogueDataset, collate_dialogue, make_loader
from .generate import sample_decode
from .losses import reconstruction_ce
from .model import ThoughtAutoencoder, make_padding_mask
from .thinker import Thinker
from .tokenizer import Tokenizer
from .train_loop import lr_lambda


def thought_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return mse + cos


def thought_loss_weighted(
    pred: torch.Tensor, target: torch.Tensor, slot_weights: torch.Tensor
) -> torch.Tensor:
    """Slot-weighted thought loss: early slots get higher weight."""
    w = slot_weights[None, :].to(device=pred.device)  # [1, k]
    mse = (F.mse_loss(pred, target, reduction="none").mean(dim=-1) * w).mean()
    cos = (1 - F.cosine_similarity(pred, target, dim=-1)) * w
    cos = cos.mean()
    return mse + cos


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """NTXent: mean-pooled (over slots) pred[i] close to tgt[i], far from tgt[j≠i]."""
    p = pred.mean(dim=1)  # [B, d]
    t = target.mean(dim=1)  # [B, d]
    p = F.normalize(p, dim=-1)
    t = F.normalize(t, dim=-1)
    sim = torch.mm(p, t.T) / tau  # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def diversity_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalize low variance: 1 - mean pairwise cosine across batch."""
    p = pred.mean(dim=1)  # [B, d]
    p = F.normalize(p, dim=-1)
    sim = torch.mm(p, p.T)  # [B, B]
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    return 1.0 - sim[mask].mean()


def _position_weights(seq_len: int, decay: float, device) -> torch.Tensor:
    """Linear decay from 1.0 to `decay` over `seq_len` positions."""
    if seq_len <= 1 or decay >= 1.0:
        return torch.ones(seq_len, device=device)
    return torch.linspace(1.0, decay, seq_len, device=device)


def _slot_weights(k_out: int, decay: float, device) -> torch.Tensor:
    """Linear decay from 1.0 to `decay` over `k_out` thought slots."""
    if k_out <= 1 or decay >= 1.0:
        return torch.ones(k_out, device=device)
    return torch.linspace(1.0, decay, k_out, device=device)


def _ss_schedule(cfg, step: int, max_steps: int, elapsed_frac: float | None) -> float:
    """Scheduled sampling probability: linear ramp ss_prob → ss_prob_end."""
    tk = cfg.thinker
    if tk.ss_prob_end <= tk.ss_prob:
        return tk.ss_prob
    if elapsed_frac is not None:
        frac = elapsed_frac
    else:
        frac = min(step / max(max_steps, 1), 1.0)
    return tk.ss_prob + (tk.ss_prob_end - tk.ss_prob) * frac


def wta_losses(
    pred: torch.Tensor, target: torch.Tensor, slot_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """pred [B, M, k, d], target [B, k, d] -> per-hypothesis thought loss [B, M].

    slot_weights [k] (optional): per-slot weight over the k_out response thoughts,
    applied to BOTH the MSE and cosine terms before the slot-mean. Lets the WTA
    (frontier) recipe up-weight the starved tail slots — without it the WTA path
    means every slot equally and slot_weight_decay is unreachable for n_hyp>1.
    Reduces exactly to the unweighted mean when slot_weights is None / uniform.
    """
    target = target[:, None]
    se = ((pred - target) ** 2).mean(dim=-1)            # [B, M, k]
    cos_k = 1 - F.cosine_similarity(pred, target, dim=-1)  # [B, M, k]
    if slot_weights is not None:
        w = slot_weights.to(pred.device)
        denom = w.sum().clamp(min=1e-6)
        mse = (se * w).sum(dim=-1) / denom
        cos = (cos_k * w).sum(dim=-1) / denom
    else:
        mse = se.mean(dim=-1)
        cos = cos_k.mean(dim=-1)
    return mse + cos


def wta_select(
    pred: torch.Tensor, target: torch.Tensor, slot_weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, ...]:
    """Winner per sample: (winning pred [B, k, d], losses [B, M], winner [B])."""
    pl = wta_losses(pred, target, slot_weights)
    winner = pl.argmin(dim=1)
    rows = torch.arange(pred.size(0), device=pred.device)
    return pred[rows, winner], pl, winner


def encode_turns(
    codec: ThoughtAutoencoder,
    ids: torch.Tensor,
    k: int,
    grad: bool = False,
    tau: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """ids [..., T] -> (thoughts [..., k, d], per-turn budgets [...] or None).

    Absent context turns are all-PAD rows: the encoder sees a fully masked
    key set and emits NaN, and 0-attention-weight * NaN = NaN downstream,
    so their thoughts must be zeroed (the thinker's key_pad already hides
    them from attention).

    tau > 0: per-turn adaptive budget from the codec's loss predictor —
    smallest k' with predicted CE <= tau (predictor output space, log1p CE
    for predictor_log codecs). The thinker masks slots beyond the budget.
    """
    flat = ids.reshape(-1, ids.size(-1))
    mask = make_padding_mask(flat)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        th_full = codec.encode(flat, mask)
    th = th_full[:, :k].masked_fill(mask.all(dim=1)[:, None, None], 0.0)
    budgets = None
    if tau > 0:
        with torch.no_grad():
            pred = codec.predictor(th_full.detach())[:, :k]  # [rows, k] CE at prefix k'
            ok = pred <= tau
            first = ok.float().argmax(dim=1) + 1  # argmax = first True (monotone curve)
            budgets = torch.where(ok.any(dim=1), first, torch.full_like(first, k))
            budgets = budgets.clamp(min=2).reshape(ids.shape[:-1])
    return th.reshape(*ids.shape[:-1], k, th.size(-1)), budgets


def out_budget_mask(
    codec: ThoughtAutoencoder, pred: torch.Tensor, out_tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive response length (inference): per-sample smallest prefix of the
    importance-ordered predicted thoughts whose codec loss-predictor CE <=
    out_tau. Returns (budgets [B], memory_padding_mask [B, k_out], True=masked).

    The thinker's output must be importance-ordered for a prefix to be a valid
    shorter reply — train with k_out_random_prefix. The predictor is the codec's
    own (frozen) head; it transfers because the WTA recipe keeps predicted
    thoughts in codec distribution. Budget clamped to [2, k_out].
    """
    k = pred.size(1)
    with torch.no_grad():
        ce = codec.predictor(pred.detach())[:, :k]  # [B, k] predicted CE per prefix
        ok = ce <= out_tau
        first = ok.float().argmax(dim=1) + 1         # monotone curve: argmax = first True
        budgets = torch.where(ok.any(dim=1), first, torch.full_like(first, k)).clamp(min=2)
    slots = torch.arange(k, device=pred.device)
    return budgets, slots[None, :] >= budgets[:, None]


class ThinkerTrainer:
    def __init__(self, cfg: Config, tokenizer: Tokenizer) -> None:
        assert torch.get_float32_matmul_precision() == "highest"
        self.cfg = cfg
        self.device = torch.device(cfg.run.device)
        self.tokenizer = tokenizer

        ckpt = torch.load(cfg.thinker.codec_ckpt, map_location="cpu", weights_only=False)
        codec_cfg = from_dict(ckpt["config"])
        self.codec_cfg = codec_cfg
        self.codec = ThoughtAutoencoder(codec_cfg.model)
        self.codec.load_state_dict(ckpt["model"])
        self.codec = self.codec.to(self.device)
        self.codec.eval()
        if cfg.thinker.unfreeze == "codec":
            # MIOpen refuses RNN backward in eval mode, and encoder grads flow
            # when the full codec is unfrozen.  train() with every dropout
            # zeroed is numerically identical to eval() here.
            self.codec.train()
            for m in self.codec.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
                elif isinstance(m, torch.nn.GRU):
                    m.dropout = 0.0
        for p in self.codec.parameters():
            p.requires_grad_(False)

        self.thinker = Thinker(cfg.thinker, codec_cfg.model.d_model).to(self.device)
        if cfg.thinker.thinker_init_from:
            init_ckpt = torch.load(cfg.thinker.thinker_init_from, map_location="cpu",
                                   weights_only=False)
            # Filter params that match shape (allows warm-start across k_out changes)
            src = init_ckpt["thinker"]
            dst = self.thinker.state_dict()
            filtered = {}
            skipped = []
            for k, v in src.items():
                if k in dst and dst[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            self.thinker.load_state_dict(filtered, strict=False)
            print(f"thinker weights warm-started from {cfg.thinker.thinker_init_from}"
                  f" (loaded: {len(filtered)}, skipped: {len(skipped)})",
                  flush=True)
            # Unfreeze checkpoints carry an adapted codec (e.g. R4_UNFREEZE's
            # decoder) — continue from it, not the pristine codec_ckpt, so the
            # thinker's warm-started weights see the decoder they were trained
            # against.
            if "codec" in init_ckpt:
                self.codec.load_state_dict(init_ckpt["codec"])
                print("codec warm-started from the same checkpoint "
                      "(adapted decoder restored)", flush=True)
        print(f"thinker params: {self.thinker.param_count() / 1e6:.2f}M "
              f"(codec d={codec_cfg.model.d_model}, frozen={cfg.thinker.unfreeze=='none'})",
              flush=True)

        groups = [{"params": list(self.thinker.parameters()), "lr": cfg.train.lr}]
        if cfg.thinker.unfreeze != "none":
            unfrozen = (
                self.codec.decoder.parameters()
                if cfg.thinker.unfreeze == "decoder"
                else self.codec.parameters()
            )
            ups = [p for p in unfrozen]
            for p in ups:
                p.requires_grad_(True)
            groups.append({"params": ups, "lr": cfg.train.lr * cfg.thinker.codec_lr_scale})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
        self.train_start: float | None = None

        def _lr(step: int) -> float:
            frac = None
            if cfg.train.max_seconds and self.train_start is not None:
                frac = (time.time() - self.train_start) / cfg.train.max_seconds
            return lr_lambda(step, cfg, frac)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr)
        self.step = 0
        self.best_val = float("inf")
        self.nan_streak = 0
        self.prior_elapsed = 0.0

        self.compress_loader = None
        if cfg.thinker.compress_frac > 0:
            self.compress_loader = iter_cycle(
                make_loader(cfg.thinker.compress_shard, cfg.train.batch_size,
                            shuffle=True, num_workers=0, seed=cfg.train.seed)
            )

        self.run_dir = Path(cfg.run.out_dir) / cfg.run.name
        self.log_dir = Path(cfg.run.log_dir) / cfg.run.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "config.yaml").write_text(
            __import__("yaml").safe_dump(to_dict(cfg), sort_keys=False)
        )
        self.metrics_file = open(self.log_dir / "metrics.jsonl", "a")
        self.samples_file = open(self.log_dir / "samples.txt", "a")

    # ----- core step -----

    def train_step(self, batch: dict) -> dict | None:
        cfg = self.cfg
        tk = cfg.thinker
        dev = self.device

        if self.compress_loader is not None and torch.rand(()).item() < tk.compress_frac:
            return self._compression_step(next(self.compress_loader))

        ctx_ids = batch["ctx_ids"].to(dev)
        resp_ids = batch["resp_ids"].to(dev)
        ctx_roles = batch["ctx_roles"].to(dev)
        ctx_turns = batch["ctx_turns"].to(dev)
        resp_roles = batch["resp_roles"].to(dev)

        # --- round-4 augmentations ---
        if tk.turn_dropout > 0:
            # Drop intermediate context turns (keep last turn always)
            for b in range(ctx_ids.size(0)):
                nturn = int(ctx_turns[b].item())
                if nturn <= 2:
                    continue
                for j in range(nturn - 1):  # all except the last turn
                    if torch.rand(()).item() < tk.turn_dropout:
                        ctx_ids[b, j].zero_()
                        ctx_turns[b] -= 1
        if tk.role_flip and torch.rand(()).item() < 0.5:
            ctx_roles = 1 - ctx_roles
            resp_roles = 1 - resp_roles

        grad_enc = tk.unfreeze == "codec"
        ctx_th, budgets = encode_turns(
            self.codec, ctx_ids, tk.k_ctx, grad=grad_enc, tau=tk.ctx_tau
        )
        with torch.no_grad():
            tgt_th, _ = encode_turns(self.codec, resp_ids, tk.k_out)

        pred = self.thinker(ctx_th, ctx_roles, ctx_turns, resp_roles,
                            target_thoughts=tgt_th, slot_budgets=budgets,
                            ss_prob=_ss_schedule(cfg, self.step, cfg.train.max_steps,
                                                 (time.time() - self.train_start) / max(cfg.train.max_seconds, 1)
                                                 if cfg.train.max_seconds and self.train_start else None))

        losses: dict[str, torch.Tensor] = {}
        if pred.dim() == 4:  # [B, M, k, d] WTA multi-hypothesis head
            sw = (
                _slot_weights(pred.size(2), tk.slot_weight_decay, pred.device)
                if tk.slot_weight_decay > 0
                else None
            )
            pred, pl, winner = wta_select(pred, tgt_th, sw)
            if tk.w_thought > 0:
                rows = torch.arange(pl.size(0), device=pl.device)
                eps = tk.wta_eps
                losses["thought"] = tk.w_thought * (
                    (1 - eps) * pl[rows, winner].mean() + eps * pl.mean()
                )
        elif tk.w_thought > 0:
            if tk.slot_weight_decay > 0:
                sw = _slot_weights(pred.size(1), tk.slot_weight_decay, pred.device)
                losses["thought"] = tk.w_thought * thought_loss_weighted(pred, tgt_th, sw)
            else:
                losses["thought"] = tk.w_thought * thought_loss(pred, tgt_th)
        if tk.w_decoder > 0:
            resp_pad = make_padding_mask(resp_ids)
            # Random-prefix training: the thinker must order its output by
            # importance so any prefix is decodable (mirrors codec training).
            dec_pred = pred
            if tk.k_out_random_prefix:
                k_full = pred.size(1)
                if torch.rand(()).item() < tk.k_out_full_frac:
                    rk = k_full
                else:
                    rk = torch.randint(tk.k_out_min, k_full + 1, (1,)).item()
                dec_pred = pred[:, :rk]
            logits = self.codec.decode(dec_pred, resp_ids[:, :-1], resp_pad[:, :-1])
            if tk.pos_weight_decay > 0:
                seq = logits.size(1)
                w = _position_weights(seq, tk.pos_weight_decay, logits.device)
                tok_ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    resp_ids[:, 1:].reshape(-1),
                    ignore_index=0, reduction="none",
                ).view(logits.size(0), seq)
                valid = (resp_ids[:, 1:] != 0).float()
                total_w = (valid * w[None, :]).sum()
                ce = (tok_ce * valid * w[None, :]).sum() / total_w.clamp(min=1)
            else:
                ce, _ = reconstruction_ce(logits, resp_ids[:, 1:])
            losses["dec_ce"] = tk.w_decoder * ce
        if tk.contrast_weight > 0:
            losses["contrast"] = tk.contrast_weight * contrastive_loss(pred, tgt_th)
        if tk.diversity_weight > 0:
            losses["diversity"] = tk.diversity_weight * diversity_loss(pred)
        if tk.w_reverse > 0 and tk.mode == "query":
            w = self._reverse_weight()
            if w > 0 and int(ctx_turns.min()) >= 1:
                rev_ctx, rev_tgt = self._reverse_arrangement(ctx_th, tgt_th, ctx_turns)
                rev_pred = self.thinker.predict_reverse(
                    rev_ctx, ctx_roles, ctx_turns, resp_roles, slot_budgets=budgets
                )
                losses["reverse"] = w * thought_loss(rev_pred, rev_tgt)
        if tk.w_cycle > 0 and torch.rand(()).item() < tk.cycle_frac:
            with torch.no_grad():
                ids = sample_decode(self.codec, pred.detach(),
                                    self.codec_cfg.model.max_seq_len, temperature=0.0)
                re_th = self.codec.encode(ids, make_padding_mask(ids))[:, : tk.k_out]
            losses["cycle"] = tk.w_cycle * thought_loss(pred, re_th)

        total = sum(losses.values())
        if not torch.is_tensor(total) or not total.isfinite():
            self.nan_streak += 1
            self.optimizer.zero_grad(set_to_none=True)
            if self.nan_streak >= 20:
                raise RuntimeError(f"20 consecutive non-finite losses at step {self.step}")
            return None
        self.nan_streak = 0

        total.backward()
        norm = nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]], cfg.train.grad_clip
        )
        if not norm.isfinite():
            # finite loss, NaN grad: clip would scale ALL grads by NaN and the
            # optimizer would poison the weights permanently (killed ab_T0 at
            # step 7334). Skip the step; weights stay clean.
            self.nan_streak += 1
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            if self.nan_streak >= 20:
                raise RuntimeError(f"20 consecutive non-finite grads at step {self.step}")
            return None
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            cos = F.cosine_similarity(pred, tgt_th, dim=-1).mean().item()
        out = {"step": self.step, "cos": round(cos, 4),
               "lr": self.scheduler.get_last_lr()[0]}
        for name, value in losses.items():
            out[name] = round(value.item(), 4)
        return out

    def _reverse_weight(self) -> float:
        tk = self.cfg.thinker
        if not self.cfg.train.max_seconds or self.train_start is None:
            return tk.w_reverse
        frac = (time.time() - self.train_start) / self.cfg.train.max_seconds
        return tk.w_reverse * max(0.0, 1 - frac / max(tk.reverse_anneal_frac, 1e-6))

    @staticmethod
    def _reverse_arrangement(ctx_th, tgt_th, ctx_turns):
        """Replace each sample's last real context turn with the true response;
        the dropped turn becomes the reverse-prediction target."""
        rev_ctx = ctx_th.clone()
        bsz = ctx_th.size(0)
        rows = torch.arange(bsz, device=ctx_th.device)
        last = (ctx_turns - 1).clamp(min=0)
        rev_tgt = ctx_th[rows, last].detach()
        rev_ctx[rows, last] = tgt_th[:, : ctx_th.size(2)]
        return rev_ctx, rev_tgt

    def _compression_step(self, input_ids: torch.Tensor) -> dict | None:
        """Anchor step: the codec must keep reconstructing prose (phase 2)."""
        input_ids = input_ids.to(self.device)
        mask = make_padding_mask(input_ids)
        th = self.codec.encode(input_ids, mask)
        k = max(2, int(torch.randint(2, self.codec_cfg.model.num_thoughts + 1, (1,)).item()))
        logits = self.codec.decode(th[:, :k], input_ids[:, :-1], mask[:, :-1])
        ce, _ = reconstruction_ce(logits, input_ids[:, 1:])
        if not ce.isfinite():
            self.optimizer.zero_grad(set_to_none=True)
            return None
        ce.backward()
        norm = nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            self.cfg.train.grad_clip,
        )
        if not norm.isfinite():
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            return None
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"step": self.step, "compress_ce": round(ce.item(), 4),
                "lr": self.scheduler.get_last_lr()[0]}

    # ----- validation / samples -----

    @torch.no_grad()
    def validate(self, val_loader, max_batches: int = 16) -> dict:
        self.thinker.eval()
        tk = self.cfg.thinker
        cos_sum, ce_sum, n = 0.0, 0.0, 0
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            ctx_ids = batch["ctx_ids"].to(self.device)
            resp_ids = batch["resp_ids"].to(self.device)
            ctx_th, budgets = encode_turns(
                self.codec, ctx_ids, tk.k_ctx, tau=tk.ctx_tau
            )
            tgt_th, _ = encode_turns(self.codec, resp_ids, tk.k_out)
            pred = self.thinker(
                ctx_th, batch["ctx_roles"].to(self.device),
                batch["ctx_turns"].to(self.device), batch["resp_roles"].to(self.device),
                target_thoughts=tgt_th if tk.mode == "prefix" else None,
                slot_budgets=budgets,
            )
            if pred.dim() == 4:
                pred, _, _ = wta_select(pred, tgt_th)  # best-of-M validation
            cos_sum += F.cosine_similarity(pred, tgt_th, dim=-1).mean().item()
            n += 1
            if tk.w_decoder > 0:
                resp_pad = make_padding_mask(resp_ids)
                logits = self.codec.decode(pred, resp_ids[:, :-1], resp_pad[:, :-1])
                ce, _ = reconstruction_ce(logits, resp_ids[:, 1:])
                ce_sum += ce.item()
        self.thinker.train()
        out = {"val_cos": cos_sum / max(n, 1)}
        if tk.w_decoder > 0:
            out["val_dec_ce"] = ce_sum / max(n, 1)
        return out

    @torch.no_grad()
    def dump_samples(self, batch: dict, max_rows: int = 4) -> None:
        self.thinker.eval()
        dev = self.device
        ctx_ids = batch["ctx_ids"][:max_rows].to(dev)
        ctx_th, budgets = encode_turns(
            self.codec, ctx_ids, self.cfg.thinker.k_ctx, tau=self.cfg.thinker.ctx_tau
        )
        pred = self.thinker(
            ctx_th, batch["ctx_roles"][:max_rows].to(dev),
            batch["ctx_turns"][:max_rows].to(dev), batch["resp_roles"][:max_rows].to(dev),
            slot_budgets=budgets,
        )
        if pred.dim() == 4:  # WTA: decode every hypothesis to eyeball diversity
            bsz, m = pred.shape[:2]
            flat_ids = sample_decode(self.codec, pred.reshape(bsz * m, *pred.shape[2:]),
                                     self.codec_cfg.model.max_seq_len,
                                     temperature=0.0, no_repeat_ngram=3)
            hyp_texts = [
                [self.tokenizer.decode(flat_ids[r * m + h].tolist()) for h in range(m)]
                for r in range(bsz)
            ]
        else:
            ids = sample_decode(self.codec, pred, self.codec_cfg.model.max_seq_len,
                                temperature=0.0, no_repeat_ngram=3)
            hyp_texts = [[self.tokenizer.decode(ids[r].tolist())] for r in range(pred.size(0))]
        lines = [f"--- step {self.step} ---"]
        for row in range(ctx_ids.size(0)):
            nturn = int(batch["ctx_turns"][row])
            for j in range(nturn):
                t = [x for x in ctx_ids[row, j].tolist() if x != 0]
                who = "user" if batch["ctx_roles"][row, j] == 0 else "bot"
                lines.append(f"  {who}: {self.tokenizer.decode(t)}")
            ref = [x for x in batch["resp_ids"][row].tolist() if x != 0]
            lines.append(f"  REF : {self.tokenizer.decode(ref)}")
            for h, text in enumerate(hyp_texts[row]):
                tag = f"PRED{h}" if len(hyp_texts[row]) > 1 else "PRED"
                lines.append(f"  {tag}: {text}")
            lines.append("")
        self.samples_file.write("\n".join(lines) + "\n")
        self.samples_file.flush()
        self.thinker.train()

    # ----- checkpointing -----

    def save_checkpoint(self, tag: str | None = None) -> Path:
        name = tag or f"step_{self.step}"
        path = self.run_dir / f"{name}.pt"
        state = {
            "step": self.step,
            "thinker": self.thinker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val": self.best_val,
            "elapsed": (time.time() - self.train_start) if self.train_start else 0.0,
            "config": to_dict(self.cfg),
            "codec_ckpt": self.cfg.thinker.codec_ckpt,
        }
        if self.cfg.thinker.unfreeze != "none":
            state["codec"] = self.codec.state_dict()
        torch.save(state, path)
        if tag is None:
            ckpts = sorted(self.run_dir.glob("step_*.pt"),
                           key=lambda p: int(p.stem.split("_")[1]))
            for old in ckpts[: -self.cfg.train.keep_ckpts]:
                old.unlink()
        return path

    def load_checkpoint(self, path: str | Path, reset_schedule: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.thinker.load_state_dict(ckpt["thinker"])
        if "codec" in ckpt:
            self.codec.load_state_dict(ckpt["codec"])
        if not reset_schedule:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.step = ckpt["step"]
            self.best_val = ckpt["best_val"]
            self.prior_elapsed = ckpt.get("elapsed", 0.0)

    # ----- main loop -----

    def fit(self, train_loader, val_loader) -> None:
        cfg = self.cfg
        self.thinker.train()
        self.train_start = time.time() - self.prior_elapsed
        t0 = time.time()
        window: list[float] = []
        data_iter = iter(train_loader)
        sample_batch = None

        while self.step < cfg.train.max_steps:
            if cfg.train.max_seconds and time.time() - self.train_start > cfg.train.max_seconds:
                print(f"wall-clock cap reached at step {self.step}", flush=True)
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            if sample_batch is None:
                sample_batch = batch

            metrics = self.train_step(batch)
            self.step += 1
            if metrics is None:
                continue
            if "cos" in metrics:
                window.append(metrics["cos"])

            if self.step % cfg.train.log_every == 0 and window:
                rate = cfg.train.log_every / (time.time() - t0)
                t0 = time.time()
                metrics["cos_avg"] = round(sum(window) / len(window), 4)
                window = []
                metrics["it_s"] = round(rate, 2)
                print(
                    f"step {self.step:>7} | cos {metrics['cos_avg']:.4f} | "
                    + " ".join(f"{k} {v}" for k, v in metrics.items()
                               if k not in ("step", "cos", "cos_avg", "it_s", "lr"))
                    + f" | {rate:.1f} it/s",
                    flush=True,
                )
                self.metrics_file.write(json.dumps(metrics) + "\n")
                self.metrics_file.flush()

            if self.step % cfg.train.sample_every == 0:
                self.dump_samples(sample_batch)

            if self.step % cfg.train.val_every == 0:
                val = self.validate(val_loader)
                parts = [f"VAL cos {val['val_cos']:.4f}"]
                if "val_dec_ce" in val:
                    parts.append(f"VAL dec CE {val['val_dec_ce']:.4f}")
                print(f"step {self.step:>7} | " + " | ".join(parts), flush=True)
                self.metrics_file.write(json.dumps({"step": self.step, **val}) + "\n")
                self.metrics_file.flush()
                score = val.get("val_dec_ce", -val["val_cos"])
                if score < self.best_val:
                    self.best_val = score
                    self.save_checkpoint("best")

            if self.step % cfg.train.ckpt_every == 0:
                self.save_checkpoint()

        final = self.validate(val_loader)
        self.metrics_file.write(json.dumps({"step": self.step, **final}) + "\n")
        self.metrics_file.flush()
        score = final.get("val_dec_ce", -final["val_cos"])
        if score < self.best_val:
            self.best_val = score
            self.save_checkpoint("best")
        self.save_checkpoint("final")
        parts = [f"final val cos {final['val_cos']:.4f}"]
        if "val_dec_ce" in final:
            parts.append(f"dec CE {final['val_dec_ce']:.4f}")
        print(f"\nRun '{cfg.run.name}' done: {self.step} steps | "
              + " | ".join(parts), flush=True)


def iter_cycle(loader):
    while True:
        for b in loader:
            yield b


def make_dialogue_loader(shard_dir, batch_size, max_context, shuffle=True, num_workers=2,
                         seed=1234, flat_context=False, max_flat_tokens=256):
    from torch.utils.data import DataLoader

    ds = DialogueDataset(shard_dir, max_context=max_context,
                         flat_context=flat_context, max_flat_tokens=max_flat_tokens)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_dialogue,
        num_workers=num_workers, pin_memory=True, drop_last=shuffle,
        generator=gen if shuffle else None, persistent_workers=num_workers > 0,
    )
