"""Round B baseline: a textbook decoder-only token LM on the thinker's data.

Same shards, tokenizer, and wall-clock protocol as the thinker
(docs/BASELINE_ABLATIONS.md). Data view: DialogueDataset(flat_context=True)
gives the history as one BOS..EOS token stream; the response is appended and
CE is taken on response positions only — the thinker's "every non-first turn
is a target" rule, in token space.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, from_dict, to_dict
from .data import DialogueDataset
from .model import PAD_ID
from .tokenizer import BOS_ID, EOS_ID, Tokenizer
from .train_loop import lr_lambda


class TokenLM(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        lm = cfg.lm
        self.max_len = lm.max_seq_len
        self.tok = nn.Embedding(cfg.model.vocab_size, lm.d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(lm.max_seq_len, lm.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=lm.d_model,
            nhead=lm.nhead,
            dim_feedforward=lm.ffn_dim,
            dropout=lm.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=lm.layers)
        self.norm = nn.LayerNorm(lm.d_model)
        self.head = nn.Linear(lm.d_model, cfg.model.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.size(1)
        x = self.tok(ids) + self.pos(torch.arange(t, device=ids.device))[None]
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=ids.device)
        h = self.trunk(x, mask=causal, src_key_padding_mask=ids.eq(PAD_ID))
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(
        self, ctx_ids: torch.Tensor, max_new: int = 64, temperature: float = 0.0
    ) -> list[int]:
        """ctx_ids [T] (flat history) -> generated reply ids (BOS/EOS stripped)."""
        ids = torch.cat(
            [ctx_ids, torch.tensor([BOS_ID], device=ctx_ids.device)]
        )[None, -self.max_len :]
        out: list[int] = []
        for _ in range(max_new):
            logits = self(ids)[0, -1]
            if temperature > 0:
                nxt = int(torch.multinomial(F.softmax(logits / temperature, -1), 1))
            else:
                nxt = int(logits.argmax())
            if nxt == EOS_ID:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            ids = ids[:, -self.max_len :]
        return out


def collate_lm(batch: list[dict], max_len: int) -> dict:
    """cat(flat context, response), PAD-batched; loss_mask marks response."""
    seqs, resp_lens = [], []
    for b in batch:
        ctx, resp = b["context"][0], b["response"]
        room = max(max_len - resp.size(0), 0)
        if ctx.size(0) > room:
            ctx = ctx[-room:]
        seqs.append(torch.cat([ctx, resp])[:max_len])
        resp_lens.append(min(resp.size(0), max_len))
    t = max(s.size(0) for s in seqs)
    ids = torch.full((len(seqs), t), PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros(len(seqs), t, dtype=torch.bool)
    for i, (s, rl) in enumerate(zip(seqs, resp_lens)):
        ids[i, : s.size(0)] = s
        loss_mask[i, s.size(0) - rl : s.size(0)] = True
    return {"ids": ids, "loss_mask": loss_mask}


def lm_ce(model: TokenLM, ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    logits = model(ids)[:, :-1]
    tgt = ids[:, 1:].masked_fill(~loss_mask[:, 1:], -100)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=-100)


class LMChatSession:
    """Drop-in for chat.ChatSession (reply / reset / history) backed by
    TokenLM, so the thinker eval scripts run unchanged against the Round B
    baseline. History is the flat BOS..EOS token stream the LM trained on."""

    def __init__(self, ckpt_path: str, device: str = "cpu",
                 max_new: int = 64) -> None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.cfg = from_dict(ckpt["cfg"])
        self.model = TokenLM(self.cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.tokenizer = Tokenizer(self.cfg.run.tokenizer_path)
        self.device = device
        self.max_new = max_new
        self.history: list[str] = []  # alternating, history[0] = user

    @torch.no_grad()
    def reply(self, user_text: str, temperature: float = 0.0) -> str:
        self.history.append(user_text.strip())
        ids: list[int] = []
        for t in self.history:
            ids += [BOS_ID] + self.tokenizer.encode(t, add_special=False) + [EOS_ID]
        room = self.model.max_len - self.max_new - 1
        ctx = torch.tensor(ids[-room:], dtype=torch.long, device=self.device)
        out = self.model.generate(ctx, max_new=self.max_new,
                                  temperature=temperature)
        text = self.tokenizer.decode(out)
        self.history.append(text)
        return text

    def reset(self) -> None:
        self.history.clear()


class LMTrainer:
    def __init__(self, cfg: Config, tokenizer: Tokenizer) -> None:
        assert torch.get_float32_matmul_precision() == "highest", (
            "float32 matmul precision must remain 'highest' on this ROCm stack"
        )
        self.cfg = cfg
        self.device = torch.device(cfg.run.device)
        self.tokenizer = tokenizer
        self.model = TokenLM(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
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

        def _mk(shard_dir: str, shuffle: bool) -> DataLoader:
            ds = DialogueDataset(
                shard_dir,
                max_context=cfg.lm.max_turns,
                flat_context=True,
                max_flat_tokens=cfg.lm.max_seq_len - 128,
            )
            gen = torch.Generator()
            gen.manual_seed(cfg.train.seed)
            return DataLoader(
                ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
                collate_fn=lambda b: collate_lm(b, cfg.lm.max_seq_len),
                num_workers=cfg.data.num_workers, pin_memory=True,
                drop_last=shuffle, generator=gen if shuffle else None,
                persistent_workers=cfg.data.num_workers > 0,
            )

        val_dir = cfg.data.val_shard_dir or cfg.data.shard_dir + "_val"
        self.loader = _mk(cfg.data.shard_dir, shuffle=True)
        self.val_loader = _mk(val_dir, shuffle=False)

        self.run_dir = Path(cfg.run.out_dir) / cfg.run.name
        self.log_dir = Path(cfg.run.log_dir) / cfg.run.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "config.yaml").write_text(
            __import__("yaml").safe_dump(to_dict(cfg), sort_keys=False)
        )
        self.metrics_file = open(self.log_dir / "metrics.jsonl", "a")
        self.samples_file = open(self.log_dir / "samples.txt", "a")
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[lm] {cfg.run.name}: {n_params/1e6:.1f}M params "
              f"(d{cfg.lm.d_model} x {cfg.lm.layers}L, ffn {cfg.lm.ffn_dim})")

    def _log(self, rec: dict) -> None:
        import json as _json

        self.metrics_file.write(_json.dumps(rec) + "\n")
        self.metrics_file.flush()

    @torch.no_grad()
    def validate(self, max_batches: int = 50) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            ce = lm_ce(self.model, batch["ids"].to(self.device),
                       batch["loss_mask"].to(self.device))
            tot += float(ce)
            n += 1
        self.model.train()
        return tot / max(n, 1)

    @torch.no_grad()
    def sample(self, n: int = 4) -> None:
        self.model.eval()
        batch = next(iter(self.val_loader))
        for i in range(min(n, batch["ids"].size(0))):
            ids, mask = batch["ids"][i], batch["loss_mask"][i]
            ctx = ids[: int(mask.float().argmax())].to(self.device)
            reply = self.model.generate(ctx)
            gold = ids[mask].tolist()
            self.samples_file.write(
                f"--- step {self.step}\nctx  > {self.tokenizer.decode(ctx.tolist())}\n"
                f"gold > {self.tokenizer.decode(gold)}\n"
                f"pred > {self.tokenizer.decode(reply)}\n"
            )
        self.samples_file.flush()
        self.model.train()

    def save(self, name: str) -> None:
        elapsed = time.time() - self.train_start if self.train_start else 0.0
        torch.save(
            {"model": self.model.state_dict(), "cfg": to_dict(self.cfg),
             "optimizer": self.optimizer.state_dict(),
             "scheduler": self.scheduler.state_dict(),
             "step": self.step, "best_val": self.best_val,
             "elapsed": elapsed},
            self.run_dir / name,
        )

    def load_checkpoint(self, path: str) -> None:
        """Resume a crashed run: weights, optimizer, and — critically for the
        wall-clock LR schedule and max_seconds stop — cumulative elapsed."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        self.best_val = ckpt["best_val"]
        self._resume_elapsed = float(ckpt.get("elapsed", 0.0))

    def fit(self) -> None:
        cfg = self.cfg
        self.model.train()
        self.train_start = time.time() - getattr(self, "_resume_elapsed", 0.0)
        done = False
        while not done:
            for batch in self.loader:
                if cfg.train.max_seconds and time.time() - self.train_start >= cfg.train.max_seconds:
                    done = True
                    break
                if self.step >= cfg.train.max_steps:
                    done = True
                    break
                ce = lm_ce(self.model, batch["ids"].to(self.device, non_blocking=True),
                           batch["loss_mask"].to(self.device, non_blocking=True))
                if not torch.isfinite(ce):
                    self.nan_streak += 1
                    if self.nan_streak >= 20:
                        raise RuntimeError("20 consecutive non-finite losses — aborting")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                ce.backward()
                # gfx1031 emits rare non-finite grads under finite loss; an
                # unchecked clip-by-NaN-norm poisons every weight permanently.
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                if not torch.isfinite(norm):
                    self.nan_streak += 1
                    if self.nan_streak >= 20:
                        raise RuntimeError("20 consecutive non-finite grad norms — aborting")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.nan_streak = 0
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                if self.step % cfg.train.log_every == 0:
                    ce_val = float(ce.detach())
                    self._log({"step": self.step, "ce": ce_val,
                               "ppl": math.exp(min(ce_val, 20)),
                               "lr": self.scheduler.get_last_lr()[0],
                               "elapsed": time.time() - self.train_start})
                if self.step % cfg.train.val_every == 0:
                    val = self.validate()
                    self._log({"step": self.step, "val_ce": val})
                    if val < self.best_val:
                        self.best_val = val
                        self.save("best.pt")
                if self.step % cfg.train.sample_every == 0:
                    self.sample()
                if self.step % cfg.train.ckpt_every == 0:
                    self.save("last.pt")
        val = self.validate()
        self._log({"step": self.step, "val_ce": val, "final": True})
        if val < self.best_val:
            self.best_val = val
            self.save("best.pt")
        self.save("last.pt")
        print(f"[lm] done: {self.step} steps, best val_ce {self.best_val:.4f}")
