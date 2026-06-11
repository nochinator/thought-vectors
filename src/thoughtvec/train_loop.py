"""Trainer: single-phase k-sampled training with joint (detached) predictor.

Per step:
  encode -> sample one k for the batch -> decode thoughts[:, :k] teacher-forced
  -> CE -> predictor MSE on detached per-sample CE -> (+ KL when kl_beta > 0).

Robustness: non-finite losses skip the step (abort after 20 consecutive),
grad-clip 1.0, full checkpoint/resume (model/optimizer/scheduler/RNG/config),
JSONL metrics, periodic sample reconstructions and mini-val.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config, to_dict
from .generate import greedy_decode
from .ksampler import KSampler
from .losses import kl_divergence, predictor_loss, reconstruction_ce
from .model import PAD_ID, ThoughtAutoencoder, make_padding_mask
from .tokenizer import Tokenizer


def build_optimizer(model: ThoughtAutoencoder, cfg: Config) -> torch.optim.AdamW:
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        if p.ndim < 2 or "embedding" in name or "thought_seed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
    )


def lr_lambda(step: int, cfg: Config) -> float:
    warmup = max(cfg.train.warmup_steps, 1)
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(cfg.train.max_steps - warmup, 1)
    progress = min(progress, 1.0)
    floor = cfg.train.min_lr_frac
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, cfg: Config, model: ThoughtAutoencoder, tokenizer: Tokenizer) -> None:
        # The "high" float32 matmul setting NaN'd at ~20K steps on gfx1031 in
        # the prior project; require the default.
        assert torch.get_float32_matmul_precision() == "highest", (
            "float32 matmul precision must remain 'highest' on this ROCm stack"
        )
        self.cfg = cfg
        self.device = torch.device(cfg.run.device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: lr_lambda(s, cfg)
        )
        self.rng = random.Random(cfg.train.seed)
        self.ksampler = KSampler(cfg.ksampler, cfg.model.num_thoughts, self.rng)
        self.step = 0
        self.best_val = float("inf")
        self.nan_streak = 0

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

    def train_step(self, input_ids: torch.Tensor) -> dict | None:
        cfg = self.cfg
        model = self.model
        input_ids = input_ids.to(self.device, non_blocking=True)
        padding_mask = make_padding_mask(input_ids)
        lengths = (~padding_mask).sum(dim=1).float()
        mean_len = lengths.mean().item()

        use_vae = cfg.reg.kl_beta > 0
        if use_vae:
            thoughts, mu, logvar = model.encoder.encode_with_kl(input_ids, padding_mask)
        else:
            thoughts = model.encode(input_ids, padding_mask)
            mu = logvar = None

        if cfg.reg.noise_std > 0:
            thoughts = thoughts + torch.randn_like(thoughts) * cfg.reg.noise_std
        if cfg.reg.mixup_prob > 0 and self.rng.random() < cfg.reg.mixup_prob:
            lam = self.rng.uniform(0.7, 0.95)
            perm = torch.randperm(thoughts.size(0), device=thoughts.device)
            thoughts = lam * thoughts + (1 - lam) * thoughts[perm]

        k = self.ksampler.sample(mean_len)
        if 0 < cfg.train.detach_encoder_below_k and k < cfg.train.detach_encoder_below_k:
            thoughts_for_dec = thoughts.detach()
        else:
            thoughts_for_dec = thoughts

        dec_in = input_ids[:, :-1]
        dec_tgt = input_ids[:, 1:]
        dec_pad = padding_mask[:, :-1]
        use_nar = cfg.reg.nar or (cfg.reg.nar_frac > 0 and self.rng.random() < cfg.reg.nar_frac)
        if use_nar:
            blank = torch.full_like(dec_in, PAD_ID)
            logits = model.decode(thoughts_for_dec[:, :k], blank, None, causal=False)
        else:
            logits = model.decode(thoughts_for_dec[:, :k], dec_in, dec_pad)
        recon, per_sample = reconstruction_ce(logits, dec_tgt)

        # Full-k anchor: a second decode at k=N keeps top-end reconstruction
        # sharp while the sampled-k path trains compression (matryoshka-style
        # multi-granularity step). Also yields a free predictor label at N.
        anchor = None
        n = cfg.model.num_thoughts
        if cfg.train.anchor_full_k_weight > 0 and k < n:
            a_logits = model.decode(thoughts_for_dec, dec_in, dec_pad)
            anchor, anchor_per_sample = reconstruction_ce(a_logits, dec_tgt)

        # Predictor labels must all be AR CEs (that's what it predicts at
        # inference), so the main term is skipped on NAR batches.
        pred = model.predictor(thoughts.detach())
        p_terms = [] if use_nar else [predictor_loss(pred, k, per_sample)]
        if anchor is not None:
            p_terms.append(predictor_loss(pred, n, anchor_per_sample))
        if cfg.train.predictor_extra_k > 0:
            extra_ks = self.ksampler.sample_distinct(mean_len, cfg.train.predictor_extra_k, k)
            with torch.no_grad():
                for ek in extra_ks:
                    e_logits = model.decode(thoughts[:, :ek].detach(), dec_in, dec_pad)
                    _, e_ps = reconstruction_ce(e_logits, dec_tgt)
                    p_terms.append(predictor_loss(pred, ek, e_ps))
        p_loss = sum(p_terms) / len(p_terms) if p_terms else torch.zeros_like(recon)

        total = recon + cfg.train.predictor_weight * p_loss
        if anchor is not None:
            total = total + cfg.train.anchor_full_k_weight * anchor
        kl = None
        if use_vae:
            beta = cfg.reg.kl_beta * min(1.0, self.step / max(cfg.reg.kl_warmup_steps, 1))
            kl = kl_divergence(mu, logvar)
            total = total + beta * kl

        if not total.isfinite():
            self.nan_streak += 1
            self.optimizer.zero_grad(set_to_none=True)
            if self.nan_streak >= 20:
                raise RuntimeError(f"20 consecutive non-finite losses at step {self.step}")
            return None
        self.nan_streak = 0

        total.backward()
        nn.utils.clip_grad_norm_(self.model.unique_parameters(), cfg.train.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        out = {
            "step": self.step,
            "k": k,
            "recon": recon.item(),
            "pred_mse": p_loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "mean_len": round(mean_len, 1),
        }
        if kl is not None:
            out["kl"] = kl.item()
        return out

    # ----- validation / samples -----

    @torch.no_grad()
    def validate(self, val_loader, max_batches: int = 8) -> float:
        self.model.eval()
        losses = []
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch.to(self.device)
            padding_mask = make_padding_mask(input_ids)
            logits = self.model(input_ids, padding_mask)  # full k
            recon, _ = reconstruction_ce(logits, input_ids[:, 1:])
            losses.append(recon.item())
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    @torch.no_grad()
    def dump_samples(self, input_ids: torch.Tensor) -> None:
        self.model.eval()
        n = self.cfg.model.num_thoughts
        row = input_ids[:1].to(self.device)
        mask = make_padding_mask(row)
        thoughts = self.model.encode(row, mask)
        original = self.tokenizer.decode(row[0].tolist())
        lines = [f"--- step {self.step} ---", f"IN : {original}"]
        for k in sorted({8, n // 4, n}):
            ids = greedy_decode(self.model, thoughts[:, :k], self.cfg.model.max_seq_len)
            lines.append(f"k={k:<3}: {self.tokenizer.decode(ids[0].tolist())}")
        self.samples_file.write("\n".join(lines) + "\n\n")
        self.samples_file.flush()
        self.model.train()

    # ----- checkpointing -----

    def save_checkpoint(self, tag: str | None = None) -> Path:
        name = tag or f"step_{self.step}"
        path = self.run_dir / f"{name}.pt"
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_val": self.best_val,
                "config": to_dict(self.cfg),
                "tokenizer_path": self.cfg.run.tokenizer_path,
                "rng": {
                    "python": self.rng.getstate(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            path,
        )
        if tag is None:
            ckpts = sorted(
                self.run_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])
            )
            for old in ckpts[: -self.cfg.train.keep_ckpts]:
                old.unlink()
        return path

    def load_checkpoint(self, path: str | Path, reset_schedule: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if not reset_schedule:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.step = ckpt["step"]
            self.best_val = ckpt["best_val"]
            self.rng.setstate(ckpt["rng"]["python"])
            torch.set_rng_state(ckpt["rng"]["torch"])
            if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])

    # ----- main loop -----

    def fit(self, train_loader, val_loader) -> None:
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        window: list[float] = []
        data_iter = iter(train_loader)
        sample_batch = None

        while self.step < cfg.train.max_steps:
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
            window.append(metrics["recon"])

            if self.step % cfg.train.log_every == 0:
                rate = cfg.train.log_every / (time.time() - t0)
                t0 = time.time()
                avg = sum(window) / len(window)
                window = []
                metrics["it_s"] = round(rate, 2)
                metrics["recon_avg"] = round(avg, 4)
                if torch.cuda.is_available():
                    metrics["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                print(
                    f"step {self.step:>7} | k={metrics['k']:>3} | recon {avg:.4f} | "
                    f"pred {metrics['pred_mse']:.4f} | lr {metrics['lr']:.2e} | "
                    f"{rate:.1f} it/s",
                    flush=True,
                )
                self.metrics_file.write(json.dumps(metrics) + "\n")
                self.metrics_file.flush()

            if self.step % cfg.train.sample_every == 0:
                self.dump_samples(sample_batch)

            if self.step % cfg.train.val_every == 0:
                val = self.validate(val_loader)
                print(f"step {self.step:>7} | VAL recon {val:.4f}", flush=True)
                self.metrics_file.write(json.dumps({"step": self.step, "val_recon": val}) + "\n")
                self.metrics_file.flush()
                if val < self.best_val:
                    self.best_val = val
                    self.save_checkpoint("best")

            if self.step % cfg.train.ckpt_every == 0:
                self.save_checkpoint()

        self.save_checkpoint("final")
        print(
            f"\nRun '{cfg.run.name}' done: {self.step} steps, best val {self.best_val:.4f}\n"
            f"Paste into RESEARCH_LOG.md:\n"
            f"| {cfg.run.name} | steps={self.step} | bs={cfg.train.batch_size} | "
            f"k-mode={cfg.ksampler.mode} | best val CE {self.best_val:.4f} |",
            flush=True,
        )
