"""Throughput probe: it/s and VRAM for candidate shapes on this GPU.

Synthetic batches, real fwd+bwd+AdamW step (optionally with the anchor's
second decode). Equal-wall-clock ablations are sized from these numbers.

Usage: .venv/bin/python scripts/probe.py [--bf16]
"""

from __future__ import annotations

import argparse
import time

import torch

from thoughtvec.config import ModelCfg
from thoughtvec.losses import reconstruction_ce
from thoughtvec.model import ThoughtAutoencoder, make_padding_mask

SHAPES = [
    # name, d, heads, enc, dec, N, seq, batch_sizes
    ("legacy-256", 256, 4, 4, 4, 128, 128, [32, 64, 128]),
    ("mid-384", 384, 6, 5, 5, 192, 192, [32, 64]),
    ("big-512", 512, 8, 6, 6, 256, 256, [16, 32]),
]


def probe(cfg: ModelCfg, batch: int, steps: int, anchor: bool, bf16: bool) -> tuple[float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = ThoughtAutoencoder(cfg).cuda()
    opt = torch.optim.AdamW(model.unique_parameters(), lr=1e-4)
    ids = torch.randint(4, cfg.vocab_size, (batch, cfg.max_seq_len), device="cuda")
    ids[:, 0] = 1
    mask = make_padding_mask(ids)
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16
        else __import__("contextlib").nullcontext()
    )

    def one_step():
        with ctx:
            thoughts = model.encode(ids, mask)
            k = cfg.num_thoughts // 4
            logits = model.decode(thoughts[:, :k], ids[:, :-1], mask[:, :-1])
            loss, _ = reconstruction_ce(logits, ids[:, 1:])
            if anchor:
                a_logits = model.decode(thoughts, ids[:, :-1], mask[:, :-1])
                a_loss, _ = reconstruction_ce(a_logits, ids[:, 1:])
                loss = loss + 0.5 * a_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    for _ in range(5):
        one_step()
    torch.cuda.synchronize()
    t0 = time.time()
    last = None
    for _ in range(steps):
        last = one_step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    assert last is not None and last.isfinite(), "non-finite loss during probe"
    vram = torch.cuda.max_memory_allocated() / 2**30
    return steps / dt, vram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()

    print("| shape | params | BS | anchor | it/s | tok/s | VRAM GB |")
    print("|---|---|---|---|---|---|---|")
    for name, d, h, e, dec, n, seq, batches in SHAPES:
        cfg = ModelCfg(
            d_model=d, nhead=h, enc_layers=e, dec_layers=dec,
            num_thoughts=n, max_seq_len=seq,
        )
        params = ThoughtAutoencoder(cfg).param_count() / 1e6
        for bs in batches:
            for anchor in (False, True):
                try:
                    rate, vram = probe(cfg, bs, args.steps, anchor, args.bf16)
                    print(
                        f"| {name} | {params:.1f}M | {bs} | {anchor} "
                        f"| {rate:.2f} | {rate * bs * seq:,.0f} | {vram:.2f} |",
                        flush=True,
                    )
                except torch.OutOfMemoryError:
                    print(f"| {name} | {params:.1f}M | {bs} | {anchor} | OOM | — | — |", flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
