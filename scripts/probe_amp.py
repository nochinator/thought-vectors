"""Verify fp16 autocast on the frozen m5_frontier codec: VRAM delta + accuracy.

Runs real encode/decode passes at batch sizes 8..32, measures peak VRAM in fp32
vs fp16 mode, and checks numerical drift on logits + CE. The codec weights stay
fp32 throughout — autocast only affects intermediate activations.
"""

from __future__ import annotations

import time

import torch

from thoughtvec.config import from_dict
from thoughtvec.data import TokenShardDataset, collate
from thoughtvec.losses import reconstruction_ce
from thoughtvec.model import ThoughtAutoencoder, make_padding_mask

CODECK = "checkpoints/m5_frontier/best.pt"
SHARD = "data/mix_uni_val"


def _load_codec(device: str) -> ThoughtAutoencoder:
    ckpt = torch.load(CODECK, map_location="cpu", weights_only=False)
    codec = ThoughtAutoencoder(from_dict(ckpt["config"]).model)
    codec.load_state_dict(ckpt["model"])
    codec = codec.to(device)
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    print(f"codec params: {codec.param_count() / 1e6:.1f}M")
    return codec


def _run_pass(
    codec: ThoughtAutoencoder,
    ids: torch.Tensor,
    mask: torch.Tensor,
    use_amp: bool,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """One encode+decode pass, returning (peak_vram_gb, elapsed_s, logits, ce)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else __import__("contextlib").nullcontext()
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        with amp_ctx:
            th = codec.encode(ids, mask)
            logits = codec.decode(th, ids[:, :-1], mask[:, :-1])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 2**30
    ce, _ = reconstruction_ce(logits.detach(), ids[:, 1:])
    return vram, elapsed, logits, ce


def _collate_batch(ds: TokenShardDataset, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = collate([ds[i] for i in range(bs)])
    ids = batch.to("cuda")
    mask = make_padding_mask(ids)
    return ids, mask


def main() -> None:
    device = "cuda"
    codec = _load_codec(device)
    ds = TokenShardDataset(SHARD)

    print()
    print(f"{'BS':>3} │ {'fp32 VRAM':>9} │ {'fp16 VRAM':>9} │ {'saved':>7} │ "
          f"{'CE drift':>9} │ {'logit ratio':>10} │ {'fp32 ms':>8} │ {'fp16 ms':>8}")
    print("─" * 90)

    for bs in [8, 16, 24, 32]:
        ids, mask = _collate_batch(ds, bs)

        vram_fp32, t_fp32, log_fp32, ce_fp32 = _run_pass(codec, ids, mask, False)
        vram_fp16, t_fp16, log_fp16, ce_fp16 = _run_pass(codec, ids, mask, True)

        saved = vram_fp32 - vram_fp16
        ce_drift = (ce_fp16 - ce_fp32).abs().item()

        # Verify logits aren't NaN'd by fp16 softmax overflow
        fp32_mean = log_fp32.mean().item()
        fp16_mean = log_fp16.mean().item()
        ratio = fp16_mean / fp32_mean if fp32_mean != 0 else float("nan")
        has_inf = torch.isinf(log_fp16).any().item()

        flag = ""
        if has_inf:
            flag += " INF"
        if ce_drift > 0.05:
            flag += " DRIFT"

        print(
            f"{bs:>3} │ {vram_fp32:>7.2f} GB │ {vram_fp16:>7.2f} GB "
            f"│ {saved:>5.2f} GB │ {ce_drift:>9.6f} │ {ratio:>10.6f} "
            f"│ {t_fp32*1000:>6.1f}ms │ {t_fp16*1000:>6.1f}ms{flag}",
            flush=True,
        )

    # Also verify: does gradient flow through autocast work?
    print("\n— gradient-through-autocast check (encode + decode, cat. CE → thinker-like grad) —")
    codec_amp = _load_codec(device)
    ids, mask = _collate_batch(ds, 8)

    # Set up a tiny thinker-like param to receive the gradient
    test_param = torch.nn.Parameter(
        torch.randn(8, codec.cfg.num_thoughts, codec.cfg.d_model, device=device) * 0.02
    )
    opt = torch.optim.AdamW([test_param], lr=1e-4)

    for _ in range(3):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            th = codec.encode(ids, mask)
            # Simulate thinker output: slightly perturb the real thoughts
            th_pred = th.detach() + test_param
            logits = codec.decode(th_pred, ids[:, :-1], mask[:, :-1])
            ce, _ = reconstruction_ce(logits, ids[:, 1:])
        ce.backward()
        g = test_param.grad
        ok = g is not None and g.isfinite().all()
        print(f"  iter {_}: grad finite={ok.item()}, grad_std={g.std().item():.6f}, ce={ce.item():.4f}")
        if not ok:
            print("  FAILED — gradient through fp16 autocast produced NaN!")
            return
    print("  gradient-through-autocast PASSED\n")


if __name__ == "__main__":
    main()
