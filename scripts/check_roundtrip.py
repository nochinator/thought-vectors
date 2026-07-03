"""Round-trip integrity guard for decoder-unfreeze runs.

nochi's invariant (2026-06-24): text -> encode -> thoughts -> decode -> original
text MUST always hold. The UNFREEZE arm adapts the decoder to the thinker's
predicted thoughts; this checks it did NOT degrade the codec's own prose
reconstruction. Compares reconstruction CE of the MODIFIED codec (embedded in
the thinker ckpt) vs the ORIGINAL frozen codec, on a held-out compression shard,
at several thought-prefix lengths.

VERDICT FAIL (nonzero exit) if the modified codec regresses beyond --tol at any
prefix => that arm is DISQUALIFIED from the 12h stack.

Usage: .venv/bin/python scripts/check_roundtrip.py \
         --ckpt checkpoints/R4_UNFREEZE/best.pt --shard data/mix_uni_val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch

from thoughtvec.config import from_dict
from thoughtvec.data import make_loader
from thoughtvec.losses import reconstruction_ce
from thoughtvec.model import ThoughtAutoencoder, make_padding_mask


def _recon_ce(codec, ids, mask, k):
    th = codec.encode(ids, mask)
    logits = codec.decode(th[:, :k], ids[:, :-1], mask[:, :-1])
    ce, _ = reconstruction_ce(logits, ids[:, 1:])
    return ce.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="thinker ckpt with embedded codec")
    ap.add_argument("--shard", default="data/mix_uni_val")
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed recon-CE regression (nats) before FAIL")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "codec" not in ckpt:
        print(json.dumps({"verdict": "N/A", "reason": "no embedded codec (frozen run)"}))
        return
    frozen_state = torch.load(ckpt["codec_ckpt"], map_location="cpu", weights_only=False)
    codec_cfg = from_dict(frozen_state["config"])
    nT = codec_cfg.model.num_thoughts

    frozen = ThoughtAutoencoder(codec_cfg.model)
    frozen.load_state_dict(frozen_state["model"])
    frozen.to(dev).eval()
    modified = ThoughtAutoencoder(codec_cfg.model)
    modified.load_state_dict(ckpt["codec"])
    modified.to(dev).eval()

    loader = make_loader(args.shard, args.batch_size, shuffle=False, num_workers=0)
    prefixes = sorted({2, 8, nT // 2, nT})
    fsum = {k: 0.0 for k in prefixes}
    msum = {k: 0.0 for k in prefixes}
    n = 0
    with torch.no_grad():
        for i, ids in enumerate(loader):
            if i >= args.batches:
                break
            ids = ids.to(dev)
            mask = make_padding_mask(ids)
            for k in prefixes:
                fsum[k] += _recon_ce(frozen, ids, mask, k)
                msum[k] += _recon_ce(modified, ids, mask, k)
            n += 1

    rows = []
    worst = -1e9
    for k in prefixes:
        f, m = fsum[k] / n, msum[k] / n
        d = m - f
        worst = max(worst, d)
        rows.append({"k": k, "frozen_ce": round(f, 4), "modified_ce": round(m, 4),
                     "delta": round(d, 4)})
    verdict = "PASS" if worst <= args.tol else "FAIL"
    print(json.dumps({"ckpt": args.ckpt, "verdict": verdict, "worst_delta": round(worst, 4),
                      "tol": args.tol, "per_prefix": rows}, indent=2))
    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
