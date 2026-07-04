"""Diagnostic: effective output rank — how many thought vectors actually carry
content, true reply vs thinker prediction.

Runs the frozen codec's loss predictor (the same monotone head that drives the
tau dial) on (a) the TRUE encoded reply thoughts and (b) the thinker's PREDICTED
thoughts, and reports the mean prefix length each needs to clear a CE bar.

The gap is the collapse: if true replies need ~8 vectors but the thinker's
predictions "need" only ~3, the thinker is under-filling its output — putting
real content in the first few thoughts and near-empty refinements after. That is
why adaptive out_tau truncation skips the 4-8 range (it reads the prediction's
collapse, not the reply's true info content).

Usage: .venv/bin/python scripts/diag_output_rank.py --ckpt checkpoints/R3_KRP/best.pt
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch

from thoughtvec.config import from_dict
from thoughtvec.model import ThoughtAutoencoder
from thoughtvec.thinker import Thinker
from thoughtvec.thinker_train import encode_turns, make_dialogue_loader, out_budget_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", default="data/dialogue_val")
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--taus", default="0.1,0.2,0.3,0.5")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    tk = cfg.thinker
    codec_state = torch.load(ckpt["codec_ckpt"], map_location="cpu", weights_only=False)
    codec_cfg = from_dict(codec_state["config"])
    codec = ThoughtAutoencoder(codec_cfg.model)
    codec.load_state_dict(ckpt.get("codec", codec_state["model"]))
    codec.to(dev).eval()
    thinker = Thinker(tk, codec_cfg.model.d_model)
    thinker.load_state_dict(ckpt["thinker"])
    thinker.to(dev).eval()

    loader = make_dialogue_loader(args.shard, args.batch_size, tk.max_turns,
                                  shuffle=False, num_workers=0,
                                  flat_context=tk.flat_context,
                                  max_flat_tokens=codec_cfg.model.max_seq_len)
    taus = [float(t) for t in args.taus.split(",")]
    true_sum = {t: 0 for t in taus}
    pred_sum = {t: 0 for t in taus}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            ctx_ids = batch["ctx_ids"].to(dev)
            resp_ids = batch["resp_ids"].to(dev)
            ctx_th, budgets = encode_turns(codec, ctx_ids, tk.k_ctx, tau=tk.ctx_tau)
            tgt_th, _ = encode_turns(codec, resp_ids, tk.k_out)
            pred = thinker(ctx_th, batch["ctx_roles"].to(dev),
                           batch["ctx_turns"].to(dev), batch["resp_roles"].to(dev),
                           slot_budgets=budgets)
            if pred.dim() == 4:
                rows = torch.arange(pred.size(0), device=dev)
                pred = pred[rows, torch.randint(pred.size(1), (pred.size(0),), device=dev)]
            for t in taus:
                tb, _ = out_budget_mask(codec, tgt_th, t)
                pb, _ = out_budget_mask(codec, pred, t)
                true_sum[t] += int(tb.sum().item())
                pred_sum[t] += int(pb.sum().item())
            n += pred.size(0)

    out = {
        "ckpt": args.ckpt,
        "k_out": tk.k_out,
        "rows": n,
        # mean vectors the codec predictor deems sufficient, per CE bar
        "true_reply_rank": {str(t): round(true_sum[t] / n, 2) for t in taus},
        "thinker_pred_rank": {str(t): round(pred_sum[t] / n, 2) for t in taus},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
