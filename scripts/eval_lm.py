"""Rank Round B baseline-LM checkpoints: decoded-reply quality on the
dialogue val shard, mirroring eval_thinker.py so the B3 comparison table
reads off the same measurements.

Metrics (same definitions as eval_thinker.py):
  val_ce      — teacher-forced CE on response positions (the training metric)
  ref_f1      — unigram F1 of greedy reply vs reference (weak signal: open-ended)
  distinct1/2 — corpus-level distinct n-grams over decoded replies
  len_ratio   — mean decoded length / mean reference length

Data view matches LMTrainer: DialogueDataset(flat_context=True), context
trimmed from the left to leave room for the reply.

Usage: .venv/bin/python scripts/eval_lm.py --ckpt checkpoints/b3_lm_48m_24h/best.pt \
           [--shard data/dialogue_val] [--rows 1024] [--dump samples.txt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from thoughtvec.config import from_dict  # noqa: E402
from thoughtvec.data import DialogueDataset  # noqa: E402
from thoughtvec.lm import TokenLM, collate_lm, lm_ce  # noqa: E402
from thoughtvec.tokenizer import Tokenizer  # noqa: E402


def unigram_f1(pred: list[int], ref: list[int]) -> float:
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred), overlap / len(ref)
    return 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", default="data/dialogue_val")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", default=None, help="write decoded samples here")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["cfg"])
    model = TokenLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    tok = Tokenizer(cfg.run.tokenizer_path)

    ds = DialogueDataset(args.shard, max_context=cfg.lm.max_turns,
                         flat_context=True,
                         max_flat_tokens=cfg.lm.max_seq_len - 128)
    room = cfg.lm.max_seq_len - args.max_new - 1

    ce_sum, f1_sum, n_rows = 0.0, 0.0, 0
    pred_len_sum, ref_len_sum = 0, 0
    grams1: Counter = Counter()
    grams2: Counter = Counter()
    tot1 = tot2 = 0
    dump_lines: list[str] = []

    with torch.no_grad():
        for i in range(min(args.rows, len(ds))):
            row = ds[i]
            ctx, resp = row["context"][0], row["response"]
            batch = collate_lm([row], cfg.lm.max_seq_len)
            ce_sum += lm_ce(model, batch["ids"].to(dev),
                            batch["loss_mask"].to(dev)).item()
            hyp = model.generate(ctx[-room:].to(dev), max_new=args.max_new,
                                 temperature=0.0)
            ref = [x for x in resp.tolist() if x > 2]
            hyp = [x for x in hyp if x > 2]
            f1_sum += unigram_f1(hyp, ref)
            pred_len_sum += len(hyp)
            ref_len_sum += len(ref)
            grams1.update(hyp)
            grams2.update(zip(hyp, hyp[1:]))
            tot1 += len(hyp)
            tot2 += max(len(hyp) - 1, 0)
            n_rows += 1
            if args.dump and len(dump_lines) < 200:
                dump_lines.append(f"ctx : {tok.decode([x for x in ctx.tolist() if x > 2])}")
                dump_lines.append(f"REF : {tok.decode(ref)}")
                dump_lines.append(f"PRED: {tok.decode(hyp)}")
                dump_lines.append("")

    out = {
        "ckpt": args.ckpt,
        "val_ce": round(ce_sum / n_rows, 4),
        "ref_f1": round(f1_sum / n_rows, 4),
        "distinct1": round(len(grams1) / max(tot1, 1), 4),
        "distinct2": round(len(grams2) / max(tot2, 1), 4),
        "len_ratio": round(pred_len_sum / max(ref_len_sum, 1), 3),
        "rows": n_rows,
    }
    print(json.dumps(out))
    if args.dump:
        with open(args.dump, "w") as f:
            f.write("\n".join(dump_lines))


if __name__ == "__main__":
    main()
