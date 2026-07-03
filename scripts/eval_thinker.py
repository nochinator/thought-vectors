"""Rank thinker checkpoints: thought-space fit + decoded-reply quality.

Metrics on the dialogue val shard:
  val_cos     — cosine(pred, target thoughts), the training proxy
  val_dec_ce  — teacher-forced CE of the reference reply from predicted thoughts
  ref_f1      — unigram F1 of greedy reply vs reference (weak signal: open-ended)
  distinct1/2 — corpus-level distinct n-grams over decoded replies (degeneracy:
                a thinker that always says "I don't know" scores ~0)
  len_ratio   — mean decoded length / mean reference length

Usage: .venv/bin/python scripts/eval_thinker.py --ckpt checkpoints/ab_T0/best.pt \
           [--shard data/dialogue_val] [--batches 32] [--dump samples.txt]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
import torch.nn.functional as F

from thoughtvec.config import from_dict
from thoughtvec.generate import sample_decode
from thoughtvec.losses import reconstruction_ce
from thoughtvec.model import ThoughtAutoencoder, make_padding_mask
from thoughtvec.thinker import Thinker
from thoughtvec.thinker_train import encode_turns, make_dialogue_loader, out_budget_mask
from thoughtvec.tokenizer import Tokenizer


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
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", default=None, help="write decoded samples here")
    ap.add_argument("--out-tau", type=float, default=None,
                    help="override thinker.out_tau at eval (adaptive response length; inference-only)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)  # reproducible hypothesis sampling for WTA ckpts

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    tk = cfg.thinker
    if args.out_tau is not None:  # sweep the adaptive-length dial without retraining
        tk.out_tau = args.out_tau
    codec_state = torch.load(ckpt["codec_ckpt"], map_location="cpu", weights_only=False)
    codec_cfg = from_dict(codec_state["config"])
    codec = ThoughtAutoencoder(codec_cfg.model)
    codec.load_state_dict(ckpt.get("codec", codec_state["model"]))
    codec.to(dev).eval()
    thinker = Thinker(tk, codec_cfg.model.d_model)
    thinker.load_state_dict(ckpt["thinker"])
    thinker.to(dev).eval()
    tok = Tokenizer(cfg.run.tokenizer_path)

    loader = make_dialogue_loader(args.shard, args.batch_size, tk.max_turns,
                                  shuffle=False, num_workers=0,
                                  flat_context=tk.flat_context,
                                  max_flat_tokens=codec_cfg.model.max_seq_len)
    cos_sum, ce_sum, f1_sum, n_rows, n_batches = 0.0, 0.0, 0.0, 0, 0
    pred_len_sum, ref_len_sum = 0, 0
    budget_sum = 0  # mean response vectors actually decoded (adaptive out_tau)
    grams1: Counter = Counter()
    grams2: Counter = Counter()
    tot1 = tot2 = 0
    dump_lines: list[str] = []

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
            if pred.dim() == 4:  # WTA head: score a RANDOM hypothesis (honest diversity)
                rows = torch.arange(pred.size(0), device=dev)
                pred = pred[rows, torch.randint(pred.size(1), (pred.size(0),), device=dev)]
            cos_sum += F.cosine_similarity(pred, tgt_th, dim=-1).mean().item()
            # adaptive response length: decode each reply from the smallest
            # importance-ordered prefix the predictor deems good enough
            mem_mask = None
            if tk.out_tau > 0:
                budgets, mem_mask = out_budget_mask(codec, pred, tk.out_tau)
                budget_sum += int(budgets.sum().item())
            else:
                budget_sum += pred.size(0) * pred.size(1)
            resp_pad = make_padding_mask(resp_ids)
            logits = codec.decode(pred, resp_ids[:, :-1], resp_pad[:, :-1],
                                  memory_padding_mask=mem_mask)
            ce, _ = reconstruction_ce(logits, resp_ids[:, 1:])
            ce_sum += ce.item()
            n_batches += 1

            ids = sample_decode(codec, pred, codec_cfg.model.max_seq_len,
                                temperature=0.0, no_repeat_ngram=3,
                                memory_padding_mask=mem_mask)
            for row in range(ids.size(0)):
                hyp = [x for x in ids[row].tolist() if x > 2]
                ref = [x for x in resp_ids[row].tolist() if x > 2]
                f1_sum += unigram_f1(hyp, ref)
                pred_len_sum += len(hyp)
                ref_len_sum += len(ref)
                grams1.update(hyp)
                grams2.update(zip(hyp, hyp[1:]))
                tot1 += len(hyp)
                tot2 += max(len(hyp) - 1, 0)
                n_rows += 1
                if args.dump and len(dump_lines) < 200:
                    nturn = int(batch["ctx_turns"][row])
                    for j in range(nturn):
                        t = [x for x in batch["ctx_ids"][row, j].tolist() if x > 2]
                        who = "user" if batch["ctx_roles"][row, j] == 0 else "bot"
                        dump_lines.append(f"{who}: {tok.decode(t)}")
                    dump_lines.append(f"REF : {tok.decode(ref)}")
                    dump_lines.append(f"PRED: {tok.decode(hyp)}")
                    dump_lines.append("")

    out = {
        "ckpt": args.ckpt,
        "val_cos": round(cos_sum / n_batches, 4),
        "val_dec_ce": round(ce_sum / n_batches, 4),
        "ref_f1": round(f1_sum / n_rows, 4),
        "distinct1": round(len(grams1) / max(tot1, 1), 4),
        "distinct2": round(len(grams2) / max(tot2, 1), 4),
        "len_ratio": round(pred_len_sum / max(ref_len_sum, 1), 3),
        "out_vectors": round(budget_sum / max(n_rows, 1), 2),  # mean response vectors decoded (k_out if out_tau=0)
        "rows": n_rows,
    }
    print(json.dumps(out))
    if args.dump:
        with open(args.dump, "w") as f:
            f.write("\n".join(dump_lines))


if __name__ == "__main__":
    main()
