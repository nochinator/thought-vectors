"""Eval harness: CE-vs-k curves, word-overlap F1, predictor calibration."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from .data import TokenShardDataset, collate
from .generate import greedy_decode
from .losses import reconstruction_ce
from .model import ThoughtAutoencoder, make_padding_mask
from .tokenizer import Tokenizer

BUCKETS = [(0, 20), (20, 50), (50, 80), (80, 129)]
DEFAULT_KS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def word_overlap_f1(reference: str, hypothesis: str) -> float:
    ref = Counter(reference.lower().split())
    hyp = Counter(hypothesis.lower().split())
    if not ref or not hyp:
        return 0.0
    overlap = sum((ref & hyp).values())
    precision = overlap / sum(hyp.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bucket_of(length: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return i
    return len(BUCKETS) - 1


@torch.no_grad()
def evaluate(
    model: ThoughtAutoencoder,
    tokenizer: Tokenizer,
    shard_dir: str | Path,
    out_dir: str | Path,
    max_texts: int = 2000,
    decode_per_bucket: int = 200,
    ks: list[int] | None = None,
    device: str = "cuda",
) -> dict:
    model.eval()
    dev = torch.device(device)
    n = model.cfg.num_thoughts
    ks = [k for k in (ks or DEFAULT_KS) if k <= n]
    if ks[-1] != n:
        ks.append(n)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TokenShardDataset(shard_dir)
    idxs = list(range(min(len(ds), max_texts)))

    # --- teacher-forced CE per k, bucketed, + predictor calibration ---
    ce_sum = {k: [0.0] * len(BUCKETS) for k in ks}
    ce_cnt = [0] * len(BUCKETS)
    pred_abs_err = {k: 0.0 for k in ks}
    pred_n = 0

    batch_size = 64
    for s in range(0, len(idxs), batch_size):
        batch = collate([ds[i] for i in idxs[s : s + batch_size]]).to(dev)
        mask = make_padding_mask(batch)
        lengths = (~mask).sum(dim=1)
        thoughts = model.encode(batch, mask)
        pred = model.predictor(thoughts)
        dec_in, dec_tgt, dec_pad = batch[:, :-1], batch[:, 1:], mask[:, :-1]
        for row, length in enumerate(lengths.tolist()):
            ce_cnt[bucket_of(length)] += 1
        for k in ks:
            logits = model.decode(thoughts[:, :k], dec_in, dec_pad)
            _, per_sample = reconstruction_ce(logits, dec_tgt)
            for row, length in enumerate(lengths.tolist()):
                ce_sum[k][bucket_of(length)] += per_sample[row].item()
            pred_abs_err[k] += (pred[:, k - 1] - per_sample).abs().sum().item()
        pred_n += batch.size(0)

    ce_table = {
        k: [s / c if c else float("nan") for s, c in zip(ce_sum[k], ce_cnt)] for k in ks
    }
    calibration = {k: pred_abs_err[k] / max(pred_n, 1) for k in ks}

    # --- greedy decode word-overlap per bucket (subsample) ---
    overlap: dict[int, list[list[float]]] = {k: [[] for _ in BUCKETS] for k in ks}
    exact: dict[int, list[int]] = {k: [0] * len(BUCKETS) for k in ks}
    taken = [0] * len(BUCKETS)
    samples_dump = []
    for i in idxs:
        row = ds[i].unsqueeze(0).to(dev)
        length = row.size(1)
        b = bucket_of(length)
        if taken[b] >= decode_per_bucket:
            continue
        taken[b] += 1
        mask = make_padding_mask(row)
        thoughts = model.encode(row, mask)
        original = tokenizer.decode(row[0].tolist())
        dump = {"text": original, "tokens": length, "recon": {}}
        for k in ks:
            ids = greedy_decode(model, thoughts[:, :k], model.cfg.max_seq_len)
            recon = tokenizer.decode(ids[0].tolist())
            f1 = word_overlap_f1(original, recon)
            overlap[k][b].append(f1)
            if recon.strip() == original.strip():
                exact[k][b] += 1
            if k in (8, 16, n) and len(samples_dump) < 40:
                dump["recon"][k] = recon
        if dump["recon"]:
            samples_dump.append(dump)

    overlap_table = {
        k: [sum(v) / len(v) if v else float("nan") for v in overlap[k]] for k in ks
    }
    exact_table = {
        k: [e / t if t else float("nan") for e, t in zip(exact[k], taken)] for k in ks
    }

    results = {
        "buckets": [f"{lo}-{hi}" for lo, hi in BUCKETS],
        "bucket_counts_ce": ce_cnt,
        "bucket_counts_decode": taken,
        "ce_by_k": ce_table,
        "overlap_f1_by_k": overlap_table,
        "exact_match_by_k": exact_table,
        "predictor_mae_by_k": calibration,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))

    lines = ["| k | " + " | ".join(f"CE {b[0]}-{b[1]}" for b in BUCKETS) + " | "
             + " | ".join(f"F1 {b[0]}-{b[1]}" for b in BUCKETS) + " | pred MAE |"]
    lines.append("|" + "---|" * (2 * len(BUCKETS) + 2))
    for k in ks:
        ce_cells = " | ".join(f"{v:.3f}" for v in ce_table[k])
        f1_cells = " | ".join(f"{v:.2f}" for v in overlap_table[k])
        lines.append(f"| {k} | {ce_cells} | {f1_cells} | {calibration[k]:.3f} |")
    table = "\n".join(lines)
    (out_dir / "eval_table.md").write_text(table)

    with open(out_dir / "eval_samples.txt", "w") as f:
        for d in samples_dump:
            f.write(f"[{d['tokens']} tok] IN : {d['text']}\n")
            for k, r in d["recon"].items():
                f.write(f"        k={k:<3}: {r}\n")
            f.write("\n")

    print(table)
    return results
