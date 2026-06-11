"""Eval harness, compression-centric.

The objective is compression: k vectors for an L-token text is a k/L ratio,
and what matters is reconstruction quality at LOW ratios plus graceful
degradation (readable English, not word-scrambles) when k is below what exact
reconstruction needs. So the headline tables are ratio-based:

  - CE / unigram-F1 / bigram-F1 at k = ceil(r * length), r in RATIOS.
    Bigram F1 is the order-sensitivity signal: scrambles keep unigram F1 high
    but destroy bigram F1.
  - The runtime lossiness knob: predictor-chosen k = min k with predicted
    CE <= tau, reported as mean chosen ratio + actual CE at the chosen k.

Absolute-k curves (CE-vs-k per length bucket) are kept for continuity with
the original project's tables.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch

from .data import TokenShardDataset, collate
from .generate import greedy_decode
from .losses import reconstruction_ce
from .model import ThoughtAutoencoder, make_padding_mask
from .tokenizer import Tokenizer

BUCKETS = [(0, 20), (20, 50), (50, 80), (80, 129), (129, 257)]
DEFAULT_KS = [2, 4, 8, 16, 32, 64, 128]
RATIOS = [0.125, 0.25, 0.5, 1.0]
TAUS = [0.25, 0.5, 1.0, 2.0]


def _ngram_f1(reference: str, hypothesis: str, n: int = 1) -> float:
    def grams(s: str) -> Counter:
        w = s.lower().split()
        return Counter(zip(*(w[i:] for i in range(n))))

    ref, hyp = grams(reference), grams(hypothesis)
    if not ref or not hyp:
        return 0.0
    overlap = sum((ref & hyp).values())
    p = overlap / sum(hyp.values())
    r = overlap / sum(ref.values())
    return 2 * p * r / (p + r) if p + r else 0.0


def word_overlap_f1(reference: str, hypothesis: str) -> float:
    return _ngram_f1(reference, hypothesis, 1)


def bucket_of(length: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return i
    return len(BUCKETS) - 1


def _ratio_ks(lengths: torch.Tensor, r: float, n: int) -> torch.Tensor:
    return (lengths.float() * r).ceil().long().clamp(2, n)


def _mem_mask(ks: torch.Tensor, n: int) -> torch.Tensor:
    slot = torch.arange(n, device=ks.device)
    return slot[None, :] >= ks[:, None]


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
    nb = len(BUCKETS)

    # --- teacher-forced CE: absolute k AND ratio-based, + predictor knob ---
    ce_sum = {k: [0.0] * nb for k in ks}
    ce_cnt = [0] * nb
    pred_abs_err = {k: 0.0 for k in ks}
    r_ce_sum = {r: [0.0] * nb for r in RATIOS}
    tau_ce_sum = {t: [0.0] * nb for t in TAUS}
    tau_ratio_sum = {t: [0.0] * nb for t in TAUS}
    pred_n = 0

    batch_size = 64
    for s in range(0, len(idxs), batch_size):
        batch = collate([ds[i] for i in idxs[s : s + batch_size]]).to(dev)
        mask = make_padding_mask(batch)
        lengths = (~mask).sum(dim=1)
        buckets = [bucket_of(le) for le in lengths.tolist()]
        thoughts = model.encode(batch, mask)
        pred = model.predictor(thoughts)
        dec_in, dec_tgt, dec_pad = batch[:, :-1], batch[:, 1:], mask[:, :-1]
        for b in buckets:
            ce_cnt[b] += 1

        for k in ks:
            logits = model.decode(thoughts[:, :k], dec_in, dec_pad)
            _, per_sample = reconstruction_ce(logits, dec_tgt)
            for row, b in enumerate(buckets):
                ce_sum[k][b] += per_sample[row].item()
            pred_abs_err[k] += (pred[:, k - 1] - per_sample).abs().sum().item()
        pred_n += batch.size(0)

        for r in RATIOS:
            ks_r = _ratio_ks(lengths, r, n)
            logits = model.decode(
                thoughts, dec_in, dec_pad, memory_padding_mask=_mem_mask(ks_r, n)
            )
            _, per_sample = reconstruction_ce(logits, dec_tgt)
            for row, b in enumerate(buckets):
                r_ce_sum[r][b] += per_sample[row].item()

        # Runtime lossiness knob: smallest k whose PREDICTED CE <= tau.
        for t in TAUS:
            ok = pred <= t  # [B, N]
            any_ok = ok.any(dim=1)
            chosen = torch.where(any_ok, ok.float().argmax(dim=1) + 1, torch.full_like(lengths, n))
            logits = model.decode(
                thoughts, dec_in, dec_pad, memory_padding_mask=_mem_mask(chosen, n)
            )
            _, per_sample = reconstruction_ce(logits, dec_tgt)
            ratio = chosen.float() / lengths.float()
            for row, b in enumerate(buckets):
                tau_ce_sum[t][b] += per_sample[row].item()
                tau_ratio_sum[t][b] += ratio[row].item()

    def _avg(table):
        return {key: [s / c if c else float("nan") for s, c in zip(v, ce_cnt)]
                for key, v in table.items()}

    ce_table = _avg(ce_sum)
    ratio_ce_table = _avg(r_ce_sum)
    tau_ce_table = _avg(tau_ce_sum)
    tau_ratio_table = _avg(tau_ratio_sum)
    calibration = {k: pred_abs_err[k] / max(pred_n, 1) for k in ks}

    # --- greedy decode: unigram + bigram F1 at ratio ks and absolute ks ---
    overlap = {k: [[] for _ in range(nb)] for k in ks}
    exact = {k: [0] * nb for k in ks}
    r_f1 = {r: [[] for _ in range(nb)] for r in RATIOS}
    r_f2 = {r: [[] for _ in range(nb)] for r in RATIOS}
    taken = [0] * nb
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
            overlap[k][b].append(word_overlap_f1(original, recon))
            if recon.strip() == original.strip():
                exact[k][b] += 1
        for r in RATIOS:
            k_r = max(2, min(n, math.ceil(r * length)))
            ids = greedy_decode(model, thoughts[:, :k_r], model.cfg.max_seq_len)
            recon = tokenizer.decode(ids[0].tolist())
            r_f1[r][b].append(_ngram_f1(original, recon, 1))
            r_f2[r][b].append(_ngram_f1(original, recon, 2))
            if len(samples_dump) < 60 and r in (0.125, 0.25, 0.5):
                dump["recon"][f"r={r} (k={k_r})"] = recon
        if dump["recon"]:
            samples_dump.append(dump)

    def _avg_lists(table):
        return {key: [sum(v) / len(v) if v else float("nan") for v in vals]
                for key, vals in table.items()}

    overlap_table = _avg_lists(overlap)
    rf1_table = _avg_lists(r_f1)
    rf2_table = _avg_lists(r_f2)
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
        "ce_by_ratio": ratio_ce_table,
        "f1_by_ratio": rf1_table,
        "bigram_f1_by_ratio": rf2_table,
        "tau_ce": tau_ce_table,
        "tau_chosen_ratio": tau_ratio_table,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))

    bnames = [f"{lo}-{hi}" for lo, hi in BUCKETS]
    lines = ["## Compression (ratio r = k/length)", ""]
    lines.append("| r | " + " | ".join(f"CE {b}" for b in bnames)
                 + " | " + " | ".join(f"F1 {b}" for b in bnames)
                 + " | " + " | ".join(f"biF1 {b}" for b in bnames) + " |")
    lines.append("|" + "---|" * (3 * nb + 1))
    for r in RATIOS:
        cells = (" | ".join(f"{v:.3f}" for v in ratio_ce_table[r]) + " | "
                 + " | ".join(f"{v:.2f}" for v in rf1_table[r]) + " | "
                 + " | ".join(f"{v:.2f}" for v in rf2_table[r]))
        lines.append(f"| {r} | {cells} |")
    lines += ["", "## Lossiness knob (predictor-chosen k, tolerance tau)", ""]
    lines.append("| tau | " + " | ".join(f"CE {b}" for b in bnames)
                 + " | " + " | ".join(f"ratio {b}" for b in bnames) + " |")
    lines.append("|" + "---|" * (2 * nb + 1))
    for t in TAUS:
        cells = (" | ".join(f"{v:.3f}" for v in tau_ce_table[t]) + " | "
                 + " | ".join(f"{v:.2f}" for v in tau_ratio_table[t]))
        lines.append(f"| {t} | {cells} |")
    lines += ["", "## Absolute k (legacy comparison)", ""]
    lines.append("| k | " + " | ".join(f"CE {b}" for b in bnames)
                 + " | " + " | ".join(f"F1 {b}" for b in bnames) + " | pred MAE |")
    lines.append("|" + "---|" * (2 * nb + 2))
    for k in ks:
        cells = (" | ".join(f"{v:.3f}" for v in ce_table[k]) + " | "
                 + " | ".join(f"{v:.2f}" for v in overlap_table[k]))
        lines.append(f"| {k} | {cells} | {calibration[k]:.3f} |")
    table = "\n".join(lines)
    (out_dir / "eval_table.md").write_text(table)

    with open(out_dir / "eval_samples.txt", "w") as f:
        for d in samples_dump:
            f.write(f"[{d['tokens']} tok] IN : {d['text']}\n")
            for label, r in d["recon"].items():
                f.write(f"  {label:<16}: {r}\n")
            f.write("\n")

    print(table)
    return results
