#!/usr/bin/env python3
"""Benchmark model architectures for long-text reconstruction (seq up to 384)."""
from __future__ import annotations

import csv, sys, time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thought_vectors import SPTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel
from thought_vectors.data import GroupTextDataset, collate_group_batch
from thought_vectors.data_loading import load_groups_from_path


def build_model(vocab: int, d_model: int, layers: int, heads: int, thoughts: int):
    enc = ThoughtEncoder(vocab, d_model, heads, layers, dropout=0.0, max_seq_len=384, num_thoughts=thoughts)
    dec = ThoughtDecoder(vocab, d_model, heads, layers, dropout=0.0, max_seq_len=384)
    return ThoughtVectorModel(enc, dec)


def count_params(model: nn.Module) -> int:
    seen: set[int] = set()
    return sum(p.numel() for p in model.parameters()
               if p.data_ptr() not in seen and not seen.add(p.data_ptr()))


CONFIGS = [
    ("A (baseline)", 256, 4, 4),
    ("B (wider)",    384, 4, 6),
    ("C (wider+)",   512, 4, 8),
    ("D (deeper+)",  256, 6, 8),
    ("E (heads+)",   256, 4, 8),
]

BATCH_SIZE = 16
STEPS = 5000
LR = 1e-3
THOUGHTS = 256
SEQ_LIMIT = 384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
VOCAB = tok.vocab_size

# Load data once (first 50K texts for speed)
groups = load_groups_from_path(Path("/tmp/c4_500k.csv"))[:50000]
dataset = GroupTextDataset(groups)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate_group_batch(b, tok.encode, tok.pad_token_id),
                    num_workers=0)

results = []
for name, d_model, layers, heads in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Config: {name}  d={d_model} L={layers}+{layers} H={heads}")
    print(f"{'='*60}")

    model = build_model(VOCAB, d_model, layers, heads, THOUGHTS).to(device)
    total = count_params(model)
    print(f"Params: {total:,}")

    seen: set[int] = set()
    opt = torch.optim.AdamW([p for p in model.parameters()
                             if p.data_ptr() not in seen and not seen.add(p.data_ptr())],
                            lr=LR, weight_decay=0)

    loader_iter = iter(loader)
    epoch_loss = 0.0
    t0 = time.time()

    for step in range(STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        ids = batch.to(device)
        if ids.size(1) > SEQ_LIMIT:
            ids = ids[:, :SEQ_LIMIT]

        seq_len = (ids != tok.pad_token_id).sum(dim=1)
        vec_count = torch.clamp((seq_len.float() * 1.5).long(), min=4, max=THOUGHTS)

        pad = ids.eq(tok.pad_token_id)
        thoughts = model.encoder(ids, pad)

        # Use per-example vector count by picking the median k for the batch
        k = int(vec_count.median().item())
        selected = thoughts[:, :k, :]

        logits = model.decoder(selected, ids[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id,
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += float(loss.detach().cpu())

        if (step + 1) % 1000 == 0:
            avg = epoch_loss / (step + 1)
            print(f"  step {step+1:5d}/{STEPS}  loss={avg:.4f}  k={k}")

    final_loss = epoch_loss / STEPS
    results.append((name, d_model, layers, heads, total, final_loss))
    print(f"  Final avg loss: {final_loss:.4f}")

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"{'Config':<20} {'d_model':<8} {'Layers':<8} {'Heads':<8} {'Params':<12} {'Loss':<8}")
print("-" * 64)
for name, d, l, h, p, loss in results:
    print(f"{name:<20} {d:<8} {l:<8} {h:<8} {p:<12,} {loss:<8.4f}")
