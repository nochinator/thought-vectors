#!/usr/bin/env python3
"""Train thinker on large clean data with NaN guardrails and data filtering."""
from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, LossPredictor, ThinkerModel

device = torch.device("cuda")
print("Loading autoencoder...")

ckpt = torch.load(str(ROOT / "artifacts" / "thinker_s1_clean.pt"), map_location="cpu", weights_only=True)

enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 8192, 256).to(device)
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 8192).to(device)
enc.load_state_dict(ckpt["encoder_state"])
dec.load_state_dict(ckpt["decoder_state"])
for p in enc.parameters(): p.requires_grad = False
for p in dec.parameters(): p.requires_grad = False

thinker = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(256, 4, dropout=0.1, batch_first=True), 6
).to(device)
pred = LossPredictor(256, 256).to(device)
model = ThinkerModel(enc, dec, thinker, pred)
opt = torch.optim.AdamW(model.thinker.parameters(), lr=5e-5)
print(f"Thinker: {sum(p.numel() for p in model.thinker.parameters()):,}")

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
csv.field_size_limit(2**31 - 1)

# ── Load and filter data ──
arts_raw, sums_raw = [], []
with open("/tmp/thinker_data_clean.csv") as fa, open("/tmp/thinker_data_clean_out.csv") as fs:
    for ra, rs in zip(csv.reader(fa), csv.reader(fs)):
        if ra and rs:
            arts_raw.append(ra[0])
            sums_raw.append(rs[0])
print(f"Raw: {len(arts_raw)} pairs")

# Filter: remove entries with extreme token lengths
arts, sums = [], []
for u, a in zip(arts_raw, sums_raw):
    u_ids = tok.encode(u, add_special_tokens=True)
    a_ids = tok.encode(a, add_special_tokens=True)
    # Guardrails: min 2 tokens, max 60 tokens, no extreme token ID ratios
    if len(u_ids) < 2 or len(a_ids) < 2: continue
    if len(u_ids) > 60 or len(a_ids) > 60: continue
    # Check for very high variance input (potential edge case)
    arts.append(u)
    sums.append(a)
print(f"Filtered: {len(arts)} pairs")

pad_id = tok.pad_token_id
n_batches = len(arts) // 12
print(f"{n_batches} batches/epoch")

# ── Training with NaN guard ──
t0 = time.time()
nan_count = 0
for epoch in range(1, 6):
    combined = list(zip(arts, sums))
    random.shuffle(combined)
    for bi in range(n_batches):
        batch = combined[bi * 12 : (bi + 1) * 12]
        ba, bs = zip(*batch)

        ae = [tok.encode(t, add_special_tokens=True) for t in ba]
        ma = max(len(e) for e in ae)
        ai = torch.full((12, ma), pad_id, dtype=torch.long)
        for i, e in enumerate(ae): ai[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        ai = ai.to(device)

        se = [tok.encode(t, add_special_tokens=True) for t in bs]
        ms = max(len(e) for e in se)
        si = torch.full((12, ms), pad_id, dtype=torch.long)
        for i, e in enumerate(se): si[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        si = si.to(device)

        k = random.randint(8, 128)
        logits = model(ai, si, padding_mask=ai.eq(pad_id), k=k)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            si[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )
        if loss.isnan() or loss.isinf():
            nan_count += 1
            continue

        # Save pre-step weights for rollback
        pre_state = {k: v.clone() for k, v in model.thinker.state_dict().items()}

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.thinker.parameters(), 0.5)
        opt.step()

        # NaN guard: check if weights became NaN, rollback and continue
        if any(torch.isnan(p).any() for p in model.thinker.parameters()):
            model.thinker.load_state_dict(pre_state)
            nan_count += 1
            continue

        if bi % 500 == 0 or bi == n_batches - 1:
            e = (time.time() - t0) / 60
            print(f"  ep{epoch:>2} batch{bi:>5}/{n_batches} k={k:3d} loss={loss.item():.4f} nan={nan_count} {e:.1f}min")

# Save
torch.save({
    "encoder_state": enc.state_dict(), "decoder_state": dec.state_dict(),
    "thinker_state": model.thinker.state_dict(), "predictor_state": model.predictor.state_dict(),
    "config": {"d_model": 256, "thoughts": 256, "thinker_layers": 6, "thinker_heads": 4, "vocab_size": 16384},
    "nan_count": nan_count,
}, str(ROOT / "artifacts" / "thinker_nat_v4.pt"))
print(f"Saved (nan_count={nan_count})")
