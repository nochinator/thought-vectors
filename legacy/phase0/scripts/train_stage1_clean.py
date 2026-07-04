#!/usr/bin/env python3
"""Stage 1: train the thinker on clean conversation data (LMSYS + OASST only)."""
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
print("Loading checkpoint...")

ckpt = torch.load("artifacts/thinker_s0_clean.pt", map_location="cpu", weights_only=True)
enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 8192, 256).to(device)
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 8192).to(device)
thinker = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(256, 4, dropout=0.1, batch_first=True), 6
).to(device)
pred = LossPredictor(256, 256).to(device)

enc.load_state_dict(ckpt["encoder_state"])
dec.load_state_dict(ckpt["decoder_state"])
thinker.load_state_dict(ckpt["thinker_state"])
pred.load_state_dict(ckpt["predictor_state"])

for p in enc.parameters(): p.requires_grad = False
for p in dec.parameters(): p.requires_grad = False

model = ThinkerModel(enc, dec, thinker, pred)
opt = torch.optim.AdamW(model.thinker.parameters(), lr=1e-4)

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
csv.field_size_limit(2**31 - 1)

arts, sums = [], []
with open("/tmp/clean_conv_input.csv") as fa, open("/tmp/clean_conv_output.csv") as fs:
    for ra, rs in zip(csv.reader(fa), csv.reader(fs)):
        if ra and rs:
            arts.append(ra[0])
            sums.append(rs[0])

print(f"Loaded {len(arts)} pairs")
pad_id = tok.pad_token_id
n_batches = len(arts) // 16
print(f"{n_batches} batches/epoch")

t0 = time.time()
for epoch in range(1, 11):
    combined = list(zip(arts, sums))
    random.shuffle(combined)
    for bi in range(n_batches):
        batch = combined[bi * 16 : (bi + 1) * 16]
        ba, bs = zip(*batch)
        ae = [tok.encode(t, add_special_tokens=True) for t in ba]
        ma = max(len(e) for e in ae)
        ai = torch.full((16, ma), pad_id, dtype=torch.long)
        for i, e in enumerate(ae):
            ai[i, : len(e)] = torch.tensor(e, dtype=torch.long)
        ai = ai.to(device)
        se = [tok.encode(t, add_special_tokens=True) for t in bs]
        ms = max(len(e) for e in se)
        si = torch.full((16, ms), pad_id, dtype=torch.long)
        for i, e in enumerate(se):
            si[i, : len(e)] = torch.tensor(e, dtype=torch.long)
        si = si.to(device)
        k = random.randint(4, 256)
        logits = model(ai, si, padding_mask=ai.eq(pad_id), k=k)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            si[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )
        if loss.isnan() or loss.isinf():
            continue
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.thinker.parameters(), 1.0)
        opt.step()
        if bi % 200 == 0 or bi == n_batches - 1:
            e = (time.time() - t0) / 60
            print(f"  ep{epoch} batch{bi:>4}/{n_batches} k={k:3d} loss={loss.item():.4f} {e:.1f}min")

torch.save(
    {
        "encoder_state": enc.state_dict(),
        "decoder_state": dec.state_dict(),
        "thinker_state": model.thinker.state_dict(),
        "predictor_state": pred.state_dict(),
        "config": {
            "d_model": 256,
            "thoughts": 256,
            "thinker_layers": 6,
            "thinker_heads": 4,
            "vocab_size": 16384,
        },
    },
    "artifacts/thinker_s1_clean.pt",
)
print("Saved thinker_s1_clean.pt")
