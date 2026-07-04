"""Continue skewed-ratio compression training at batch=8."""
from __future__ import annotations
import sys
sys.path.insert(0, "thought-vectors-main")
from pathlib import Path
import torch

ckpt = torch.load("thought-vectors-main/artifacts/vae_compressed_nar.pt", map_location="cpu", weights_only=True)
cfg = ckpt["config"]
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel
tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")

enc = ThoughtEncoder(cfg["vocab_size"], cfg["d_model"], cfg["heads"],
    cfg["encoder_layers"], cfg["dropout"], cfg["max_seq_len"], cfg["num_thoughts"])
dec = ThoughtDecoder(cfg["vocab_size"], cfg["d_model"], cfg["heads"],
    cfg["decoder_layers"], cfg["dropout"], cfg["max_seq_len"])
model = ThoughtVectorModel(enc, dec)
model.load_state_dict(ckpt["model_state"], strict=False)

# Count unique params
seen_ptrs: set[int] = set()
unique_params = sum(p.numel() for p in model.parameters()
    if p.data_ptr() not in seen_ptrs and not seen_ptrs.add(p.data_ptr()))
print(f"Model: {unique_params:,} params")
print(f"Checkpoint loss: N/A")
