"""Demonstrate finding minimum k using the loss predictor."""
from __future__ import annotations
import sys, math
sys.path.insert(0, "thought-vectors-main")

import torch
import torch.nn.functional as F
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel

# Load model + predictor from a compression checkpoint
base = "thought-vectors-main/artifacts"
ckpt = torch.load(f"{base}/vae_compressed_blend.pt", map_location="cpu", weights_only=True)
cfg = ckpt["config"]

enc = ThoughtEncoder(cfg["vocab_size"], cfg["d_model"], cfg["heads"],
    cfg["encoder_layers"], cfg["dropout"], cfg["max_seq_len"], cfg["num_thoughts"])
dec = ThoughtDecoder(cfg["vocab_size"], cfg["d_model"], cfg["heads"],
    cfg["decoder_layers"], cfg["dropout"], cfg["max_seq_len"])
model = ThoughtVectorModel(enc, dec)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval().cuda()

# Build and load predictor (matches LossPredictor class definition)
from torch import nn

class LossPredictor(nn.Module):
    def __init__(self, d_model: int, max_thoughts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_thoughts),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1))

predictor = LossPredictor(cfg["d_model"], cfg["num_thoughts"])
predictor.load_state_dict(ckpt["predictor_state"])
predictor.eval().cuda()

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")

def find_min_k(text: str, target_loss: float = 0.5) -> tuple[int, float, str]:
    """Find minimum k where predicted loss <= target_loss."""
    ids = torch.tensor([tok.encode(text, add_special_tokens=True)], device="cuda")
    n_t = ids.size(1) - 2
    with torch.no_grad():
        thoughts = model.encoder(ids, ids.eq(tok.pad_token_id))
        pred_losses = predictor(thoughts)[0]  # [256]
    
    # Find min k
    min_k = cfg["num_thoughts"]
    for k in range(1, cfg["num_thoughts"] + 1):
        if pred_losses[k - 1].item() <= target_loss:
            min_k = k
            break
    
    # Decode at that k
    prefix = thoughts[:, :min_k, :]
    gen = torch.full((1, 1), tok.bos_token_id, dtype=torch.long, device="cuda")
    for _ in range(100):
        logits = model.decoder(prefix, gen)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen = torch.cat([gen, nxt], dim=1)
        if nxt.item() == tok.eos_token_id:
            break
    out = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    
    ratio = f"{n_t // max(1, min_k)}:1" if min_k <= n_t else f"1:{min_k // max(1, n_t)}"
    return min_k, pred_losses[min_k - 1].item(), out, n_t, ratio

print("Predictor-based k selection at target_loss=0.5")
print()
tests = [
    "a plane is taking off.",
    "The stock market crashed after the unexpected interest rate hike.",
    "Two roads diverged in a yellow wood, and sorry I could not travel both.",
    "The patient was diagnosed with acute myeloid leukemia and started induction chemotherapy.",
    "LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday.",
    "Hola, como estas hoy? Espero que todo vaya bien.",
]

for text in tests:
    k, pred_loss, out, n_t, ratio = find_min_k(text)
    match = "✓" if out.strip() == text.strip() else " "
    print(f"[{n_t:2d} tok]  target_loss=0.50  pred at k={k:3d} = {pred_loss:.4f}  ({ratio})  {match}")
    print(f"  IN:  {text[:70]}")
    print(f"  OUT: {out[:70]}")
    print()
