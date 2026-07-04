"""Compare VAE compressed vs NAR compressed models."""
from __future__ import annotations
import sys
sys.path.insert(0, "thought-vectors-main")
import torch
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 1024, 256)
    dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 1024)
    model = ThoughtVectorModel(enc, dec)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval().cuda()
    return model

model_vae = load_model("thought-vectors-main/artifacts/vae_compressed.pt")
model_nar = load_model("thought-vectors-main/artifacts/vae_compressed_nar.pt")

tests = [
    (6, "a plane is taking off."),
    (12, "The stock market crashed after the unexpected interest rate hike."),
    (62, "LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported \u00a320 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him."),
]

for n_t, text in tests:
    ids = torch.tensor([tok.encode(text, add_special_tokens=True)], device="cuda")
    print(f"\n--- {n_t} tokens ---")
    print(f"IN: {text[:80]}...")
    with torch.no_grad():
        for label, model in [("VAE", model_vae), ("NAR", model_nar)]:
            thoughts = model.encoder(ids, ids.eq(tok.pad_token_id))
            k = min(256, max(4, int(n_t * 1.5)))
            prefix = thoughts[:, :k, :]
            gen = torch.full((1, 1), tok.bos_token_id, dtype=torch.long, device="cuda")
            for _ in range(200):
                logits = model.decoder(prefix, gen)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, nxt], dim=1)
                if nxt.item() == tok.eos_token_id:
                    break
            out = tok.decode(gen[0].tolist(), skip_special_tokens=True)
            match = "\u2713" if out.strip() == text.strip() else " "
            print(f"  {label} k={k:3d}: {match} {out[:120]}")
