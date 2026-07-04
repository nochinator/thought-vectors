
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel
device = "cuda"
tok = SPTokenizer(); tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id; bos_id = tok.bos_token_id; eos_id = tok.eos_token_id

ckpt = torch.load("artifacts/vae_compressed_blend.pt", map_location="cpu", weights_only=True)
state = ckpt.get("model_state", ckpt)
cfg = ckpt.get("config", {})
enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 512, 256).to(device)
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 512).to(device)
model = ThoughtVectorModel(enc, dec).to(device)
model.load_state_dict(state, strict=False)
model.eval()

tests = [
    "Hello, how are you?",
    "The sky is blue and the grass is green.",
    "The stock market crashed yesterday due to rising interest rates.",
    "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes.",
    "Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities like burning fossil fuels.",
    "The theory of general relativity, proposed by Albert Einstein in 1915, describes how gravity is not a force but a curvature of spacetime caused by mass and energy. This revolutionary idea changed our understanding of the universe.",
]

for text in tests:
    ids = torch.tensor([tok.encode(text, add_special_tokens=True)], device=device)
    n_tok = ids.size(1) - 2
    with torch.no_grad():
        thoughts = model.encoder(ids, ids.eq(pad_id))
        gen = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(n_tok + 20):
            lg = model.decoder(thoughts, gen)
            nx = lg[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nx], dim=1)
            if nx.item() == eos_id: break
    recon = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    print(f"\n{n_tok} tok input: {text[:60]}...")
    print(f"  reconstruction: {recon[:80]}")
