
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel

device = "cuda"
tok = SPTokenizer(); tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id; bos_id = tok.bos_token_id; eos_id = tok.eos_token_id

# Load compressed checkpoint DIRECTLY into ThoughtVectorModel (no key remapping)
ckpt = torch.load("artifacts/compressed_128t.pt", map_location="cpu", weights_only=True)
state = ckpt.get("model_state", ckpt)
print(f"First 5 keys: {list(state.keys())[:5]}")
print(f"Has encoder.encoder: {any(k.startswith('encoder.encoder') for k in state.keys())}")
print(f"Has encoder.token_: {any(k.startswith('encoder.token_') for k in state.keys())}")

# Check what config says
cfg = ckpt.get("config", {})
print(f"Config: {cfg}")

# Build model that matches
d = cfg.get("d_model", 256)
thoughts = cfg.get("thoughts", 256)
enc = ThoughtEncoder(16384, d, 4, 4, 0.1, 8192, thoughts).to(device)
dec = ThoughtDecoder(16384, d, 4, 4, 0.1, 8192).to(device)
model = ThoughtVectorModel(enc, dec).to(device)

# Try loading WITHOUT any remapping - the state dict should match the model structure
try:
    model.load_state_dict(state, strict=False)
    missed = model.load_state_dict(state, strict=False)
    print(f"Missing keys: {len(missed.missing_keys)}")
    print(f"Unexpected keys: {len(missed.unexpected_keys)}")
    if len(missed.unexpected_keys) == 0:
        print("All keys loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    
model.eval()

# Test reconstruction at k=thoughts (full vectors)
text = "Hello, how are you today?"
ids = torch.tensor([tok.encode(text, add_special_tokens=True)], device=device)
with torch.no_grad():
    thoughts_v = model.encoder(ids, ids.eq(pad_id))
    gen = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    for _ in range(30):
        lg = model.decoder(thoughts_v, gen)
        nx = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        gen = torch.cat([gen, nx], dim=1)
        if nx.item() == eos_id: break
recon = tok.decode(gen[0].tolist(), skip_special_tokens=True)
print(f"Reconstruction: {recon}")
