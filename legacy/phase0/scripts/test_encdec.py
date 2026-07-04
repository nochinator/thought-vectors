
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel

device = "cuda"
tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id
bos_id = tok.bos_token_id
eos_id = tok.eos_token_id

# Test multiple checkpoints for pure encoder-decoder reconstruction
checkpoints = [
    ("compressed.pt", "base compression"),
    ("vae_compressed.pt", "VAE compression"),
    ("c4_256d_128t_compressed_SUCCESS.pt", "128t C4"),
    ("sw_compressed_half.pt", "shallow-wide"),
]

tests = [
    "Hello, how are you?",
    "What is the capital of France? Paris",
    "Neural networks are computing systems inspired by biological brains.",
]

for fname, label in checkpoints:
    try:
        ckpt = torch.load(f"artifacts/{fname}", map_location="cpu", weights_only=True)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        config = ckpt.get("config", {})
        d_model = config.get("d_model", 256)
        enc = ThoughtEncoder(16384, d_model, 4, 4, 0.1, 8192, d_model).to(device)
        dec = ThoughtDecoder(16384, d_model, 4, 4, 0.1, 8192).to(device)
        
        # Try loading
        if isinstance(state, dict) and any(k.startswith("encoder.") for k in state.keys()):
            enc.load_state_dict({k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")})
            dec.load_state_dict({k.replace("decoder.", ""): v for k, v in state.items() if k.startswith("decoder.")})
            model = ThoughtVectorModel(enc, dec).to(device)
            model.eval()
        elif isinstance(state, dict) and any(k.endswith("weight") for k in state.keys()):
            # Direct encoder/decoder state dicts
            model = ThoughtVectorModel(enc, dec).to(device)
            model.load_state_dict(state)
            model.eval()
        else:
            print(f"{fname}: unknown state format")
            continue
        
        print(f"\n{fname} ({label}, d_model={d_model}):")
        for text in tests:
            ids = torch.tensor([tok.encode(text, add_special_tokens=True)], device=device)
            with torch.no_grad():
                thoughts = model.encoder(ids, ids.eq(pad_id))
                gen = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                for _ in range(40):
                    lg = model.decoder(thoughts, gen)
                    nx = lg[:, -1, :].argmax(dim=-1, keepdim=True)
                    gen = torch.cat([gen, nx], dim=1)
                    if nx.item() == eos_id: break
            recon = tok.decode(gen[0].tolist(), skip_special_tokens=True)
            print(f"  '{text[:50]}' -> {recon[:60]}")
    except Exception as e:
        print(f"{fname}: error - {e}")
