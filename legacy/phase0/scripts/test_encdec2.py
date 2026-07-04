
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel
from torch import nn
from collections import OrderedDict

device = "cuda"
tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id
bos_id = tok.bos_token_id
eos_id = tok.eos_token_id

# Load old compressed.pt checkpoints — note the double-nested prefix
tests = [
    ("Hello, how are you?", 7),
    ("What is the capital of France?", 8),
    ("The quick brown fox jumps over the lazy dog.", 11),
    ("Neural networks are computing systems inspired by biological neural networks.", 14),
]

# Check multiple VAE and compression checkpoints
checkpoints = [
    "vae_compressed.pt",
    "vae_compressed_nar.pt", 
    "vae_compressed_blend.pt",
    "vae_compressed_skew.pt",
    "compressed.pt",
    "compressed_128t.pt",
    "c4_256d_4x4_10ep_C4.pt",
]

for fname in checkpoints:
    try:
        ckpt = torch.load(f"artifacts/{fname}", map_location="cpu", weights_only=True)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        
        # Check dimensions
        first_weight = next(iter(state.values()))
        d_model = first_weight.shape[-1] if len(first_weight.shape) >= 2 else 256
        n_thoughts = 256  # default
        
        # Build model matching the checkpoint config
        cfg = ckpt.get("config", {})
        d_model = cfg.get("d_model", d_model)
        n_thoughts_val = cfg.get("thoughts", n_thoughts)
        
        enc = ThoughtEncoder(16384, d_model, 4, 4, 0.1, 8192, n_thoughts_val).to(device)
        dec = ThoughtDecoder(16384, d_model, 4, 4, 0.1, 8192).to(device)
        model = ThoughtVectorModel(enc, dec).to(device)
        
        # Fix state dict keys: strip the extra "encoder." and "decoder." prefix
        fixed = OrderedDict()
        for k, v in state.items():
            if k.startswith("encoder.encoder."):
                fixed["encoder." + k[len("encoder.encoder."):]] = v
            elif k.startswith("encoder.") and not k.startswith("encoder.token_"):
                new_k = k.replace("encoder.", "encoder.encoder.", 1)  
                fixed[new_k] = v
            elif k.startswith("decoder.decoder."):
                fixed["decoder." + k[len("decoder.decoder."):]] = v
            elif k.startswith("decoder.") and not k.startswith("decoder.token_"):
                new_k = k.replace("decoder.", "decoder.decoder.", 1)
                fixed[new_k] = v
            elif k.startswith("thought_seed") or k.startswith("encoder.thought_seed"):
                pass  # skip learned seed for now
            elif k == "token_embedding.weight":
                fixed[k] = v
            else:
                fixed[k] = v
        
        model.load_state_dict(fixed, strict=False)
        model.eval()
        
        print(f"\n{fname} (d={d_model}, thoughts={n_thoughts_val}):")
        for text, expected_tokens in tests:
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
            print(f"  {text[:45]:45s} -> {recon[:60]}")
    except Exception as e:
        err = str(e)[:80]
        print(f"\n{fname}: {err}")
