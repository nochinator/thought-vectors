
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel

device = "cuda"
tok = SPTokenizer(); tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id; bos_id = tok.bos_token_id; eos_id = tok.eos_token_id

# Try ALL compression checkpoints with correct config
items = [
    ("compressed.pt", {}),
    ("compressed_128t.pt", {}),
    ("c4_256d_4x4_10ep_C4.pt", {}),
    ("vae_compressed.pt", {}),
    ("vae_compressed_nar.pt", {}),
    ("vae_compressed_blend.pt", {}),
    ("c4_256d_6x6_10ep_SUCCESS.pt", {}),
    ("shallow_wide_vae.pt", {}),
    ("large_512.pt", {}),
]

tests = [
    "Hello, how are you?",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog.",
]

for fname, _ in items:
    try:
        ckpt = torch.load(f"artifacts/{fname}", map_location="cpu", weights_only=True)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        # If state is just the state dict
        cfg = ckpt.get("config", {})
        d = cfg.get("d_model", 256)
        heads = cfg.get("heads", 4)
        enc_l = cfg.get("encoder_layers", 4)
        dec_l = cfg.get("decoder_layers", 4)
        do = cfg.get("dropout", 0.1)
        max_s = cfg.get("max_seq_len", 512)
        num_t = cfg.get("num_thoughts", cfg.get("thoughts", 256))
        
        enc = ThoughtEncoder(16384, d, heads, enc_l, do, max_s, num_t).to(device)
        dec = ThoughtDecoder(16384, d, heads, dec_l, do, max_s).to(device)
        model = ThoughtVectorModel(enc, dec).to(device)
        
        # Handle thought_seed size mismatch
        if "thought_seed" in state and state["thought_seed"].shape[-2] != num_t:
            del state["thought_seed"]
        
        result = model.load_state_dict(state, strict=False)
        if len(result.unexpected_keys) > 0:
            # Try with "model." prefix stripping
            fixed = {}
            for k, v in state.items():
                if k.startswith("model."):
                    fixed[k[6:]] = v  # strip "model." prefix
                else:
                    fixed[k] = v
            if "thought_seed" in fixed and fixed["thought_seed"].shape[-2] != num_t:
                del fixed["thought_seed"]
            result = model.load_state_dict(fixed, strict=False)
        
        misses = len(result.missing_keys)
        unexps = len(result.unexpected_keys)
        model.eval()
        
        print(f"\n{fname} ({d}d, {num_t}t): miss={misses} unexp={unexps}")
        for text in tests:
            ids = torch.tensor([tok.encode(text, add_special_tokens=True)], device=device)
            with torch.no_grad():
                thoughts_v = model.encoder(ids, ids.eq(pad_id))
                gen = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                for _ in range(35):
                    lg = model.decoder(thoughts_v, gen)
                    nx = lg[:, -1, :].argmax(dim=-1, keepdim=True)
                    gen = torch.cat([gen, nx], dim=1)
                    if nx.item() == eos_id: break
            recon = tok.decode(gen[0].tolist(), skip_special_tokens=True)
            print(f"  {text[:40]:40s} -> {recon[:55]}")
    except Exception as e:
        print(f"\n{fname}: {str(e)[:60]}")
