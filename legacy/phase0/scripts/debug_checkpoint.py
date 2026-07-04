
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import SPTokenizer
from torch import nn

device = "cuda"
tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id
bos_id = tok.bos_token_id
eos_id = tok.eos_token_id

# Build a bare encoder+decoder from scratch (old format)
# Old checkpoints used raw nn.TransformerEncoder without wrapper
raw_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, batch_first=True, dropout=0.1), 4)
raw_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(256, 4, batch_first=True, dropout=0.1), 4)

# Try loading old checkpoint
ckpt = torch.load("artifacts/compressed.pt", map_location="cpu", weights_only=True)
state = ckpt["model_state"]
print(f"State dict keys (first 10): {list(state.keys())[:10]}")
print(f"Has encoder prefix: {any(k.startswith('encoder.') for k in state.keys())}")
print(f"Has decoder prefix: {any(k.startswith('decoder.') for k in state.keys())}")
print(f"thought_seed shape: {state.get('thought_seed', 'none')}")

# Separate encoder/decoder keys
enc_keys = {k: v for k, v in state.items() if k.startswith("encoder.")}
dec_keys = {k: v for k, v in state.items() if k.startswith("decoder.")}
raw_enc_keys = {k: v for k, v in state.items() if not any(k.startswith(p) for p in ["encoder.", "decoder.", "thought_seed"])}
raw_dec_keys = {}  # decoders might be separate

print(f"encoder.-prefixed: {len(enc_keys)}")
print(f"decoder.-prefixed: {len(dec_keys)}")
print(f"raw keys (no prefix): {len(raw_enc_keys)}")
