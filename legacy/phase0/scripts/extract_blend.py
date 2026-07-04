
import torch, sys
sys.path.insert(0, ".")
from collections import OrderedDict

ckpt = torch.load("artifacts/vae_compressed_blend.pt", map_location="cpu", weights_only=True)
state = ckpt.get("model_state", ckpt)

# Extract: strip only the FIRST "encoder." / "decoder." prefix
# encoder.encoder.layers... -> encoder.layers...  (keep inner encoder)
# encoder.token_embedding -> token_embedding
enc_state = OrderedDict()
dec_state = OrderedDict()
for k, v in state.items():
    if k.startswith("encoder."):
        clean_k = k[len("encoder."):]  # strip outer "encoder." prefix
        enc_state[clean_k] = v
    elif k.startswith("decoder."):
        clean_k = k[len("decoder."):]  # strip outer "decoder." prefix
        dec_state[clean_k] = v

print(f"Encoder keys: {list(enc_state.keys())[:6]}")
print(f"Decoder keys: {list(dec_state.keys())[:6]}")

# Verify against actual ThoughtEncoder/ThoughtDecoder shapes
from thought_vectors import ThoughtEncoder, ThoughtDecoder
enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 512, 256)
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 512)

try:
    enc.load_state_dict(enc_state)
    print("Encoder loaded OK")
except Exception as e:
    print(f"Encoder load error: {e}")

try:
    dec.load_state_dict(dec_state)
    print("Decoder loaded OK")
except Exception as e:
    print(f"Decoder load error: {e}")

output = {"encoder_state": enc_state, "decoder_state": dec_state, "config": {"d_model": 256, "thoughts": 256, "vocab_size": 16384}}
torch.save(output, "artifacts/blend_encdec.pt")
print("Saved blend_encdec.pt")
