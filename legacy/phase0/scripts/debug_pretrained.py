
import torch, sys
sys.path.insert(0, ".")
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, LossPredictor, ThinkerModel
from torch import nn

device = "cuda"
tok = SPTokenizer(); tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id

ckpt = torch.load("artifacts/thinker_big_s1.pt", map_location="cpu", weights_only=True)
enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 8192, 256).to(device)
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 8192).to(device)
th = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, dropout=0.1, batch_first=True), 6).to(device)
pr = LossPredictor(256, 256).to(device)
enc.load_state_dict(ckpt["encoder_state"])
dec.load_state_dict(ckpt["decoder_state"])
th.load_state_dict(ckpt["thinker_state"])
pr.load_state_dict(ckpt["predictor_state"])
model = ThinkerModel(enc, dec, th, pr, max_turns=4).to(device)
model.eval()

ids = torch.randint(4, 1000, (1, 32), device=device)
thoughts = model.encoder(ids, ids.eq(pad_id))
print(f"Encoder NaN: {thoughts.isnan().any().item()}")
out = model.thinker_forward(thoughts[:,:16,:])
print(f"Thinker NaN: {out.isnan().any().item()}")
loss = torch.nn.functional.cross_entropy(
    model.decoder(out, ids[:,:-1]).reshape(-1, 16384),
    ids[:,1:].reshape(-1),
    ignore_index=pad_id
)
print(f"Loss: {loss.item():.4f} NaN: {loss.isnan().item()}")
if not loss.isnan().item():
    print("Pretrained thinker IS stable on ROCm!")
