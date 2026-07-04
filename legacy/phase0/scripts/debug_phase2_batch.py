#!/usr/bin/env python3
import torch, sys, csv, time, random as rnd
sys.path.insert(0, '.')
from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, LossPredictor, ThinkerModel
from torch import nn

device = 'cuda'
tok = SPTokenizer(); tok.load('/tmp/sp_c4_16k.model')
pad_id = tok.pad_token_id

t0 = time.time()
arts, sums = [], []
with open('/tmp/thinker_data_input.csv') as fa, open('/tmp/thinker_data_output.csv') as fs:
    for ar, sr in zip(csv.reader(fa), csv.reader(fs)):
        if ar and sr: arts.append(ar[0]); sums.append(sr[0])
print(f'load: {time.time()-t0:.1f}s', flush=True)

inputs, outputs = [], []
for i, (a, s) in enumerate(zip(arts, sums)):
    if len(tok.encode(a, True)) <= 128 and len(tok.encode(s, True)) <= 128:
        inputs.append(a); outputs.append(s)
print(f'filter: {time.time()-t0:.1f}s ({len(inputs)} pairs)', flush=True)

ckpt = torch.load('artifacts/thinker_big_s1.pt', map_location='cpu', weights_only=True)
enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 8192, 256).to(device)
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 8192).to(device)
th = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, dropout=0.1, batch_first=True), 6).to(device)
pr = LossPredictor(256, 256).to(device)
enc.load_state_dict(ckpt['encoder_state'])
dec.load_state_dict(ckpt['decoder_state'])
th.load_state_dict(ckpt['thinker_state'])
pr.load_state_dict(ckpt['predictor_state'])
model = ThinkerModel(enc, dec, th, pr, max_turns=4).to(device)
model.train()

p1 = torch.load('artifacts/thinker_phase1.pt', map_location='cpu', weights_only=True)
enc.load_state_dict(p1['encoder_state'])
dec.load_state_dict(p1['decoder_state'])
th.load_state_dict(p1['thinker_state'])
pr.load_state_dict(p1['predictor_state'])
if 'thinker_embeddings' in p1:
    e = p1['thinker_embeddings']
    model.turn_embedding.load_state_dict(e['turn_embedding'])
    model.speaker_embedding.load_state_dict(e['speaker_embedding'])
    model.decode_embedding.data.copy_(e['decode_embedding'].to(device))
print(f'model: {time.time()-t0:.1f}s', flush=True)

def encode_batch(texts):
    encoded = [tok.encode(t, add_special_tokens=True) for t in texts]
    max_len = max(len(e) for e in encoded)
    out = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return out.to(device)

B, n_past, k = 8, 3, 64
batch_indices = rnd.sample(range(len(inputs)), B)

t1 = time.time()
past_idx_pairs = []
for idx in batch_indices:
    k_needed = min(n_past, len(inputs) - 1)
    chosen = rnd.sample(range(len(inputs)), k=k_needed)
    for j in range(len(chosen)):
        if chosen[j] == idx:
            chosen[j] = len(inputs) - 1 - j
    past_idx_pairs.append(chosen)
print(f'sample: {time.time()-t1:.3f}s', flush=True)

texts_to_encode = []
for i, idx in enumerate(batch_indices):
    na = len(past_idx_pairs[i])
    for pi in range(na):
        pair_idx = past_idx_pairs[i][pi]
        texts_to_encode.append(inputs[pair_idx])
        texts_to_encode.append(outputs[pair_idx])
    texts_to_encode.append(inputs[idx])

t1 = time.time()
all_ids = encode_batch(texts_to_encode)
print(f'tokenize: {time.time()-t1:.3f}s ({all_ids.shape})', flush=True)

t1 = time.time()
all_thoughts = model.encoder(all_ids, all_ids.eq(pad_id))
torch.cuda.synchronize()
print(f'encode: {time.time()-t1:.3f}s ({all_thoughts.shape})', flush=True)

t1 = time.time()
all_thoughts_k = all_thoughts[:, :k, :]
total_k = (2*n_past+1)*k
contexts = torch.zeros(B, total_k, 256, device=device)
turn_ids = torch.zeros(B, total_k, dtype=torch.long, device=device)
speaker_ids = torch.zeros(B, total_k, dtype=torch.long, device=device)
decode_mask = torch.zeros(B, total_k, dtype=torch.bool, device=device)
ptr = 0
for i in range(B):
    contexts[i, :total_k] = all_thoughts_k[ptr:ptr+2*n_past+1].reshape(total_k, 256)
    ptr += 2*n_past+1
    pos = 0
    for t in range(n_past):
        turn_ids[i, pos:pos+k] = t; speaker_ids[i, pos:pos+k] = 0; pos += k
        turn_ids[i, pos:pos+k] = t; speaker_ids[i, pos:pos+k] = 1; pos += k
    turn_ids[i, pos:pos+k] = n_past; speaker_ids[i, pos:pos+k] = 0
    decode_mask[i, pos:pos+k] = 1
print(f'context: {time.time()-t1:.3f}s ({contexts.shape})', flush=True)

t1 = time.time()
out = model.thinker_forward(contexts, turn_ids, speaker_ids, decode_mask)
torch.cuda.synchronize()
print(f'thinker: {time.time()-t1:.3f}s ({out.shape})', flush=True)

target_texts = [outputs[idx] for idx in batch_indices]
target_ids = encode_batch(target_texts)
t1 = time.time()
decoded = []
for i in range(B):
    seg = out[i, decode_mask[i]].unsqueeze(0)
    lg = model.decoder(seg, target_ids[i:i+1, :-1])
    decoded.append(lg)
logits = torch.cat(decoded, dim=0)
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1), ignore_index=pad_id)
print(f'decode: {time.time()-t1:.3f}s', flush=True)
print(f'loss={loss.item():.4f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)
