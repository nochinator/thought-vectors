"""One-batch overfit smoke test: the full loss path must drive CE near zero.

Runs on GPU when available (the real M0 gate), CPU otherwise (slower, looser).
"""

import torch

from thoughtvec.config import ModelCfg
from thoughtvec.losses import reconstruction_ce
from thoughtvec.model import ThoughtAutoencoder, make_padding_mask

CFG = ModelCfg(
    vocab_size=512,
    d_model=64,
    nhead=2,
    ffn_dim=128,
    enc_layers=2,
    dec_layers=2,
    max_seq_len=32,
    num_thoughts=16,
    dropout=0.0,
    thought_dropout=0.0,
)


def test_one_batch_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    model = ThoughtAutoencoder(CFG).to(device)
    ids = torch.randint(4, CFG.vocab_size, (8, 24), device=device)
    ids[:, 0] = 1
    ids[:, -1] = 2
    mask = make_padding_mask(ids)
    opt = torch.optim.AdamW(model.unique_parameters(), lr=3e-4)

    steps = 600 if device == "cuda" else 300
    final = None
    for _ in range(steps):
        logits = model(ids, mask)
        loss, _ = reconstruction_ce(logits, ids[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final = loss.item()
        if final < 0.05:
            break
    threshold = 0.05 if device == "cuda" else 0.5
    assert final is not None and final < threshold, f"overfit stalled at CE={final:.3f}"
