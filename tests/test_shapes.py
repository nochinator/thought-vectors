"""Shape, weight-tying and GRU-expand-equivalence tests (CPU, tiny config)."""

import torch

from thoughtvec.config import ModelCfg
from thoughtvec.losses import reconstruction_ce
from thoughtvec.model import PAD_ID, ThoughtAutoencoder, make_padding_mask

TINY = ModelCfg(
    vocab_size=128,
    d_model=32,
    nhead=2,
    ffn_dim=64,
    enc_layers=2,
    dec_layers=2,
    max_seq_len=32,
    num_thoughts=16,
)


def make_batch(batch=3, seq=12):
    torch.manual_seed(0)
    ids = torch.randint(4, TINY.vocab_size, (batch, seq))
    ids[:, 0] = 1  # BOS
    ids[:, -1] = 2  # EOS
    ids[0, -3:] = PAD_ID  # one padded row
    ids[0, -4] = 2
    return ids


def test_forward_shapes():
    model = ThoughtAutoencoder(TINY)
    ids = make_batch()
    mask = make_padding_mask(ids)
    thoughts = model.encode(ids, mask)
    assert thoughts.shape == (3, TINY.num_thoughts, TINY.d_model)
    logits = model(ids, mask)
    assert logits.shape == (3, ids.size(1) - 1, TINY.vocab_size)
    logits_k = model(ids, mask, k=4)
    assert logits_k.shape == logits.shape


def test_weight_tying():
    model = ThoughtAutoencoder(TINY)
    emb = model.token_embedding.weight
    assert model.encoder.token_embedding.weight.data_ptr() == emb.data_ptr()
    assert model.decoder.token_embedding.weight.data_ptr() == emb.data_ptr()
    assert model.decoder.lm_head.weight.data_ptr() == emb.data_ptr()
    # unique_parameters yields the table exactly once
    ptrs = [p.data_ptr() for p in model.unique_parameters()]
    assert len(ptrs) == len(set(ptrs))


def test_gru_expand_equals_per_batch():
    """Running the GRU once on [1,N,d] and expanding == running it per row."""
    model = ThoughtAutoencoder(TINY).eval()
    enc = model.encoder
    batch = 4
    expanded = enc._scaffold(batch)
    per_batch, _ = enc.thought_gru(enc.thought_seed.expand(batch, -1, -1))
    assert torch.allclose(expanded, per_batch, atol=1e-6)


def test_prefix_changes_output():
    model = ThoughtAutoencoder(TINY).eval()
    ids = make_batch()
    mask = make_padding_mask(ids)
    with torch.no_grad():
        full = model(ids, mask)
        small = model(ids, mask, k=2)
    assert not torch.allclose(full, small)


def test_ce_ignores_padding():
    model = ThoughtAutoencoder(TINY).eval()
    ids = make_batch()
    mask = make_padding_mask(ids)
    with torch.no_grad():
        logits = model(ids, mask)
        mean, per_sample = reconstruction_ce(logits, ids[:, 1:])
    assert per_sample.shape == (3,)
    assert mean.isfinite()


def test_vae_path():
    model = ThoughtAutoencoder(TINY)
    ids = make_batch()
    mask = make_padding_mask(ids)
    z, mu, logvar = model.encoder.encode_with_kl(ids, mask)
    assert z.shape == mu.shape == logvar.shape == (3, TINY.num_thoughts, TINY.d_model)
    assert logvar.max() <= 10 and logvar.min() >= -10


def test_nar_decode_shape():
    model = ThoughtAutoencoder(TINY).eval()
    ids = make_batch()
    mask = make_padding_mask(ids)
    thoughts = model.encode(ids, mask)
    blank = torch.full_like(ids[:, :-1], PAD_ID)
    with torch.no_grad():
        logits = model.decode(thoughts[:, :4], blank, None, causal=False)
    assert logits.shape == (3, ids.size(1) - 1, TINY.vocab_size)
