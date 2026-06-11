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


def test_memory_mask_equals_prefix_slice():
    """Per-sample k via memory_key_padding_mask must equal prefix slicing."""
    model = ThoughtAutoencoder(TINY)
    model.eval()
    ids = make_batch()
    mask = make_padding_mask(ids)
    n = TINY.num_thoughts
    with torch.no_grad():
        thoughts = model.encode(ids, mask)
        k = 5
        sliced = model.decode(thoughts[:, :k], ids[:, :-1], mask[:, :-1])
        ks = torch.full((ids.size(0),), k, dtype=torch.long)
        slot = torch.arange(n)
        mem_pad = slot[None, :] >= ks[:, None]
        masked = model.decode(thoughts, ids[:, :-1], mask[:, :-1], memory_padding_mask=mem_pad)
    assert torch.allclose(sliced, masked, atol=1e-5), (sliced - masked).abs().max()


def test_per_sample_k_changes_rows_independently():
    model = ThoughtAutoencoder(TINY)
    model.eval()
    ids = make_batch()
    mask = make_padding_mask(ids)
    n = TINY.num_thoughts
    slot = torch.arange(n)
    with torch.no_grad():
        thoughts = model.encode(ids, mask)
        ks_a = torch.tensor([2, 8, n])
        ks_b = torch.tensor([2, 3, n])  # only row 1 differs
        out_a = model.decode(thoughts, ids[:, :-1], mask[:, :-1],
                             memory_padding_mask=slot[None, :] >= ks_a[:, None])
        out_b = model.decode(thoughts, ids[:, :-1], mask[:, :-1],
                             memory_padding_mask=slot[None, :] >= ks_b[:, None])
    assert torch.allclose(out_a[0], out_b[0], atol=1e-5)
    assert torch.allclose(out_a[2], out_b[2], atol=1e-5)
    assert not torch.allclose(out_a[1], out_b[1], atol=1e-3)


def test_predictor_loss_per_k_matches_scalar():
    from thoughtvec.losses import predictor_loss, predictor_loss_per_k

    torch.manual_seed(1)
    pred = torch.rand(4, 16)
    actual = torch.rand(4)
    k = 7
    scalar = predictor_loss(pred, k, actual)
    vector = predictor_loss_per_k(pred, torch.full((4,), k, dtype=torch.long), actual)
    assert torch.allclose(scalar, vector)


def test_monotone_predictor_non_increasing():
    from thoughtvec.predictor import LossPredictor

    torch.manual_seed(2)
    p = LossPredictor(32, 16, monotone=True)
    out = p(torch.randn(5, 16, 32))
    diffs = out[:, 1:] - out[:, :-1]
    assert (diffs <= 1e-6).all(), "monotone predictor must be non-increasing in k"
