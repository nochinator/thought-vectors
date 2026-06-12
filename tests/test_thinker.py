"""Thinker + dialogue-data tests (CPU, tiny config)."""

import json

import numpy as np
import torch

from thoughtvec.config import ThinkerCfg
from thoughtvec.data import DialogueDataset, collate_dialogue
from thoughtvec.thinker import Thinker

D = 32
TINY = ThinkerCfg(layers=2, nhead=2, ffn_dim=64, k_ctx=4, k_out=4, max_turns=6)


def make_batch(bsz=3, c=3, k=4):
    torch.manual_seed(0)
    ctx_th = torch.randn(bsz, c, k, D)
    ctx_roles = torch.tensor([[0, 1, 0]] * bsz)
    ctx_turns = torch.tensor([3, 2, 1])  # ragged: rows 1-2 have padded turns
    resp_roles = torch.tensor([1, 0, 1])
    return ctx_th, ctx_roles, ctx_turns, resp_roles


def test_query_forward_shape():
    th = Thinker(TINY, D)
    ctx_th, roles, turns, rr = make_batch()
    out = th(ctx_th, roles, turns, rr)
    assert out.shape == (3, TINY.k_out, D)
    assert out.isfinite().all()


def test_padded_turns_do_not_leak():
    """Row with ctx_turns=1 must ignore thoughts in its padded turn slots."""
    th = Thinker(TINY, D)
    th.eval()
    ctx_th, roles, turns, rr = make_batch()
    out1 = th(ctx_th, roles, turns, rr)
    poisoned = ctx_th.clone()
    poisoned[2, 1:] = 1e3  # row 2 has only 1 real turn; rest is padding
    out2 = th(poisoned, roles, turns, rr)
    assert torch.allclose(out1[2], out2[2], atol=1e-5)


def test_prefix_mask():
    mask = Thinker._prefix_mask(ctx_len=5, k_out=3, device="cpu")
    assert mask.shape == (8, 8)
    assert mask[:5, 5:].all()          # context cannot attend to response slots
    assert not mask[:5, :5].any()      # context fully visible to itself
    assert not mask[5:, :5].any()      # response sees all context
    resp = mask[5:, 5:]
    assert not resp.diagonal().any() and resp[0, 1] and resp[0, 2] and resp[1, 2]
    assert not resp[1, 0] and not resp[2, 0] and not resp[2, 1]


def test_prefix_teacher_forcing_matches_generate_first_slot():
    """Slot 0 has no teacher-forced input, so TF forward and AR generation
    must agree on it exactly."""
    cfg = ThinkerCfg(layers=2, nhead=2, ffn_dim=64, k_ctx=4, k_out=4, mode="prefix")
    th = Thinker(cfg, D)
    th.eval()
    ctx_th, roles, turns, rr = make_batch()
    tgt = torch.randn(3, cfg.k_out, D)
    with torch.no_grad():
        tf = th(ctx_th, roles, turns, rr, target_thoughts=tgt)
        ar = th(ctx_th, roles, turns, rr)  # no targets -> _prefix_generate
    assert tf.shape == ar.shape == (3, cfg.k_out, D)
    assert torch.allclose(tf[:, 0], ar[:, 0], atol=1e-5)


def test_slot_budgets_mask_tail_slots():
    """Thoughts beyond a turn's budget must not influence the output —
    equivalent to having encoded that turn at a smaller k."""
    th = Thinker(TINY, D)
    th.eval()
    ctx_th, roles, turns, rr = make_batch()
    budgets = torch.tensor([[2, 3, 4], [4, 2, 2], [3, 2, 2]])
    out1 = th(ctx_th, roles, turns, rr, slot_budgets=budgets)
    poisoned = ctx_th.clone()
    poisoned[0, 0, 2:] = 1e3  # row 0 turn 0 has budget 2; slots 2+ are masked
    out2 = th(poisoned, roles, turns, rr, slot_budgets=budgets)
    assert torch.allclose(out1[0], out2[0], atol=1e-5)
    # ...and within-budget slots DO matter
    poisoned2 = ctx_th.clone()
    poisoned2[0, 0, 1] = 1e3
    out3 = th(poisoned2, roles, turns, rr, slot_budgets=budgets)
    assert not torch.allclose(out1[0], out3[0], atol=1e-3)


def test_k_ctx_schedule_matches_explicit_budgets():
    """The recency schedule must equal handing in the same budgets directly."""
    cfg = ThinkerCfg(layers=2, nhead=2, ffn_dim=64, k_ctx=4, k_out=4,
                     max_turns=6, k_ctx_schedule=[4, 3, 2])
    th = Thinker(cfg, D)
    th.eval()
    ctx_th, roles, turns, rr = make_batch()
    out_sched = th(ctx_th, roles, turns, rr)
    # distances for turns=[3,2,1]: row0 -> [3,2,1] -> budgets [2,3,4]; etc.
    budgets = torch.tensor([[2, 3, 4], [3, 4, 2], [4, 2, 2]])
    th.cfg = ThinkerCfg(layers=2, nhead=2, ffn_dim=64, k_ctx=4, k_out=4, max_turns=6)
    out_explicit = th(ctx_th, roles, turns, rr, slot_budgets=budgets)
    assert torch.allclose(out_sched, out_explicit, atol=1e-6)


def test_predict_reverse_shape():
    cfg = ThinkerCfg(layers=2, nhead=2, ffn_dim=64, k_ctx=4, k_out=4, w_reverse=0.5)
    th = Thinker(cfg, D)
    ctx_th, roles, turns, rr = make_batch()
    out = th.predict_reverse(ctx_th, roles, turns, rr)
    assert out.shape == (3, cfg.k_ctx, D)


def test_reverse_arrangement():
    from thoughtvec.thinker_train import ThinkerTrainer

    ctx_th, _, turns, _ = make_batch()
    tgt = torch.randn(3, 4, D)
    rev_ctx, rev_tgt = ThinkerTrainer._reverse_arrangement(ctx_th, tgt, turns)
    for row, last in enumerate(turns - 1):
        assert torch.equal(rev_tgt[row], ctx_th[row, last])
        assert torch.equal(rev_ctx[row, last], tgt[row, :4])
    # untouched turns are preserved
    assert torch.equal(rev_ctx[0, :2], ctx_th[0, :2])


def _write_dialogue_shard(tmp_path, convs):
    """Minimal turn-aware shard mimicking pretokenize_dialogue output."""
    tokens, turns = [], []
    pos = 0
    for cid, conv in enumerate(convs):
        for body in conv:
            ids = [1] + body + [2]
            tokens.extend(ids)
            turns.append((pos, len(ids), cid))
            pos += len(ids)
    np.asarray(tokens, dtype=np.uint16).tofile(tmp_path / "tokens.bin")
    np.save(tmp_path / "turns.npy", np.asarray(turns, dtype=np.uint64))
    (tmp_path / "meta.json").write_text(json.dumps({"num_turns": len(turns)}))


def test_dialogue_dataset_and_collate(tmp_path):
    convs = [
        [[10, 11], [12], [13, 14, 15], [16]],  # 4 turns
        [[20], [21, 22]],                       # 2 turns
    ]
    _write_dialogue_shard(tmp_path, convs)
    ds = DialogueDataset(tmp_path, max_context=2)
    assert len(ds) == 4  # every non-first turn

    items = [ds[i] for i in range(len(ds))]
    batch = collate_dialogue(items)
    assert batch["ctx_ids"].shape[0] == 4
    assert batch["ctx_ids"].shape[1] == 2  # max_context cap

    # sample for conv0 turn 3 (parity 1 -> bot reply): context = turns 1,2
    it = items[2]
    assert it["resp_parity"] == 1
    assert it["response"].tolist() == [1, 16, 2]
    assert [t.tolist() for t in it["context"]] == [[1, 12, 2], [1, 13, 14, 15, 2]]

    # collate role parity: context roles alternate, ending opposite the response
    row = 2
    n = int(batch["ctx_turns"][row])
    rr = int(batch["resp_roles"][row])
    roles = batch["ctx_roles"][row, :n].tolist()
    assert roles == [(rr - (n - j)) % 2 for j in range(n)]
    assert roles[-1] != rr  # last context speaker != responder


def test_flat_context_dataset(tmp_path):
    convs = [[[10, 11], [12], [13, 14, 15], [16]]]
    _write_dialogue_shard(tmp_path, convs)
    ds = DialogueDataset(tmp_path, max_context=4, flat_context=True)
    it = ds[2]  # responding with turn 3; context = turns 0-2 concatenated
    assert len(it["context"]) == 1
    assert it["context"][0].tolist() == [1, 10, 11, 2, 1, 12, 2, 1, 13, 14, 15, 2]

    # window overflow drops OLDEST whole turns first
    ds_small = DialogueDataset(tmp_path, max_context=4, flat_context=True,
                               max_flat_tokens=10)
    it = ds_small[2]
    assert it["context"][0].tolist() == [1, 12, 2, 1, 13, 14, 15, 2]

    batch = collate_dialogue([ds[i] for i in range(len(ds))])
    assert batch["ctx_ids"].shape[1] == 1  # C=1 always
    assert (batch["ctx_turns"] == 1).all()
