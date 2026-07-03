"""End-to-end ThinkerTrainer smoke test on CPU: tiny codec, real tokenizer,
synthetic dialogue shard, every loss mode exercised for a few steps."""

import json

import pytest
import torch

from thoughtvec.config import Config, to_dict
from thoughtvec.data import pretokenize_dialogue
from thoughtvec.model import ThoughtAutoencoder
from thoughtvec.tokenizer import Tokenizer

TOKENIZER = "artifacts/tokenizer/spm16k_bpe.model"

CONVS = [
    ["hi there", "hello, how are you?", "doing well thanks", "glad to hear it"],
    ["what did you do today?", "went for a walk by the river", "sounds relaxing"],
    ["do you like coffee?", "i prefer tea in the morning", "me too actually"],
    ["any plans this weekend?", "probably just reading a book", "which one?", "a mystery novel"],
] * 4


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("thinker_smoke")
    tok = Tokenizer(TOKENIZER)

    cfg = Config()
    cfg.run.device = "cpu"
    cfg.run.out_dir = str(root / "ckpt")
    cfg.run.log_dir = str(root / "logs")
    cfg.run.tokenizer_path = TOKENIZER
    cfg.model.d_model = 32
    cfg.model.nhead = 2
    cfg.model.ffn_dim = 64
    cfg.model.enc_layers = 1
    cfg.model.dec_layers = 1
    cfg.model.max_seq_len = 32
    cfg.model.num_thoughts = 8

    codec = ThoughtAutoencoder(cfg.model)
    codec_path = root / "codec.pt"
    torch.save({"model": codec.state_dict(), "config": to_dict(cfg)}, codec_path)

    jsonl = root / "convs.jsonl"
    jsonl.write_text(
        "".join(json.dumps({"source": "t", "turns": c}) + "\n" for c in CONVS)
    )
    shard = root / "dialogue"
    pretokenize_dialogue(jsonl, shard, tok, max_turn_tokens=30, val_frac=0.2)

    # tiny text shard for the compression-anchor path (TokenShardDataset format)
    import numpy as np

    text_shard = root / "text"
    text_shard.mkdir()
    tokens, offsets, pos = [], [], 0
    for c in CONVS:
        ids = tok.encode(" ".join(c))[:30]
        tokens.extend(ids)
        offsets.append((pos, len(ids)))
        pos += len(ids)
    np.asarray(tokens, dtype=np.uint16).tofile(text_shard / "tokens.bin")
    np.save(text_shard / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    (text_shard / "meta.json").write_text(json.dumps({"num_samples": len(offsets)}))
    return root, cfg, codec_path, shard, text_shard


def make_trainer(setup, **thinker_overrides):
    from thoughtvec.thinker_train import ThinkerTrainer

    root, base, codec_path, shard, text_shard = setup
    cfg = Config()
    cfg.run = base.run
    cfg.data.shard_dir = str(shard)
    cfg.train.batch_size = 2
    cfg.train.max_steps = 4
    cfg.train.warmup_steps = 2
    cfg.train.log_every = 2
    cfg.train.sample_every = 3
    cfg.train.val_every = 4
    cfg.train.ckpt_every = 4
    cfg.thinker.layers = 1
    cfg.thinker.nhead = 2
    cfg.thinker.ffn_dim = 64
    cfg.thinker.k_ctx = 4
    cfg.thinker.k_out = 4
    cfg.thinker.codec_ckpt = str(codec_path)
    cfg.thinker.compress_shard = str(text_shard)
    for k, v in thinker_overrides.items():
        setattr(cfg.thinker, k, v)
    cfg.run.name = "smoke_" + "_".join(f"{k}{v}" for k, v in thinker_overrides.items()) or "smoke"
    return ThinkerTrainer(cfg, Tokenizer(TOKENIZER))


@pytest.mark.parametrize(
    "ov",
    [
        {},                                              # T0: thought loss only
        {"w_thought": 0.0, "w_decoder": 1.0},            # T1: decoder CE only
        {"w_decoder": 0.5, "w_reverse": 0.5},            # T4: mixed + reverse
        {"cycle_frac": 1.0, "w_cycle": 0.5},             # T5: cycle always on
        {"mode": "prefix"},                              # P0
        {"unfreeze": "decoder", "compress_frac": 1.0, "w_decoder": 1.0},  # U1 anchor path
        {"ctx_tau": 0.5},                                # TAU: predictor-adaptive budgets
        {"k_ctx_schedule": [4, 3, 2]},                   # SCHED: recency-decayed budgets
        {"n_hypotheses": 3, "w_decoder": 0.5},           # WTA multi-hypothesis
        {"n_hypotheses": 3, "w_decoder": 0.5, "slot_weight_decay": 0.3},  # WTA + slot weighting
        {"mode": "prefix", "tf_noise_std": 0.2, "w_decoder": 0.5},  # PN exposure fix
        {"n_hypotheses": 3, "w_decoder": 0.5, "contrast_weight": 0.2},  # R4: NTXent anti-collapse (WTA path)
        {"n_hypotheses": 3, "w_decoder": 0.5, "diversity_weight": 0.1},  # R4: batch-variance anti-collapse
        {"n_hypotheses": 3, "w_decoder": 0.5, "role_flip": True},  # R4: user<->bot flip
        {"k_out": 4, "k_out_random_prefix": True, "k_out_min": 2, "w_decoder": 0.5},  # R4: KRP + small k_out
        {"n_hypotheses": 3, "w_decoder": 0.5, "role_flip": True, "unfreeze": "decoder",
         "compress_frac": 0.4, "codec_lr_scale": 0.05},  # R4 COMBO: role_flip × unfreeze stack (the 12h candidate)
    ],
    ids=["thought", "decoder", "mixed_rev", "cycle", "prefix", "unfreeze_compress",
         "ctx_tau", "sched", "wta", "wta_slotw", "prefix_noise",
         "r4_contrast", "r4_diversity", "r4_roleflip", "r4_krp_kout", "r4_combo"],
)
def test_trainer_modes(setup, ov):
    from thoughtvec.thinker_train import make_dialogue_loader

    trainer = make_trainer(setup, **ov)
    _, _, _, shard, _ = setup
    loader = make_dialogue_loader(shard, 2, max_context=4, shuffle=True, num_workers=0)
    val_loader = make_dialogue_loader(
        str(shard) + "_val", 2, max_context=4, shuffle=False, num_workers=0
    )
    trainer.fit(loader, val_loader)
    assert trainer.step == 4
    assert (trainer.run_dir / "final.pt").exists()
    import math

    val = trainer.validate(val_loader)
    assert math.isfinite(val["val_cos"])
    if trainer.cfg.thinker.w_decoder > 0:
        assert math.isfinite(val["val_dec_ce"])
    assert trainer.nan_streak == 0  # padded context turns must not NaN the loss
    # resume path
    trainer2 = make_trainer(setup, **ov)
    trainer2.load_checkpoint(trainer.run_dir / "final.pt")
    assert trainer2.step == 4


def test_trainer_flat_context(setup):
    from thoughtvec.thinker_train import make_dialogue_loader

    trainer = make_trainer(setup, flat_context=True)
    _, _, _, shard, _ = setup
    loader = make_dialogue_loader(shard, 2, max_context=4, shuffle=True, num_workers=0,
                                  flat_context=True, max_flat_tokens=32)
    trainer.fit(loader, loader)
    assert trainer.step == 4
    val = trainer.validate(loader)
    import math

    assert math.isfinite(val["val_cos"])
    if trainer.cfg.thinker.w_decoder > 0:
        assert math.isfinite(val["val_dec_ce"])

    # chat path builds the flat context itself
    from thoughtvec.chat import ChatSession

    session = ChatSession(str(trainer.run_dir / "final.pt"), device="cpu")
    session.reply("hi")
    reply = session.reply("how are you?")
    assert isinstance(reply, str)


def test_chat_session_wta(setup):
    """Chat must handle [1, M, k, d] predictions: predictor-ranked at temp 0,
    random hypothesis at temp > 0."""
    from thoughtvec.chat import ChatSession
    from thoughtvec.thinker_train import make_dialogue_loader

    trainer = make_trainer(setup, n_hypotheses=3, w_decoder=0.5)
    _, _, _, shard, _ = setup
    loader = make_dialogue_loader(shard, 2, max_context=4, shuffle=True, num_workers=0)
    trainer.fit(loader, loader)

    session = ChatSession(str(trainer.run_dir / "final.pt"), device="cpu")
    assert isinstance(session.reply("hello!"), str)
    assert isinstance(session.reply("what's up?", temperature=0.8), str)


def test_chat_session(setup):
    from thoughtvec.chat import ChatSession
    from thoughtvec.thinker_train import make_dialogue_loader

    trainer = make_trainer(setup)
    _, _, _, shard, _ = setup
    loader = make_dialogue_loader(shard, 2, max_context=4, shuffle=True, num_workers=0)
    trainer.fit(loader, loader)

    session = ChatSession(str(trainer.run_dir / "final.pt"), device="cpu")
    reply = session.reply("hello there, how are you?")
    assert isinstance(reply, str)
    assert len(session.history) == 2
    session.reply("nice. any plans today?")
    assert len(session.history) == 4
    session.reset()
    assert not session.history


def test_wta_slot_weights_reduce_to_unweighted():
    """Uniform slot weights must reproduce the unweighted WTA loss exactly,
    and a decay vector must change the per-hypothesis ranking it computes."""
    from thoughtvec.thinker_train import _slot_weights, wta_losses

    torch.manual_seed(0)
    pred = torch.randn(5, 3, 8, 16)   # [B, M, k_out, d]
    target = torch.randn(5, 8, 16)    # [B, k_out, d]

    base = wta_losses(pred, target)
    uniform = wta_losses(pred, target, torch.ones(8))
    assert torch.allclose(base, uniform, atol=1e-6)

    decay = _slot_weights(8, 0.3, pred.device)
    assert decay[0] == 1.0 and decay[-1] == pytest.approx(0.3)
    weighted = wta_losses(pred, target, decay)
    assert weighted.shape == base.shape
    assert not torch.allclose(weighted, base)  # weighting actually bites


def test_out_budget_mask_adaptive(setup):
    """out_budget_mask returns per-sample budgets in [2, k_out] and a mask that
    hides exactly the slots past each budget."""
    import math

    from thoughtvec.thinker_train import out_budget_mask

    _, _, codec_path, _, _ = setup
    from thoughtvec.config import from_dict
    from thoughtvec.model import ThoughtAutoencoder

    ck = torch.load(codec_path, map_location="cpu", weights_only=False)
    codec = ThoughtAutoencoder(from_dict(ck["config"]).model)
    codec.load_state_dict(ck["model"])
    codec.eval()

    k_out = 8
    pred = torch.randn(5, k_out, 32)  # tiny test codec d_model = 32
    budgets, mask = out_budget_mask(codec, pred, out_tau=0.5)
    assert budgets.shape == (5,)
    assert int(budgets.min()) >= 2 and int(budgets.max()) <= k_out
    assert mask.shape == (5, k_out)
    # mask is True exactly where slot index >= budget
    for b in range(5):
        cut = int(budgets[b])
        assert not mask[b, :cut].any()
        assert mask[b, cut:].all()
    # out_tau huge → any prefix clears the bar → use the FEWEST vectors (budget 2)
    lo_b, _ = out_budget_mask(codec, pred, out_tau=float("inf"))
    assert int(lo_b.max()) == 2
    # out_tau=0 → no prefix qualifies → keep ALL slots, mask nothing
    hi_b, hi_m = out_budget_mask(codec, pred, out_tau=0.0)
    assert int(hi_b.min()) == k_out and not hi_m.any()
