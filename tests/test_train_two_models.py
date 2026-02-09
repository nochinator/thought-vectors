from __future__ import annotations

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from thought_vectors import SimpleTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel, train_model

import pytest

torch = pytest.importorskip("torch")



def _build_tiny_groups() -> list[list[str]]:
    return [
        ["cat sits on mat", "feline on rug", "small cat resting"],
        ["dog runs fast", "canine sprints quickly", "rapid dog motion"],
        ["rain falls tonight", "it is raining", "night rain starts"],
    ]


def _make_model(vocab_size: int) -> ThoughtVectorModel:
    encoder = ThoughtEncoder(
        vocab_size=vocab_size,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.0,
        max_seq_len=64,
        num_thoughts=4,
    )
    decoder = ThoughtDecoder(
        vocab_size=vocab_size,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.0,
        max_seq_len=64,
    )
    return ThoughtVectorModel(encoder, decoder)


def test_can_train_two_small_models_on_tiny_dataset() -> None:
    groups = _build_tiny_groups()
    tokenizer = SimpleTokenizer()
    tokenizer.fit(groups)

    model_a = _make_model(tokenizer.vocab_size)
    model_b = _make_model(tokenizer.vocab_size)

    history_a = train_model(
        model_a,
        groups,
        tokenizer.encode,
        tokenizer.pad_token_id,
        device=torch.device("cpu"),
        epochs=4,
        batch_size=2,
        learning_rate=3e-3,
        length_penalty=0.0,
        seed=1,
    )
    history_b = train_model(
        model_b,
        groups,
        tokenizer.encode,
        tokenizer.pad_token_id,
        device=torch.device("cpu"),
        epochs=4,
        batch_size=2,
        learning_rate=3e-3,
        length_penalty=0.0,
        seed=2,
    )

    assert len(history_a) == 4
    assert len(history_b) == 4
    assert history_a[-1] <= history_a[0]
    assert history_b[-1] <= history_b[0]
