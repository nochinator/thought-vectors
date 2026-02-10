from __future__ import annotations

from thought_vectors.tokenization import SimpleTokenizer


def test_tokenizer_from_checkpoint_can_fit_new_dataset_tokens() -> None:
    tokenizer = SimpleTokenizer()
    tokenizer.fit([["hello world"]])

    restored = SimpleTokenizer.from_token_to_id(tokenizer.token_to_id)
    restored.fit([["new tokens arrive"]])

    encoded = restored.encode("new tokens arrive", add_special_tokens=False)
    unk_id = restored.token_to_id[restored.unk_token]
    assert all(token_id != unk_id for token_id in encoded)


def test_tokenizer_handles_punctuation_and_case_consistently() -> None:
    tokenizer = SimpleTokenizer()
    tokenizer.fit([["Hello, WORLD!"]])

    encoded = tokenizer.encode("hello, world!", add_special_tokens=False)
    decoded = tokenizer.decode(encoded)

    assert decoded == "hello, world!"


def test_tokenizer_fit_with_memory_limited_counter_bounds_vocab_growth() -> None:
    tokenizer = SimpleTokenizer()
    groups = [[f"token_{i}"] for i in range(50)]

    tokenizer.fit(groups, count_memory_limit=10)

    assert tokenizer.vocab_size <= 14  # 4 special + at most 10 retained candidates
