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
