from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimpleTokenizer:
    """A tiny whitespace tokenizer suitable for local tests and demos."""

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        self.token_to_id = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @classmethod
    def from_token_to_id(cls, token_to_id: dict[str, int]) -> "SimpleTokenizer":
        tokenizer = cls()
        tokenizer.token_to_id = dict(token_to_id)
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.token_to_id.items()}
        return tokenizer

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def fit(self, groups: list[list[str]]) -> None:
        for group in groups:
            for text in group:
                for token in text.strip().split():
                    if token not in self.token_to_id:
                        idx = len(self.token_to_id)
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        token_ids = [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in text.strip().split()]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []
        for idx in token_ids:
            if skip_special_tokens and idx in special_ids:
                continue
            tokens.append(self.id_to_token.get(idx, self.unk_token))
        return " ".join(tokens)
