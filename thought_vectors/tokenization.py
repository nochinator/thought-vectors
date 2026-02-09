from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class SimpleTokenizer:
    """A lightweight regex tokenizer with deterministic vocabulary building."""

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    lowercase: bool = True
    token_pattern: str = r"\w+|[^\w\s]"
    _token_re: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._token_re = re.compile(self.token_pattern, flags=re.UNICODE)
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
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return self._token_re.findall(text.strip())

    def fit(
        self,
        groups: list[list[str]],
        *,
        min_frequency: int = 1,
        max_vocab_size: int | None = None,
    ) -> None:
        counts: Counter[str] = Counter()
        for group in groups:
            for text in group:
                counts.update(self._tokenize(text))

        candidates = [
            token
            for token, frequency in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency >= min_frequency and token not in self.token_to_id
        ]

        if max_vocab_size is not None:
            free_slots = max(0, max_vocab_size - len(self.token_to_id))
            candidates = candidates[:free_slots]

        for token in candidates:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        token_ids = [self.token_to_id.get(tok, self.unk_token_id) for tok in self._tokenize(text)]
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

        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text
