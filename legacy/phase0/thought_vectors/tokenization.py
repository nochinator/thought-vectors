from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from heapq import nlargest
from typing import Iterable


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
        groups: Iterable[list[str]],
        *,
        min_frequency: int = 1,
        max_vocab_size: int | None = None,
        count_memory_limit: int | None = None,
    ) -> None:
        counts: Counter[str] = Counter()
        for group in groups:
            for text in group:
                counts.update(self._tokenize(text))
                if count_memory_limit is not None and len(counts) > count_memory_limit:
                    # Keep only the highest-frequency candidates to bound RAM usage.
                    counts = Counter(dict(nlargest(count_memory_limit, counts.items(), key=lambda item: item[1])))

        candidates = [token for token, frequency in counts.items() if frequency >= min_frequency and token not in self.token_to_id]
        candidates.sort(key=lambda token: (-counts[token], token))

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


class SPTokenizer:
    """SentencePiece unigram tokenizer wrapper with BOS/EOS/PAD special tokens.

    Requires the ``sentencepiece`` package to be installed.
    Uses fixed IDs: 0=PAD, 1=BOS, 2=EOS, 3=UNK, 4+ = learned subword tokens.
    """

    def __init__(self) -> None:
        self.sp: SentencePieceProcessor | None = None  # type: ignore[name-defined]
        self._model_path: str | None = None

    # --- special token IDs (fixed layout) ---
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def load(self, model_path: str) -> None:
        """Load a pre-trained SentencePiece model from disk."""
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self._model_path = model_path

    def train(self, input_files: str | list[str], *, vocab_size: int = 8192, model_prefix: str = "sp_tokenizer") -> None:
        """Train a unigram SentencePiece model on the given text file(s)."""
        import sentencepiece as spm

        spm.SentencePieceTrainer.train(
            input=input_files,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            pad_id=self.pad_token_id,
            bos_id=self.bos_token_id,
            eos_id=self.eos_token_id,
            unk_id=self.unk_token_id,
            pad_piece="<pad>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            unk_piece="<unk>",
            character_coverage=1.0,
            input_sentence_size=10_000_000,
        )
        self.load(f"{model_prefix}.model")

    @property
    def vocab_size(self) -> int:
        if self.sp is not None:
            return self.sp.GetPieceSize()
        return 4  # just special tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert self.sp is not None, "SPTokenizer: load or train a model first"
        ids = self.sp.EncodeAsIds(text)
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert self.sp is not None, "SPTokenizer: load or train a model first"
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in special]
        return self.sp.DecodeIds(token_ids)

    def to_dict(self) -> dict[str, int]:
        """Export token->id mapping for checkpoint storage."""
        assert self.sp is not None
        d: dict[str, int] = {}
        for i in range(self.sp.GetPieceSize()):
            d[self.sp.IdToPiece(i)] = i
        return d

    @classmethod
    def from_dict(cls, token_to_id: dict[str, int]) -> "SPTokenizer":
        """Reconstruct from a token->id dict (for checkpoint resume)."""
        tok = cls()
        # Build a SentencePiece processor from the dict
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        # SentencePiece can be reconstructed from a serialized model proto,
        # but the simplest path is: build the dict and the rest of the code
        # uses .encode / .decode which need a real model file.
        # For resume, the caller should save and reload .model files alongside
        # the checkpoint. This method is provided for API compatibility.
        raise NotImplementedError(
            "SPTokenizer.from_dict() is not supported. "
            "Save and load the .model file instead, or use SimpleTokenizer for checkpoint resume."
        )

