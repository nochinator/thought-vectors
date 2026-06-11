"""SentencePiece BPE tokenizer: training, loading, encode/decode.

Fixed special ids: pad=0, bos=1, eos=2, unk=3 (model.py depends on these).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import sentencepiece as spm

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

csv.field_size_limit(sys.maxsize)


def iter_csv_texts(csv_path: str | Path, every_nth: int = 1, max_chars: int = 2000):
    """Yield text rows from a single-text-column CSV.

    Handles both headerless files (C4 subsets) and files with a `text` header
    (minipile). Long documents are split into <= max_chars line-ish chunks.
    """
    import itertools

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return
        if first and first[0].strip().lower() == "text":
            rows = reader  # header row, skip it
        else:
            rows = itertools.chain([first], reader)
        for i, row in enumerate(rows):
            if i % every_nth != 0 or not row:
                continue
            text = row[0].strip()
            if not text:
                continue
            if len(text) <= max_chars:
                yield text
            else:
                for j in range(0, len(text), max_chars):
                    chunk = text[j : j + max_chars].strip()
                    if chunk:
                        yield chunk


def train_tokenizer(
    corpus_files: list[tuple[str, int]],
    out_prefix: str | Path,
    vocab_size: int = 16384,
    max_sentences: int = 4_000_000,
) -> Path:
    """Train SentencePiece BPE. corpus_files: [(csv_path, every_nth), ...]."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def sentence_iter():
        count = 0
        for path, nth in corpus_files:
            for text in iter_csv_texts(path, every_nth=nth):
                yield text
                count += 1
                if count >= max_sentences:
                    return

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iter(),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        train_extremely_large_corpus=False,
    )
    return out_prefix.with_suffix(".model")


class Tokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.model_path = str(model_path)
        assert self.sp.pad_id() == PAD_ID and self.sp.bos_id() == BOS_ID
        assert self.sp.eos_id() == EOS_ID and self.sp.unk_id() == UNK_ID

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = self.sp.encode(text)
        return [BOS_ID] + ids + [EOS_ID] if add_special else ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)])
