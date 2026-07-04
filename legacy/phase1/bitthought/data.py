"""
BitThought — Data loading utilities.

Supports JSON, JSONL, and CSV formats.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader

csv.field_size_limit(sys.maxsize)


def load_groups(path: Path, preprocess: bool = True) -> list[list[str]]:
    """Load grouped text data from a file.

    Supports:
      - .json: list[list[str]] or list[str]
      - .jsonl: one object per line with {"texts": [...]}
      - .csv: first column is text, each row is a single-element group
    """
    suffix = path.suffix.lower()

    def _clean(t: str) -> str:
        if not preprocess:
            return t.strip()
        import unicodedata, re
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON must contain a top-level list")
        groups: list[list[str]] = []
        for item in data:
            if isinstance(item, list):
                cleaned = [_clean(str(x)) for x in item if str(x).strip()]
                if cleaned:
                    groups.append(cleaned)
            else:
                t = _clean(str(item))
                if t:
                    groups.append([t])
        return groups

    if suffix == ".csv":
        groups = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue
                raw = str(row[0])
                if i == 0 and raw.strip().lower() in {"text", "sentence", "content"}:
                    continue
                t = _clean(raw)
                if t:
                    groups.append([t])
        return groups

    # Default: JSONL
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        texts = obj.get("texts", [])
        cleaned = [_clean(str(x)) for x in texts if str(x).strip()]
        if cleaned:
            groups.append(cleaned)
    return groups


def load_pairs(path: Path, preprocess: bool = True) -> list[tuple[str, str, float]]:
    """Load paired text data with similarity scores from CSV.

    Format: text_a, text_b, similarity_score
    Used for contrastive training on datasets like STSB.

    Returns list of (text_a, text_b, score) tuples.
    """
    suffix = path.suffix.lower()
    if suffix != ".csv":
        raise ValueError(f"Paired data only supports CSV, got {suffix}")

    def _clean(t: str) -> str:
        if not preprocess:
            return t.strip()
        import unicodedata, re
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    pairs: list[tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 3:
                continue
            if i == 0 and row[0].strip().lower() in {"text", "sentence", "content"}:
                continue
            t1 = _clean(row[0])
            t2 = _clean(row[1])
            try:
                score = float(row[2])
            except (ValueError, IndexError):
                continue
            if t1 and t2:
                pairs.append((t1, t2, max(0.0, min(1.0, score))))
    return pairs


def tokenize_dataset(
    groups: list[list[str]],
    tokenizer_encode: Callable[[str], list[int]],
    max_seq_len: int = 512,
    cache_path: Path | None = None,
) -> list[list[int]]:
    """Tokenize a dataset and optionally cache the result.

    Cache format: flat tensor + lengths for fast loading.
    Returns a list of token ID lists (backward-compatible).
    """
    if cache_path is not None:
        flat_path = cache_path.with_suffix(".flat.pt")
        # Try flat format first (fast), then list format (slow fallback)
        if flat_path.exists():
            cached = torch.load(flat_path, weights_only=True)
            tokens_t, lengths = cached["tokens"], cached["lengths"]
            result = []
            offset = 0
            for l in lengths.tolist():
                result.append(tokens_t[offset:offset + l].tolist())
                offset += l
            print(f"[cache] loaded {len(result)} sequences from {flat_path}")
            return result
        if cache_path.exists():
            print(f"[cache] loading list format (slow, converting)...")
            cached = torch.load(cache_path, weights_only=True)
            # Save as flat for next time
            lengths = torch.tensor([len(s) for s in cached], dtype=torch.int)
            tokens = torch.cat([torch.tensor(s, dtype=torch.int) for s in cached])
            flat_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"tokens": tokens, "lengths": lengths}, flat_path)
            print(f"[cache] loaded {len(cached)} sequences, converted to flat")
            return cached

    # Tokenize from scratch
    tokenized = []
    for group in groups:
        text = " ".join(group)
        ids = tokenizer_encode(text)[:max_seq_len]
        tokenized.append(ids)

    if cache_path is not None:
        flat_path = cache_path.with_suffix(".flat.pt")
        lengths = torch.tensor([len(s) for s in tokenized], dtype=torch.int)
        tokens = torch.cat([torch.tensor(s, dtype=torch.int) for s in tokenized])
        flat_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tokens": tokens, "lengths": lengths}, flat_path)
        print(f"[cache] saved {len(tokenized)} sequences to {flat_path}")

    return tokenized


class ThoughtDataset(Dataset):
    """Dataset wrapping tokenized sequences.

    Accepts pre-tokenized lists of token IDs to avoid re-tokenizing
    on every DataLoader call.
    """

    def __init__(self, tokenized_seqs: list[list[int]]):
        self.seqs = [s for s in tokenized_seqs if s]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


class PairedThoughtDataset(Dataset):
    """Dataset for paired texts with similarity scores (STSB-style)."""

    def __init__(self, pairs: list[tuple[str, str, float]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_thought_batch(
    batch: list[list[str]],
    tokenizer_encode: Callable[[str], list[int]],
    pad_token_id: int,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Collate a batch of text groups into a padded tensor.

    Each group's texts are concatenated with spaces, then tokenized.
    This is the on-the-fly version (slower but doesn't need pre-tokenization).
    """
    joined = [" ".join(group) for group in batch]
    encoded = [tokenizer_encode(text) for text in joined]
    encoded = [ids[:max_seq_len] for ids in encoded]
    max_len = max(len(ids) for ids in encoded)
    out = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(encoded):
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def collate_tokenized_batch(
    batch: list[list[int]],
    pad_token_id: int,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Collate a batch of pre-tokenized sequences into a padded tensor.

    No tokenization — just padding and truncation.
    This is the cached version (faster, used with pre-tokenized datasets).
    """
    encoded = [ids[:max_seq_len] for ids in batch]
    max_len = max(len(ids) for ids in encoded)
    out = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(encoded):
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def collate_pair_batch(
    batch: list[tuple[str, str, float]],
    tokenizer_encode: Callable[[str], list[int]],
    pad_token_id: int,
    max_seq_len: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a batch of paired texts into padded tensors.

    Returns (input_ids_a, input_ids_b, scores).
    """
    texts_a = [p[0] for p in batch]
    texts_b = [p[1] for p in batch]
    scores = torch.tensor([p[2] for p in batch], dtype=torch.float)

    def _collate(texts):
        encoded = [tokenizer_encode(t)[:max_seq_len] for t in texts]
        max_len = max(len(ids) for ids in encoded)
        out = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
        for i, ids in enumerate(encoded):
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    return _collate(texts_a), _collate(texts_b), scores
