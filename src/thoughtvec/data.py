"""Data pipeline: CSV -> memmap token shards; map-style dataset over shards.

Shard format (directory):
    tokens.bin   uint16 token ids, all samples concatenated (BOS...EOS included)
    offsets.npy  uint64 [num_samples, 2] (start, length)
    meta.json    tokenizer path, counts, length histogram
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import PAD_ID
from .tokenizer import Tokenizer, iter_csv_texts


def pretokenize(
    csv_paths: str | Path | list,
    out_dir: str | Path,
    tokenizer: Tokenizer,
    min_tokens: int = 4,
    max_tokens: int = 126,
    val_frac: float = 0.004,
    max_rows: int | None = None,
    merge_rows: bool = False,
    chunk_long: bool = True,
    seed: int = 1234,
) -> dict:
    """Stream a CSV, tokenize, write train + val shard dirs.

    max_tokens excludes BOS/EOS (stored length is max_tokens + 2).
    merge_rows: concatenate consecutive short rows up to max_tokens (for
    fragment-style corpora). chunk_long: split over-long docs into pieces.
    max_rows applies per CSV.
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    out_dir = Path(out_dir)
    val_dir = out_dir.parent / (out_dir.name + "_val")
    rng = np.random.default_rng(seed)

    writers = {}
    for name, d in (("train", out_dir), ("val", val_dir)):
        d.mkdir(parents=True, exist_ok=True)
        writers[name] = {
            "bin": open(d / "tokens.bin", "wb"),
            "offsets": [],
            "pos": 0,
            "dir": d,
            "hist": np.zeros(max_tokens + 3, dtype=np.int64),
        }

    def emit(ids: list[int]) -> None:
        if not (min_tokens + 2 <= len(ids) <= max_tokens + 2):
            return
        w = writers["val"] if rng.random() < val_frac else writers["train"]
        arr = np.asarray(ids, dtype=np.uint16)
        w["bin"].write(arr.tobytes())
        w["offsets"].append((w["pos"], len(ids)))
        w["pos"] += len(ids)
        w["hist"][len(ids)] += 1

    from .tokenizer import BOS_ID, EOS_ID

    for csv_path in csv_paths:
        pending: list[int] = []
        rows_read = 0
        for text in iter_csv_texts(csv_path, max_chars=20000):
            rows_read += 1
            if max_rows is not None and rows_read > max_rows:
                break
            body = tokenizer.encode(text, add_special=False)
            if merge_rows:
                if pending and len(pending) + len(body) > max_tokens:
                    emit([BOS_ID] + pending + [EOS_ID])
                    pending = []
                if len(body) <= max_tokens:
                    pending.extend(body)
                    continue
            if len(body) > max_tokens:
                if not chunk_long:
                    continue
                for j in range(0, len(body), max_tokens):
                    emit([BOS_ID] + body[j : j + max_tokens] + [EOS_ID])
            else:
                emit([BOS_ID] + body + [EOS_ID])
        if pending:
            emit([BOS_ID] + pending + [EOS_ID])

    meta_out = {}
    for name, w in writers.items():
        w["bin"].close()
        offsets = np.asarray(w["offsets"], dtype=np.uint64)
        np.save(w["dir"] / "offsets.npy", offsets)
        lengths = offsets[:, 1].astype(np.int64) if len(offsets) else np.array([], dtype=np.int64)
        meta = {
            "tokenizer": tokenizer.model_path,
            "source_csv": [str(p) for p in csv_paths],
            "num_samples": int(len(offsets)),
            "total_tokens": int(w["pos"]),
            "length_mean": float(lengths.mean()) if len(lengths) else 0.0,
            "length_median": float(np.median(lengths)) if len(lengths) else 0.0,
            "length_p10_p90": (
                [float(np.percentile(lengths, 10)), float(np.percentile(lengths, 90))]
                if len(lengths)
                else [0.0, 0.0]
            ),
        }
        (w["dir"] / "meta.json").write_text(json.dumps(meta, indent=2))
        meta_out[name] = meta
    return meta_out


class TokenShardDataset(Dataset):
    def __init__(self, shard_dir: str | Path) -> None:
        shard_dir = Path(shard_dir)
        self.tokens = np.memmap(shard_dir / "tokens.bin", dtype=np.uint16, mode="r")
        self.offsets = np.load(shard_dir / "offsets.npy")
        self.meta = json.loads((shard_dir / "meta.json").read_text())

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start, length = self.offsets[idx]
        return torch.from_numpy(
            self.tokens[int(start) : int(start) + int(length)].astype(np.int64)
        )


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.size(0) for t in batch)
    out = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, t in enumerate(batch):
        out[i, : t.size(0)] = t
    return out


def make_loader(
    shard_dir: str | Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    seed: int = 1234,
) -> DataLoader:
    ds = TokenShardDataset(shard_dir)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        generator=gen if shuffle else None,
        persistent_workers=num_workers > 0,
    )
