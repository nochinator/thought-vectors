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
    chunk_jitter: bool = False,
    seed: int = 1234,
) -> dict:
    """Stream a CSV, tokenize, write train + val shard dirs.

    max_tokens excludes BOS/EOS (stored length is max_tokens + 2).
    merge_rows: concatenate consecutive short rows up to max_tokens (for
    fragment-style corpora). chunk_long: split over-long docs into pieces.
    chunk_jitter: randomize piece/merge target lengths (uniform 16..max_tokens)
    so the shard covers the whole length spectrum — on long-only corpora the
    decoder can minimize CE by pure LM-ing and the thought channel never
    engages (see RESEARCH_LOG 2026-06-11 E256 collapse).
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

    def next_target() -> int:
        if chunk_jitter:
            return int(rng.integers(min(16, max_tokens), max_tokens + 1))
        return max_tokens

    for csv_path in csv_paths:
        pending: list[int] = []
        target = next_target()
        rows_read = 0
        for text in iter_csv_texts(csv_path, max_chars=20000):
            rows_read += 1
            if max_rows is not None and rows_read > max_rows:
                break
            body = tokenizer.encode(text, add_special=False)
            if merge_rows:
                if pending and len(pending) + len(body) > target:
                    emit([BOS_ID] + pending + [EOS_ID])
                    pending = []
                    target = next_target()
                if len(body) <= target:
                    pending.extend(body)
                    continue
            if len(body) > target:
                if not chunk_long:
                    continue
                j = 0
                while j < len(body):
                    emit([BOS_ID] + body[j : j + target] + [EOS_ID])
                    j += target
                    target = next_target()
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


def pretokenize_dialogue(
    jsonl_path: str | Path,
    out_dir: str | Path,
    tokenizer: Tokenizer,
    max_turn_tokens: int = 126,
    min_turns: int = 2,
    val_frac: float = 0.01,
    seed: int = 1234,
) -> dict:
    """conversations.jsonl -> turn-aware shards for thinker training.

    Format: tokens.bin (uint16, every turn stored BOS..EOS) +
    turns.npy uint64 [n_turns, 3] = (start, length, conv_id) + meta.json.
    Turn parity within a conversation gives the role (even = user).
    Long turns are tail-truncated to max_turn_tokens. Split is BY
    CONVERSATION so no dialogue straddles train/val.
    """
    import json as _json

    from .tokenizer import BOS_ID, EOS_ID

    out_dir = Path(out_dir)
    val_dir = out_dir.parent / (out_dir.name + "_val")
    rng = np.random.default_rng(seed)
    writers = {}
    for name, d in (("train", out_dir), ("val", val_dir)):
        d.mkdir(parents=True, exist_ok=True)
        writers[name] = {"bin": open(d / "tokens.bin", "wb"), "turns": [], "pos": 0,
                         "dir": d, "convs": 0}

    with open(jsonl_path) as f:
        for line in f:
            conv = _json.loads(line)["turns"]
            if len(conv) < min_turns:
                continue
            w = writers["val"] if rng.random() < val_frac else writers["train"]
            cid = w["convs"]
            kept = 0
            for turn in conv:
                body = tokenizer.encode(turn, add_special=False)[:max_turn_tokens]
                if not body:
                    continue
                ids = np.asarray([BOS_ID] + body + [EOS_ID], dtype=np.uint16)
                w["bin"].write(ids.tobytes())
                w["turns"].append((w["pos"], len(ids), cid))
                w["pos"] += len(ids)
                kept += 1
            if kept >= min_turns:
                w["convs"] += 1
            else:  # roll back bookkeeping is overkill; just advance conv id
                w["convs"] += 1

    meta_out = {}
    for name, w in writers.items():
        w["bin"].close()
        turns = np.asarray(w["turns"], dtype=np.uint64)
        np.save(w["dir"] / "turns.npy", turns)
        lengths = turns[:, 1].astype(np.int64) if len(turns) else np.array([0])
        meta = {
            "tokenizer": tokenizer.model_path,
            "num_turns": int(len(turns)),
            "num_convs": int(w["convs"]),
            "total_tokens": int(w["pos"]),
            "turn_len_mean": float(lengths.mean()),
            "turn_len_p90": float(np.percentile(lengths, 90)),
        }
        (w["dir"] / "meta.json").write_text(json.dumps(meta, indent=2))
        meta_out[name] = meta
    return meta_out


class DialogueDataset(Dataset):
    """Samples (context turns, response turn) pairs from turn-aware shards.

    A sample exists for every turn with at least one predecessor in its
    conversation; context is the up-to-max_context preceding turns. Roles
    alternate by turn parity (even = user), so responses at odd parity are
    "bot" turns — but every turn is a training target (role symmetry doubles
    the data; the speaker embedding tells the thinker who is replying).
    """

    def __init__(self, shard_dir: str | Path, max_context: int = 6) -> None:
        shard_dir = Path(shard_dir)
        self.tokens = np.memmap(shard_dir / "tokens.bin", dtype=np.uint16, mode="r")
        self.turns = np.load(shard_dir / "turns.npy")
        self.meta = json.loads((shard_dir / "meta.json").read_text())
        self.max_context = max_context
        conv_ids = self.turns[:, 2]
        first_of_conv = np.concatenate([[True], conv_ids[1:] != conv_ids[:-1]])
        self.samples = np.nonzero(~first_of_conv)[0]  # every non-first turn

    def __len__(self) -> int:
        return len(self.samples)

    def _turn(self, idx: int) -> torch.Tensor:
        start, length, _ = self.turns[idx]
        return torch.from_numpy(
            self.tokens[int(start) : int(start) + int(length)].astype(np.int64)
        )

    def __getitem__(self, i: int):
        t = int(self.samples[i])
        cid = self.turns[t, 2]
        lo = t
        while lo > 0 and self.turns[lo - 1, 2] == cid and t - lo < self.max_context:
            lo -= 1
        return {
            "context": [self._turn(j) for j in range(lo, t)],
            "response": self._turn(t),
            "resp_parity": int((t - self._conv_start(t)) % 2),
        }

    def _conv_start(self, t: int) -> int:
        cid = self.turns[t, 2]
        lo = t
        while lo > 0 and self.turns[lo - 1, 2] == cid:
            lo -= 1
        return lo


def collate_dialogue(batch: list[dict]) -> dict:
    """Pads turns to a common length and context to a common turn count.

    Returns:
      ctx_ids   [B, C, T]  (PAD-filled; absent turns all-PAD)
      ctx_turns [B]        number of real context turns per sample
      ctx_roles [B, C]     0 user / 1 bot (parity-derived), PAD turns = 0
      resp_ids  [B, T2]
      resp_roles [B]
    """
    bsz = len(batch)
    c_max = max(len(b["context"]) for b in batch)
    t_max = max(max(t.size(0) for t in b["context"]) for b in batch)
    t2_max = max(b["response"].size(0) for b in batch)
    ctx_ids = torch.full((bsz, c_max, t_max), PAD_ID, dtype=torch.long)
    ctx_roles = torch.zeros(bsz, c_max, dtype=torch.long)
    ctx_turns = torch.zeros(bsz, dtype=torch.long)
    resp_ids = torch.full((bsz, t2_max), PAD_ID, dtype=torch.long)
    resp_roles = torch.zeros(bsz, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["context"])
        ctx_turns[i] = n
        resp_roles[i] = b["resp_parity"]
        for j, t in enumerate(b["context"]):
            ctx_ids[i, j, : t.size(0)] = t
            ctx_roles[i, j] = (b["resp_parity"] - (n - j)) % 2
        resp_ids[i, : b["response"].size(0)] = b["response"]
    return {
        "ctx_ids": ctx_ids,
        "ctx_turns": ctx_turns,
        "ctx_roles": ctx_roles,
        "resp_ids": resp_ids,
        "resp_roles": resp_roles,
    }


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
