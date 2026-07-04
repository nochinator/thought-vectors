from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset


@dataclass
class GroupExample:
    texts: list[str]


class GroupTextDataset(Dataset[GroupExample]):
    """Simple group-based dataset where each example contains semantically related strings."""

    def __init__(self, groups: list[list[str]]) -> None:
        self.groups = [GroupExample(texts=g) for g in groups if g]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> GroupExample:
        return self.groups[idx]


def collate_group_batch(
    batch: list[GroupExample],
    tokenizer: Callable[[str], list[int]],
    pad_token_id: int,
) -> torch.Tensor:
    """Flattens groups into one text per example for baseline training.

    A richer strategy (sampling or contrastive pairing within group) can be layered on top.
    """
    joined = [" ".join(example.texts) for example in batch]
    encoded = [tokenizer(text) for text in joined]
    max_len = max(len(ids) for ids in encoded)

    out = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(encoded):
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def collate_contrastive_batch(
    batch: list[GroupExample],
    tokenizer: Callable[[str], list[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample two texts per group for contrastive paired training.

    Returns a pair of padded tensors ``(view_a, view_b)`` each shaped
    ``(B, max_len)``.  Groups with only one text duplicate it.
    """
    texts_a: list[str] = []
    texts_b: list[str] = []
    for example in batch:
        texts = example.texts
        if len(texts) >= 2:
            a, b = random.sample(texts, 2)
        else:
            a = b = texts[0]
        texts_a.append(a)
        texts_b.append(b)

    encoded_a = [tokenizer(t) for t in texts_a]
    encoded_b = [tokenizer(t) for t in texts_b]

    max_len = 0
    for ids_a, ids_b in zip(encoded_a, encoded_b):
        max_len = max(max_len, len(ids_a), len(ids_b))

    out_a = torch.full((len(encoded_a), max_len), pad_token_id, dtype=torch.long)
    out_b = torch.full((len(encoded_b), max_len), pad_token_id, dtype=torch.long)

    for i, (ids_a, ids_b) in enumerate(zip(encoded_a, encoded_b)):
        out_a[i, : len(ids_a)] = torch.tensor(ids_a, dtype=torch.long)
        out_b[i, : len(ids_b)] = torch.tensor(ids_b, dtype=torch.long)

    return out_a, out_b
