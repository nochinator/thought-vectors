from __future__ import annotations

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
