from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from thought_vectors.data import GroupTextDataset, collate_group_batch
from thought_vectors.model import ThoughtVectorModel


def training_step(
    model: ThoughtVectorModel,
    input_ids: torch.Tensor,
    pad_token_id: int,
    length_penalty: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    padding_mask = input_ids.eq(pad_token_id)

    thoughts = model.encoder(input_ids, padding_mask)
    logits = model.decoder(thoughts, input_ids[:, :-1], padding_mask[:, :-1])

    target = input_ids[:, 1:]
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_token_id,
    )
    penalty = recon.new_tensor(length_penalty * thoughts.size(1))
    total = recon + penalty
    stats = {
        "reconstruction_loss": float(recon.detach().cpu()),
        "length_penalty": float(penalty.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, stats


def train_model(
    model: ThoughtVectorModel,
    groups: list[list[str]],
    tokenizer_encode,
    pad_token_id: int,
    *,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    length_penalty: float = 0.01,
    shuffle: bool = True,
    seed: int = 0,
) -> list[float]:
    """Train thought-vector model on grouped text data and return epoch losses."""
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = GroupTextDataset(groups)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_group_batch(batch, tokenizer_encode, pad_token_id),
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    history: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_total = 0.0
        batches = 0
        for input_ids in loader:
            input_ids = input_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = training_step(model, input_ids, pad_token_id=pad_token_id, length_penalty=length_penalty)
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.detach().cpu())
            batches += 1

        history.append(epoch_total / max(1, batches))

    return history
