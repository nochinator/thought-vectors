from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from thought_vectors.model import ThoughtVectorModel


@torch.no_grad()
def encode(
    model: ThoughtVectorModel,
    text: str,
    tokenizer_encode: Callable[[str], list[int]],
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer_encode(text)
    input_ids = torch.tensor([tokens], device=device)
    return model.encoder(input_ids)


@torch.no_grad()
def decode_greedy(
    model: ThoughtVectorModel,
    thought_vectors: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int = 256,
) -> torch.Tensor:
    generated = torch.full((thought_vectors.size(0), 1), bos_token_id, dtype=torch.long, device=thought_vectors.device)

    for _ in range(max_length):
        logits = model.decoder(thought_vectors, generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if (next_token == eos_token_id).all():
            break
    return generated


@torch.no_grad()
def encode_with_compression(
    model: ThoughtVectorModel,
    input_ids: torch.Tensor,
    loss_threshold: float,
    pad_token_id: int,
) -> tuple[torch.Tensor, list[float]]:
    thoughts = model.encoder(input_ids)
    losses: list[float] = []
    for i in range(1, thoughts.size(1) + 1):
        subset = thoughts[:, :i, :]
        logits = model.decoder(subset, input_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=pad_token_id,
        )
        value = float(loss.detach().cpu())
        losses.append(value)
        if value < loss_threshold:
            return subset, losses
    return thoughts, losses
