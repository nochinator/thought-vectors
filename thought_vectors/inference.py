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
def _reconstruction_loss(
    model: ThoughtVectorModel,
    thought_subset: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> float:
    logits = model.decoder(thought_subset, input_ids[:, :-1])
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        ignore_index=pad_token_id,
    )
    return float(loss.detach().cpu())


@torch.no_grad()
def find_minimum_vectors_for_target(
    model: ThoughtVectorModel,
    thoughts: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    loss_target: float,
    pad_token_id: int,
    stride: int = 2,
    max_vectors: int | None = None,
) -> tuple[int, list[float]]:
    """Find the smallest thought-prefix size whose reconstruction loss is <= loss_target.

    Efficient strategy:
    1) Coarse scan with `stride` to find first interval where target is met.
    2) Fine linear scan only inside that interval.
    """
    total = min(thoughts.size(1), max_vectors or thoughts.size(1))
    losses: list[float | None] = [None for _ in range(total)]

    coarse_points = sorted(set([1, *range(stride, total + 1, stride), total]))
    low = 1
    high = total
    found = False

    for i in coarse_points:
        loss_i = _reconstruction_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
        losses[i - 1] = loss_i
        if loss_i <= loss_target:
            high = i
            low = max(1, i - stride + 1)
            found = True
            break

    if not found:
        # fill any missing losses linearly to keep diagnostics complete
        for i in range(1, total + 1):
            if losses[i - 1] is None:
                losses[i - 1] = _reconstruction_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
        return total, [float(x) for x in losses]

    for i in range(low, high + 1):
        if losses[i - 1] is None:
            losses[i - 1] = _reconstruction_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
        if losses[i - 1] <= loss_target:
            for j in range(1, total + 1):
                if losses[j - 1] is None:
                    losses[j - 1] = _reconstruction_loss(model, thoughts[:, :j, :], input_ids, pad_token_id)
            return i, [float(x) for x in losses]

    for i in range(1, total + 1):
        if losses[i - 1] is None:
            losses[i - 1] = _reconstruction_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
    return total, [float(x) for x in losses]


@torch.no_grad()
def encode_with_compression(
    model: ThoughtVectorModel,
    input_ids: torch.Tensor,
    loss_target: float,
    pad_token_id: int,
    stride: int = 2,
    max_vectors: int | None = None,
) -> tuple[torch.Tensor, list[float]]:
    thoughts = model.encoder(input_ids)
    num_vectors, losses = find_minimum_vectors_for_target(
        model,
        thoughts,
        input_ids,
        loss_target=loss_target,
        pad_token_id=pad_token_id,
        stride=stride,
        max_vectors=max_vectors,
    )
    return thoughts[:, :num_vectors, :], losses
