"""
BitThought — Inference utilities with K-predictor.
"""

from typing import Callable

import torch
import torch.nn.functional as F

from bitthought.model import BitThoughtModel


@torch.no_grad()
def encode(
    model: BitThoughtModel,
    text: str,
    tokenizer_encode: Callable[[str], list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text to thought vectors using K-predictor.

    Returns (thought_vectors, K_pred).
    The K-predictor determines how many vectors to keep.
    """
    tokens = tokenizer_encode(text)
    input_ids = torch.tensor([tokens], device=device)
    thoughts, k_pred, _ = model.encoder(input_ids, K=None)
    return thoughts, k_pred


@torch.no_grad()
def decode_greedy(
    model: BitThoughtModel,
    thought_vectors: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int = 256,
) -> torch.Tensor:
    """Greedy autoregressive decoding from thought vectors."""
    device = thought_vectors.device
    batch = thought_vectors.size(0)
    generated = torch.full((batch, 1), bos_token_id, dtype=torch.long, device=device)
    for _ in range(max_length):
        logits = model.decoder(thought_vectors, generated)
        next_tok = logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_tok], dim=1)
        if (next_tok == eos_token_id).all():
            break
    return generated


@torch.no_grad()
def encode_with_compression(
    model: BitThoughtModel,
    input_ids: torch.Tensor,
    loss_threshold: float,
    pad_token_id: int,
    stride: int = 2,
    max_vectors: int | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Encode with adaptive stopping at inference using K-predictor.

    Returns (selected_thought_vectors, loss_curve).
    """
    thoughts, k_pred, _ = model.encoder(input_ids, K=None)
    K = max(1, int(round(k_pred[0, 0].item())))
    K = min(K, max_vectors or thoughts.size(1))
    return thoughts[:, :K, :], [float(K)]


@torch.no_grad()
def find_minimum_vectors(
    model: BitThoughtModel,
    thoughts: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    loss_target: float,
    pad_token_id: int,
    stride: int = 2,
    max_vectors: int | None = None,
) -> tuple[int, list[float]]:
    """Find smallest thought-prefix meeting loss_target (for analysis/comparison)."""
    total = min(thoughts.size(1), max_vectors or thoughts.size(1))
    losses: list[float | None] = [None for _ in range(total)]
    coarse = sorted(set([1, *range(stride, total + 1, stride), total]))
    low, high = 1, total
    found = False
    for i in coarse:
        loss_i = _recon_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
        losses[i - 1] = loss_i
        if loss_i <= loss_target:
            high = i
            low = max(1, i - stride + 1)
            found = True
            break
    if not found:
        for i in range(1, total + 1):
            if losses[i - 1] is None:
                losses[i - 1] = _recon_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
        return total, [float(x) for x in losses]
    for i in range(low, high + 1):
        if losses[i - 1] is None:
            losses[i - 1] = _recon_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
        if losses[i - 1] <= loss_target:
            for j in range(1, total + 1):
                if losses[j - 1] is None:
                    losses[j - 1] = _recon_loss(model, thoughts[:, :j, :], input_ids, pad_token_id)
            return i, [float(x) for x in losses]
    for i in range(1, total + 1):
        if losses[i - 1] is None:
            losses[i - 1] = _recon_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
    return total, [float(x) for x in losses]


def _recon_loss(model, thought_subset, input_ids, pad_token_id):
    logits = model.decoder(thought_subset, input_ids[:, :-1])
    return float(F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        ignore_index=pad_token_id,
    ).detach().cpu())
