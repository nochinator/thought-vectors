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
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Greedy (or sampled) autoregressive decoding from thought vectors.

    Args:
        model: the thought-vector model.
        thought_vectors: [B, k, d_model].
        bos_token_id: BOS token id.
        eos_token_id: EOS token id.
        max_length: maximum generation length.
        temperature: scaling for logits before softmax.
            0.0 = greedy argmax (default).  0.7–1.0 = sampling.
        top_k: if set, sample only from the top-k highest probability tokens.
        top_p: if set, nucleus sampling — sample from the smallest
            set whose cumulative probability exceeds top_p.

    Returns:
        [B, T] generated token ids.
    """
    generated = torch.full(
        (thought_vectors.size(0), 1), bos_token_id,
        dtype=torch.long, device=thought_vectors.device,
    )

    for _ in range(max_length):
        logits = model.decoder(thought_vectors, generated)
        next_logits = logits[:, -1, :]  # [B, V]

        if temperature <= 0.0:
            # Greedy argmax (default)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            # Temperature-scaled sampling
            scaled = next_logits / temperature
            probs = F.softmax(scaled, dim=-1)

            if top_k is not None:
                # Keep only top-k, renormalise
                top_probs, top_indices = probs.topk(top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if top_p is not None:
                # Nucleus (top-p) filtering
                sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                cutoff = (cumsum - sorted_probs) >= top_p  # exclude first token above threshold
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                # Restore original order
                probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
        if (next_token == eos_token_id).all():
            break

    return generated


@torch.no_grad()
def _fill_missing(
    losses: list[float | None],
    total: int,
    model: ThoughtVectorModel,
    thoughts: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> None:
    for i in range(1, total + 1):
        if losses[i - 1] is not None:
            continue
        losses[i - 1] = _reconstruction_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)


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

    if found:
        for i in range(low, high + 1):
            if losses[i - 1] is None:
                losses[i - 1] = _reconstruction_loss(model, thoughts[:, :i, :], input_ids, pad_token_id)
            if losses[i - 1] <= loss_target:
                _fill_missing(losses, total, model, thoughts, input_ids, pad_token_id)
                return i, [float(x) for x in losses]

    _fill_missing(losses, total, model, thoughts, input_ids, pad_token_id)
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

