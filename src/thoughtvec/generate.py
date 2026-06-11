"""Autoregressive decoding from thought vectors."""

from __future__ import annotations

import torch

from .model import BOS_ID, EOS_ID, ThoughtAutoencoder


@torch.no_grad()
def greedy_decode(
    model: ThoughtAutoencoder,
    thoughts: torch.Tensor,
    max_len: int = 128,
    memory_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return sample_decode(
        model, thoughts, max_len, temperature=0.0, memory_padding_mask=memory_padding_mask
    )


@torch.no_grad()
def sample_decode(
    model: ThoughtAutoencoder,
    thoughts: torch.Tensor,
    max_len: int = 128,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    memory_padding_mask: torch.Tensor | None = None,
    no_repeat_ngram: int = 0,
) -> torch.Tensor:
    """Decode token ids [B, <=max_len] from thoughts. temperature=0 -> greedy.

    memory_padding_mask [B, N] (True = masked) enables per-sample prefix
    lengths in one batched decode. no_repeat_ngram=n bans completing any
    already-generated n-gram (kills the low-k repetition loops)."""
    device = thoughts.device
    batch = thoughts.size(0)
    ids = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = model.decode(thoughts, ids, memory_padding_mask=memory_padding_mask)[:, -1]
        if no_repeat_ngram > 1 and ids.size(1) >= no_repeat_ngram:
            n = no_repeat_ngram
            for b in range(batch):
                row = ids[b].tolist()
                prefix = tuple(row[-(n - 1):])
                banned = [
                    row[i + n - 1]
                    for i in range(len(row) - n + 1)
                    if tuple(row[i : i + n - 1]) == prefix
                ]
                if banned:
                    logits[b, banned] = float("-inf")
        if temperature <= 0:
            next_ids = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p > 0:
                sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
                cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                cut = cum > top_p
                cut[:, 1:] = cut[:, :-1].clone()  # keep first token past the threshold
                cut[:, 0] = False
                remove = torch.zeros_like(cut).scatter(1, sorted_idx, cut)
                logits = logits.masked_fill(remove, float("-inf"))
            next_ids = torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)
        next_ids = next_ids.masked_fill(finished, EOS_ID)
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
        finished |= next_ids == EOS_ID
        if finished.all():
            break
    return ids
