from __future__ import annotations

import torch
import torch.nn.functional as F

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
    penalty = length_penalty * thoughts.size(1)
    total = recon + penalty
    stats = {
        "reconstruction_loss": float(recon.detach().cpu()),
        "length_penalty": float(penalty),
        "total_loss": float(total.detach().cpu()),
    }
    return total, stats
