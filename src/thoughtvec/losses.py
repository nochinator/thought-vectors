"""Loss functions: per-sample CE, KL divergence, predictor regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import PAD_ID


def reconstruction_ce(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(batch-mean CE, per-sample CE [B]) over non-pad target positions.

    logits [B, T, V], target_ids [B, T] (the inputs shifted left, i.e.
    input_ids[:, 1:]).
    """
    vocab = logits.size(-1)
    tok_ce = F.cross_entropy(
        logits.reshape(-1, vocab), target_ids.reshape(-1), ignore_index=PAD_ID, reduction="none"
    ).view_as(target_ids)
    valid = (target_ids != PAD_ID).float()
    per_sample = (tok_ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    mean = (tok_ce * valid).sum() / valid.sum().clamp(min=1)
    return mean, per_sample


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(N(mu, sigma^2) || N(0, 1)) per thought dimension."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def predictor_loss(
    predicted: torch.Tensor, k: int, actual_per_sample_ce: torch.Tensor
) -> torch.Tensor:
    """MSE between the predictor's column for prefix length k and observed CE.

    predicted [B, N] (column k-1 = prefix of k vectors), actual [B] detached.
    """
    return F.mse_loss(predicted[:, k - 1], actual_per_sample_ce.detach())
