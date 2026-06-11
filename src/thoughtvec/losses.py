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


def _pred_mse(cols: torch.Tensor, actual: torch.Tensor, log_space: bool) -> torch.Tensor:
    if log_space:
        # MSE in log1p space: equal relative weight to the low-CE region the
        # tau knob lives in (linear MSE lets rare CE-15 labels dominate and
        # leaves small-k predictions badly under-calibrated).
        return F.mse_loss(torch.log1p(cols), torch.log1p(actual.detach()))
    return F.mse_loss(cols, actual.detach())


def predictor_loss(
    predicted: torch.Tensor,
    k: int,
    actual_per_sample_ce: torch.Tensor,
    log_space: bool = False,
) -> torch.Tensor:
    """MSE between the predictor's column for prefix length k and observed CE.

    predicted [B, N] (column k-1 = prefix of k vectors), actual [B] detached.
    """
    return _pred_mse(predicted[:, k - 1], actual_per_sample_ce, log_space)


def predictor_loss_per_k(
    predicted: torch.Tensor,
    ks: torch.Tensor,
    actual_per_sample_ce: torch.Tensor,
    log_space: bool = False,
) -> torch.Tensor:
    """Per-sample variant: row b is compared at its own prefix length ks[b]."""
    cols = predicted.gather(1, (ks - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
    return _pred_mse(cols, actual_per_sample_ce, log_space)
