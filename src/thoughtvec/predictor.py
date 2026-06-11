"""LossPredictor: estimate per-sample reconstruction CE at every prefix length.

Input is mean-pooled thoughts, DETACHED by the caller — gradients never reach
the encoder/decoder, which is what lets it train jointly without the circular
search problem the original project hit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LossPredictor(nn.Module):
    def __init__(self, d_model: int, num_thoughts: int, monotone: bool = False) -> None:
        super().__init__()
        self.monotone = monotone
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_thoughts),
        )

    def forward(self, thoughts: torch.Tensor) -> torch.Tensor:
        """thoughts [B, N, d] (detach before calling) -> predicted CE [B, N].

        Column k-1 is the predicted reconstruction CE when decoding from the
        first k thought vectors. Softplus keeps predictions non-negative.

        monotone=True parameterizes the curve as a right-to-left cumulative
        sum of non-negative increments, so predicted CE is non-increasing in
        k by construction. Without this, under-predictions at tiny k make
        "smallest k with pred <= tau" pick garbage prefixes (the tau cliff
        seen in the 2026-06-11 W0-W5 evals).
        """
        pooled = thoughts.mean(dim=1)
        raw = self.net(pooled)
        if self.monotone:
            vals = nn.functional.softplus(raw)
            return vals.flip(1).cumsum(dim=1).flip(1)
        return nn.functional.softplus(raw)
