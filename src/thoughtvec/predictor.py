"""LossPredictor: estimate per-sample reconstruction CE at every prefix length.

Input is mean-pooled thoughts, DETACHED by the caller — gradients never reach
the encoder/decoder, which is what lets it train jointly without the circular
search problem the original project hit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LossPredictor(nn.Module):
    def __init__(self, d_model: int, num_thoughts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_thoughts),
        )

    def forward(self, thoughts: torch.Tensor) -> torch.Tensor:
        """thoughts [B, N, d] (detach before calling) -> predicted CE [B, N].

        Column k-1 is the predicted reconstruction CE when decoding from the
        first k thought vectors. Softplus keeps predictions non-negative.
        """
        pooled = thoughts.mean(dim=1)
        return nn.functional.softplus(self.net(pooled))
