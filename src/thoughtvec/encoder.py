"""ThoughtEncoder: text -> ordered thought vectors.

Wiring (empirically essential, see RESEARCH_LOG.md / prior project's RESEARCH.md):
learned seed -> GRU (builds the ordered "structure scaffold") -> cross-attention
into the encoded text (fills each slot) -> LayerNorm(residual).

The GRU input is the batch-independent seed, so we run it once on [1, N, d]
and expand the result to the batch — mathematically identical to running it
per batch row, and makes the deep-narrow GRU's cost independent of batch size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelCfg
from .modules import SinusoidalPositions


class ThoughtEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg, token_embedding: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.token_embedding = token_embedding
        self.positions = SinusoidalPositions(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm stacks need a final norm or the residual stream (carrying
        # the sqrt(d)-scaled embeddings) reaches cross-attention unnormalized.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.enc_layers, norm=nn.LayerNorm(cfg.d_model)
        )

        self.thought_seed = nn.Parameter(torch.randn(1, cfg.num_thoughts, cfg.d_model) * 0.02)
        self.thought_gru = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(
            cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        # VAE posterior heads; forward() returns deterministic mu, so these are
        # inert until kl_beta > 0 enables encode_with_kl in training.
        self.mu_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.logvar_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def _scaffold(self, batch_size: int) -> torch.Tensor:
        out, _ = self.thought_gru(self.thought_seed)
        return out.expand(batch_size, -1, -1)

    def _encode(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positions(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        scaffold = self._scaffold(input_ids.size(0))
        attended, _ = self.cross_attention(
            query=scaffold, key=encoded, value=encoded, key_padding_mask=padding_mask
        )
        return self.norm(scaffold + attended)

    def forward(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Deterministic thought vectors [B, N, d] (the VAE mean)."""
        return self.mu_proj(self._encode(input_ids, padding_mask))

    def encode_with_kl(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(sampled z, mu, logvar) for VAE training."""
        h = self._encode(input_ids, padding_mask)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar
