"""ThoughtDecoder: thought-vector prefix + shifted targets -> token logits.

causal=False with all-pad target inputs is NAR mode: the decoder must predict
every token from the thoughts + positions alone (M3b fine-tune).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelCfg
from .modules import SinusoidalPositions


class ThoughtDecoder(nn.Module):
    def __init__(self, cfg: ModelCfg, token_embedding: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.token_embedding = token_embedding
        self.positions = SinusoidalPositions(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm stacks need a final norm; without it the unnormalized
        # residual stream hits the tied LM head and logits start at std ~150
        # (init CE in the thousands instead of ln(vocab)).
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=cfg.dec_layers, norm=nn.LayerNorm(cfg.d_model)
        )
        # Dropout on the thought memory: the decoder can't rely on any single
        # slot always being present, so it must actually read the thoughts
        # (posterior-collapse guard).
        self.thought_dropout = nn.Dropout(cfg.thought_dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = token_embedding.weight  # tied

        # Learned cross-attention position bias: decoder position i attending
        # to thought slot j gets bias * (i - j), aligning the GRU's slot
        # ordering with output token ordering.
        if cfg.position_attn_bias:
            self.position_attn_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.position_attn_bias = None

    def forward(
        self,
        thoughts: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        causal: bool = True,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = target_input_ids.size(1)
        tgt = self.token_embedding(target_input_ids) * math.sqrt(self.d_model)
        tgt = self.positions(tgt)

        tgt_mask = None
        if causal:
            tgt_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=tgt.device, dtype=torch.bool), diagonal=1
            )

        thoughts = self.thought_dropout(thoughts)

        if memory_padding_mask is not None and memory_padding_mask.dtype == torch.bool:
            # Float position bias + bool padding mask is a deprecated combo;
            # convert to additive float.
            memory_padding_mask = torch.zeros_like(
                memory_padding_mask, dtype=thoughts.dtype
            ).masked_fill_(memory_padding_mask, float("-inf"))

        memory_mask = None
        if self.position_attn_bias is not None:
            mem_len = thoughts.size(1)
            tgt_pos = torch.arange(seq_len, device=tgt.device, dtype=thoughts.dtype)
            mem_pos = torch.arange(mem_len, device=tgt.device, dtype=thoughts.dtype)
            memory_mask = self.position_attn_bias * (tgt_pos[:, None] - mem_pos[None, :])

        decoded = self.decoder(
            tgt=tgt,
            memory=thoughts,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.lm_head(decoded)
