from __future__ import annotations

import math
import warnings

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_len = self.pe.size(1)
        seq_len = x.size(1)
        if seq_len > max_len:
            warnings.warn(
                f"Input sequence length ({seq_len}) exceeds maximum positional encoding length ({max_len}).", stacklevel=2,
            )
            x = x[:, :max_len]
            seq_len = max_len
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ThoughtEncoder(nn.Module):
    """V1 — Current: learned seeds → GRU → cross-attn → norm."""

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 512, num_thoughts: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_thoughts = num_thoughts
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.thought_seed = nn.Parameter(torch.randn(1, num_thoughts, d_model) * 0.02)
        self.thought_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        max_len = self.positional_encoding.pe.size(1)
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, :max_len]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :max_len]
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        batch_size = input_ids.size(0)
        thoughts = self.thought_seed.expand(batch_size, -1, -1)
        thoughts, _ = self.thought_gru(thoughts)
        attended, _ = self.cross_attention(query=thoughts, key=encoded, value=encoded, key_padding_mask=padding_mask)
        return self.norm(thoughts + attended)


class ThoughtEncoderV2(nn.Module):
    """V2 — Proposed: encoder states → cross-attn (pooling) → GRU → hidden states."""

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 512, num_thoughts: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_thoughts = num_thoughts
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Learned queries pool encoder states into K thought vectors
        self.thought_queries = nn.Parameter(torch.randn(1, num_thoughts, d_model) * 0.02)
        self.pooler = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # GRU refines pooled vectors, maintaining sequential hierarchy
        self.thought_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        max_len = self.positional_encoding.pe.size(1)
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, :max_len]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :max_len]
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        # Pool encoder states into K thought vectors via learned queries
        batch_size = input_ids.size(0)
        queries = self.thought_queries.expand(batch_size, -1, -1)
        pooled, _ = self.pooler(query=queries, key=encoded, value=encoded, key_padding_mask=padding_mask)
        # GRU refines, all hidden states become thought vectors
        thoughts, _ = self.thought_gru(pooled)
        return self.norm(thoughts)

