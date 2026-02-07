from __future__ import annotations

import math
from dataclasses import dataclass

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
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ThoughtEncoder(nn.Module):
    """Text -> fixed-count thought vectors."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_thoughts: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_thoughts = num_thoughts

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.thought_seed = nn.Parameter(torch.randn(1, num_thoughts, d_model) * 0.02)
        self.thought_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        batch_size = input_ids.size(0)
        thoughts = self.thought_seed.expand(batch_size, -1, -1)
        thoughts, _ = self.thought_gru(thoughts)

        attended, _ = self.cross_attention(query=thoughts, key=encoded, value=encoded, key_padding_mask=padding_mask)
        return self.norm(thoughts + attended)


class ThoughtDecoder(nn.Module):
    """Thought vectors + shifted targets -> token logits."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        thought_vectors: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt = self.token_embedding(target_input_ids) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)

        seq_len = target_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=target_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        decoded = self.decoder(
            tgt=tgt,
            memory=thought_vectors,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
        )
        return self.lm_head(decoded)


@dataclass
class ThoughtVectorModel(nn.Module):
    encoder: ThoughtEncoder
    decoder: ThoughtDecoder

    def __init__(self, encoder: ThoughtEncoder, decoder: ThoughtDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reconstruct_logits(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        thoughts = self.encoder(input_ids, padding_mask)
        return self.decoder(thoughts, input_ids[:, :-1], None if padding_mask is None else padding_mask[:, :-1])
