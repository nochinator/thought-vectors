"""ThoughtAutoencoder: tied-embedding encoder/decoder pair + loss predictor."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelCfg
from .decoder import ThoughtDecoder
from .encoder import ThoughtEncoder
from .predictor import LossPredictor

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class ThoughtAutoencoder(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        # One 16384x256 table serving encoder input, decoder input and LM head.
        # std = 1/sqrt(d): inputs are unit-scale after the sqrt(d) embedding
        # multiplier, and logits from LayerNorm'd states (norm sqrt(d)) are
        # unit-variance, so init CE starts at ~ln(vocab).
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.token_embedding.weight, std=cfg.d_model**-0.5)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()
        self.encoder = ThoughtEncoder(cfg, self.token_embedding)
        self.decoder = ThoughtDecoder(cfg, self.token_embedding)
        self.predictor = LossPredictor(cfg.d_model, cfg.num_thoughts)

    def encode(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(input_ids, padding_mask)

    def decode(
        self,
        thoughts: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        return self.decoder(thoughts, target_input_ids, target_padding_mask, causal=causal)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        k: int | None = None,
    ) -> torch.Tensor:
        """Teacher-forced reconstruction logits from a k-prefix of thoughts."""
        thoughts = self.encode(input_ids, padding_mask)
        if k is not None:
            thoughts = thoughts[:, :k]
        dec_in = input_ids[:, :-1]
        dec_pad = padding_mask[:, :-1] if padding_mask is not None else None
        return self.decode(thoughts, dec_in, dec_pad)

    def param_count(self) -> int:
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def unique_parameters(self):
        """Parameters deduplicated by storage (tied weights counted once)."""
        seen: set[int] = set()
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                yield p


def make_padding_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """True where padded (the convention nn.Transformer* expects)."""
    return input_ids == PAD_ID
