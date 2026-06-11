"""Blended k-sampler: which thought-prefix length to train on this batch.

mode="full"       -> always k=N (pure autoencoder, M1).
mode="blended"    -> 10% k=N, 45% uniform [min_k, N], 45% length-aware ratio
                     bands skewed toward aggressive compression (M2+). The blend
                     replicates the 50/50 uniform+skewed mix that empirically
                     improved 3:1-6:1 compression in the prior project.
mode="per_sample" -> the blended distribution drawn independently PER SAMPLE
                     (each row trains a different prefix length in one decode,
                     via memory key-padding masks — denser k coverage per FLOP).
"""

from __future__ import annotations

import random

import torch

from .config import KSamplerCfg


class KSampler:
    def __init__(self, cfg: KSamplerCfg, num_thoughts: int, rng: random.Random | None = None):
        self.cfg = cfg
        self.n = num_thoughts
        self.rng = rng or random.Random()

    def sample(self, mean_token_len: float) -> int:
        if self.cfg.mode == "full":
            return self.n
        r = self.rng.random()
        if r < self.cfg.full_frac:
            return self.n
        if r < self.cfg.full_frac + self.cfg.uniform_frac:
            return self.rng.randint(self.cfg.min_k, self.n)
        bands = self.cfg.ratio_bands
        weights = [b[2] for b in bands]
        lo, hi, _ = self.rng.choices(bands, weights=weights, k=1)[0]
        ratio = self.rng.uniform(lo, hi)
        k = round(ratio * mean_token_len)
        return max(self.cfg.min_k, min(self.n, k))

    def sample_per_sample(self, token_lens: torch.Tensor) -> torch.Tensor:
        """One k per sample [B], each drawn from the blended distribution
        against that sample's own length (not the batch mean)."""
        return torch.tensor(
            [self.sample(float(length)) for length in token_lens], dtype=torch.long
        )

    def sample_distinct(self, mean_token_len: float, count: int, exclude: int) -> list[int]:
        """Extra k values for predictor labels, distinct from each other and `exclude`."""
        ks: set[int] = set()
        attempts = 0
        while len(ks) < count and attempts < 20 * count:
            k = self.rng.randint(self.cfg.min_k, self.n)
            if k != exclude:
                ks.add(k)
            attempts += 1
        return sorted(ks)
