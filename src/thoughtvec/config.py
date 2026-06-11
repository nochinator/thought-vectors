"""Dataclass-based config with YAML loading and dotted CLI overrides.

Usage:
    cfg = load_config("configs/m1_autoencoder.yaml", ["train.lr=1e-4", "model.num_thoughts=256"])
Every run/checkpoint embeds the resolved dict from `to_dict(cfg)`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelCfg:
    vocab_size: int = 16384
    d_model: int = 256
    nhead: int = 4
    ffn_dim: int = 2048
    enc_layers: int = 4
    dec_layers: int = 4
    dropout: float = 0.1
    thought_dropout: float = 0.1
    max_seq_len: int = 128
    num_thoughts: int = 128
    activation: str = "gelu"  # "relu" fallback for ROCm NaN bisection
    position_attn_bias: bool = True


@dataclass
class KSamplerCfg:
    mode: str = "full"  # "full" (always k=N), "blended", or "per_sample"
    min_k: int = 2
    full_frac: float = 0.10
    uniform_frac: float = 0.45
    # ratio-skewed remainder: list of (lo, hi, weight) over k/token_len ratios
    ratio_bands: list = field(
        default_factory=lambda: [
            [0.4, 0.6, 0.30],
            [0.25, 0.4, 0.30],
            [0.15, 0.25, 0.20],
            [0.6, 1.5, 0.20],
        ]
    )


@dataclass
class RegCfg:
    kl_beta: float = 0.0           # VAE KL weight
    kl_warmup_steps: int = 1000
    noise_std: float = 0.0          # gaussian noise on thoughts
    mixup_prob: float = 0.0         # convex blend of batch items
    nar: bool = False               # always non-autoregressive (legacy M3b mode)
    nar_frac: float = 0.0           # probability per batch of NAR reconstruction
    word_dropout: float = 0.0       # blank this frac of decoder input tokens; blanked
                                    # positions are only predictable via the thoughts
                                    # (anti-LM-attractor, Bowman et al. 2016)


@dataclass
class DataCfg:
    shard_dir: str = "data/c4_train"
    val_shard_dir: str = ""         # default: <shard_dir>_val
    num_workers: int = 2


@dataclass
class TrainCfg:
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 60000
    max_seconds: int = 0            # hard wall-clock stop, 0 = off (LR schedule still follows max_steps)
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    predictor_weight: float = 1.0
    predictor_extra_k: int = 0      # extra no-grad decodes per step for predictor labels
    anchor_full_k_weight: float = 0.0  # extra full-k decode loss (anchors top-end)
    anchor_every: int = 1           # apply the anchor decode every Nth step (it costs 35-50%)
    detach_encoder_below_k: int = 0 # legacy Phase-1 trick, off by default
    amp: str = "off"                # "off" | "bf16" (experimental on RDNA2 — abandon at first NaN)
    seed: int = 1234
    log_every: int = 50
    sample_every: int = 500
    val_every: int = 2000
    ckpt_every: int = 2000
    keep_ckpts: int = 3


@dataclass
class RunCfg:
    name: str = "run"
    out_dir: str = "checkpoints"
    log_dir: str = "logs"
    tokenizer_path: str = "artifacts/tokenizer/spm16k_bpe.model"
    device: str = "cuda"


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    ksampler: KSamplerCfg = field(default_factory=KSamplerCfg)
    reg: RegCfg = field(default_factory=RegCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    run: RunCfg = field(default_factory=RunCfg)


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return yaml.safe_load(value)
    return value


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key.subkey=value, got: {ov!r}")
        key, value = ov.split("=", 1)
        parts = key.split(".")
        obj: Any = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise AttributeError(f"Unknown config key: {key}")
        setattr(obj, leaf, _coerce(value, getattr(obj, leaf)))
    return cfg


def _update_dataclass(obj: Any, data: dict) -> None:
    for k, v in data.items():
        if not hasattr(obj, k):
            raise AttributeError(f"Unknown config key: {k} on {type(obj).__name__}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur):
            _update_dataclass(cur, v)
        elif isinstance(v, str) and isinstance(cur, (int, float)) and not isinstance(cur, bool):
            # YAML 1.1 parses "3e-4" (no dot) as a string, not a float.
            setattr(obj, k, _coerce(v, cur))
        else:
            setattr(obj, k, v)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
        _update_dataclass(cfg, data)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def from_dict(data: dict) -> Config:
    cfg = Config()
    _update_dataclass(cfg, data)
    return cfg
