"""
BitThought configuration presets.

4 model sizes spanning 1M-85M parameters.
Each config defines the full architecture.

Tokenizer strategy:
  - Tiny / Small: custom BPE (~4K vocab) trained on the dataset
  - Medium / Large: pre-existing GPT-2 tokenizer (50K vocab)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BitThoughtConfig:
    """Configuration for a BitThought model."""

    # Vocabulary — set by tokenizer at runtime
    vocab_size: int = 4000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Tokenizer selection: "" = train custom BPE, "gpt2" = use pre-existing
    tokenizer_name: str = ""

    # Core dimensions
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512

    # Thought vector parameters
    num_thoughts: int = 16
    thought_dim: int = 0  # 0 = same as d_model

    # Gradient checkpointing — saves ~30% memory at ~20% throughput cost
    use_gradient_checkpointing: bool = False

    # K-predictor: regression head that outputs number of vectors to keep
    use_k_predictor: bool = True
    k_predictor_hidden: int = 0  # 0 = auto (d_model // 4)
    k_temperature: float = 10.0  # soft mask temperature for differentiable K
    k_noise: float = 0.1         # ±10% random noise during compression training

    # Compression curriculum
    target_ratio_start: float = 0.7    # starting tokens-per-vector (0.7 = no real compression yet)
    target_ratio_inc: float = 0.1      # increment when accuracy threshold met
    target_ratio_max: float = 20.0     # hard ceiling
    acc_threshold: float = 0.95        # rolling accuracy required to advance
    acc_window: int = 1000             # rolling window size in batches
    stop_weight: float = 1.0           # (reserved for future use)

    @property
    def effective_thought_dim(self) -> int:
        return self.thought_dim if self.thought_dim > 0 else self.d_model

    @property
    def k_hidden_dim(self) -> int:
        return self.k_predictor_hidden if self.k_predictor_hidden > 0 else max(64, self.d_model // 4)


PRESET_REGISTRY: dict[str, BitThoughtConfig] = {}


def _register(name: str, cfg: BitThoughtConfig) -> BitThoughtConfig:
    PRESET_REGISTRY[name.lower()] = cfg
    return cfg


# ---------------------------------------------------------------------------
# Tiny     ~1.1M params    d=64   L=2  H=2   custom BPE
# ---------------------------------------------------------------------------
_register("tiny", BitThoughtConfig(
    d_model=64,
    nhead=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    num_thoughts=8,
    tokenizer_name="",
    target_ratio_start=0.7,
))

# ---------------------------------------------------------------------------
# Small    ~4.5M params    d=128  L=3  H=4   custom BPE
# ---------------------------------------------------------------------------
_register("small", BitThoughtConfig(
    d_model=128,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    num_thoughts=12,
    tokenizer_name="",
    target_ratio_start=0.7,
))

# ---------------------------------------------------------------------------
# Medium   ~14M params     d=256  L=4  H=8   custom BPE
# ---------------------------------------------------------------------------
_register("medium512", BitThoughtConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=2048,
    num_thoughts=16,
    tokenizer_name="gpt2",
    target_ratio_start=0.7,
))
_register("medium768", BitThoughtConfig(
    d_model=768,
    nhead=8,
    num_encoder_layers=5,
    num_decoder_layers=5,
    dim_feedforward=3072,
    num_thoughts=128,
    tokenizer_name="llama2",
    target_ratio_start=0.7,
))
_register("medium512-llama2", BitThoughtConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=2048,
    num_thoughts=32,
    tokenizer_name="llama2",
    target_ratio_start=0.7,
))

# ---------------------------------------------------------------------------
# Medium   ~14M params     d=256  L=4  H=8   custom BPE
# ---------------------------------------------------------------------------
_register("medium", BitThoughtConfig(
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    num_thoughts=16,
    tokenizer_name="",
    target_ratio_start=0.7,
))

# ---------------------------------------------------------------------------
# Large    ~85M params     d=384  L=6  H=8   GPT-2 tokenizer
# ---------------------------------------------------------------------------
_register("large", BitThoughtConfig(
    d_model=384,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=1536,
    num_thoughts=24,
    tokenizer_name="gpt2",
    target_ratio_start=0.7,
))


class ModelPresets:
    """Access model configuration presets."""

    @staticmethod
    def get(name: str) -> BitThoughtConfig:
        name = name.lower()
        if name not in PRESET_REGISTRY:
            raise KeyError(
                f"Unknown preset {name!r}. Available: {list(PRESET_REGISTRY)}"
            )
        return PRESET_REGISTRY[name]

    @staticmethod
    def list_names() -> list[str]:
        return list(PRESET_REGISTRY)

    @staticmethod
    def list_with_params() -> dict[str, dict]:
        """Return measured param counts (builds each model once, caches)."""
        if hasattr(ModelPresets, "_param_cache"):
            return ModelPresets._param_cache
        _GPT2_VOCAB = 50257
        _LLAMA2_VOCAB = 32000
        info = {}
        import copy
        for name, cfg in PRESET_REGISTRY.items():
            from bitthought.model import BitThoughtModel
            if cfg.tokenizer_name == "gpt2":
                v = _GPT2_VOCAB
            elif cfg.tokenizer_name == "llama2":
                v = _LLAMA2_VOCAB
            else:
                v = cfg.vocab_size
            c = copy.copy(cfg)
            c.vocab_size = v
            m = BitThoughtModel(c)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            emb_head = v * cfg.d_model
            pct = emb_head / n * 100
            info[name] = {
                "fp32_params": n,
                "emb_head_pct": pct,
                "d_model": cfg.d_model,
                "layers": cfg.num_encoder_layers,
                "nhead": cfg.nhead,
                "num_thoughts": cfg.num_thoughts,
                "tokenizer": cfg.tokenizer_name if cfg.tokenizer_name else "custom-bpe",
            }
        ModelPresets._param_cache = info
        return info

    @staticmethod
    def print_table():
        info = ModelPresets.list_with_params()
        print(f"{'Preset':<8} {'Params':<12} {'Emb%':<6} "
              f"{'d_model':<8} {'Layers':<8} {'Heads':<8} {'Thoughts':<10} {'Tok':<12}")
        print("-" * 66)
        for name, d in info.items():
            p = f"{d['fp32_params']:,}"
            e = f"{d['emb_head_pct']:.0f}%"
            print(f"{name:<8} {p:<12} {e:<6} "
                  f"{d['d_model']:<8} {d['layers']:<8} {d['nhead']:<8} {d['num_thoughts']:<10} {d['tokenizer']:<12}")
