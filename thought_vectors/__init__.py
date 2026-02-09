from __future__ import annotations

from importlib import import_module

__all__ = [
    "ThoughtEncoder",
    "ThoughtDecoder",
    "ThoughtVectorModel",
    "SimpleTokenizer",
    "normalize_apostrophes",
    "normalize_text",
    "load_groups_from_path",
    "training_step",
    "compute_dynamic_loss_target",
    "train_model",
    "encode",
    "decode_greedy",
    "find_minimum_vectors_for_target",
    "encode_with_compression",
]

_EXPORT_MAP = {
    "ThoughtEncoder": ("thought_vectors.model", "ThoughtEncoder"),
    "ThoughtDecoder": ("thought_vectors.model", "ThoughtDecoder"),
    "ThoughtVectorModel": ("thought_vectors.model", "ThoughtVectorModel"),
    "SimpleTokenizer": ("thought_vectors.tokenization", "SimpleTokenizer"),
    "normalize_apostrophes": ("thought_vectors.preprocessing", "normalize_apostrophes"),
    "normalize_text": ("thought_vectors.preprocessing", "normalize_text"),
    "load_groups_from_path": ("thought_vectors.data_loading", "load_groups_from_path"),
    "training_step": ("thought_vectors.train", "training_step"),
    "compute_dynamic_loss_target": ("thought_vectors.train", "compute_dynamic_loss_target"),
    "train_model": ("thought_vectors.train", "train_model"),
    "encode": ("thought_vectors.inference", "encode"),
    "decode_greedy": ("thought_vectors.inference", "decode_greedy"),
    "find_minimum_vectors_for_target": ("thought_vectors.inference", "find_minimum_vectors_for_target"),
    "encode_with_compression": ("thought_vectors.inference", "encode_with_compression"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'thought_vectors' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
