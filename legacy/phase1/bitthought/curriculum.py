"""
BitThought — Curriculum data loading for progressive training.

Stages progress from clean/simple → diverse → complex/messy:
  1. STSB   (5.7K clean sentence pairs)
  2. SNLI   (700K reasoning examples)
  3. minipile (5.6G diverse web text)
  4. C4 subset (763M real-world web data)
"""

from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from bitthought.data import load_groups, ThoughtDataset, collate_thought_batch
from bitthought.tokenization import ThoughtTokenizer


# Resolved from bitthought/curriculum.py → AI_construction/datasets
DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


CURRICULUM_STAGES = [
    {
        "name": "stsb",
        "path": DATASET_DIR / "STSB_train.csv",
        "epochs": 5,
        "note": "Clean sentence pairs",
    },
    {
        "name": "snli",
        "path": DATASET_DIR / "SNLI_train.csv",
        "epochs": 3,
        "note": "Reasoning examples",
    },
    {
        "name": "minipile",
        "path": DATASET_DIR / "minipile.csv",
        "epochs": 3,
        "note": "Diverse web text",
    },
    {
        "name": "c4",
        "path": DATASET_DIR / "C4subset-1.csv",
        "epochs": 2,
        "note": "Real-world web data",
    },
]


def _text_iterator(path: Path, preprocess: bool = True) -> Iterator[str]:
    """Stream texts from a dataset file without loading all into memory."""
    groups = load_groups(path, preprocess=preprocess)
    for group in groups:
        for text in group:
            yield text


class CurriculumTrainer:
    """Manages progressive curriculum training across multiple datasets.

    Usage:
        ct = CurriculumTrainer(config, tokenizer, device)
        ct.train(model, epochs_per_stage=[5, 3, 3, 2])
    """

    def __init__(self, config, tokenizer, pad_token_id, device,
                 batch_size: int = 8):
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.device = device
        self.batch_size = batch_size

    def _make_loader(self, groups: list[list[str]]) -> DataLoader:
        """Create a DataLoader from grouped texts."""
        dataset = ThoughtDataset(groups)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_thought_batch(
                batch, self.tokenizer.encode,
                self.pad_token_id, self.config.max_seq_len,
            ),
        )

    def train_stage(self, model, stage: dict, epochs: int,
                    train_fn, **train_kwargs):
        """Train on one curriculum stage."""
        path = stage["path"]
        if not path.exists():
            print(f"[curriculum] SKIP {stage['name']}: {path} not found")
            return []

        print(f"\n{'='*60}")
        print(f"[curriculum] Stage: {stage['name']} — {stage['note']}")
        print(f"[curriculum] Data: {path}")
        print(f"[curriculum] Epochs: {epochs}")
        print(f"{'='*60}")

        groups = load_groups(path, preprocess=True)
        print(f"[curriculum] Loaded {len(groups)} groups")

        loader = self._make_loader(groups)
        print(f"[curriculum] Batches per epoch: {len(loader)}")

        history = train_fn(
            model, self.config, groups,
            self.tokenizer.encode, self.pad_token_id,
            device=self.device,
            epochs=epochs,
            batch_size=self.batch_size,
            **train_kwargs,
        )
        return history

    def train(self, model, train_fn,
              stages: list[dict] | None = None,
              epochs_per_stage: list[int] | None = None,
              **train_kwargs):
        """Run full curriculum training.

        Args:
            model: BitThoughtModel
            train_fn: train_model function
            stages: stage configs (defaults to CURRICULUM_STAGES)
            epochs_per_stage: override epochs per stage
            **train_kwargs: passed to train_fn
        """
        if stages is None:
            stages = CURRICULUM_STAGES
        if epochs_per_stage is None:
            epochs_per_stage = [s["epochs"] for s in stages]

        all_history = {}
        for stage, n_epochs in zip(stages, epochs_per_stage):
            history = self.train_stage(model, stage, n_epochs,
                                       train_fn, **train_kwargs)
            all_history[stage["name"]] = history
        return all_history
