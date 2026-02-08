# Thought Vectors

Implementation scaffold for the Thought Vector System specification.

## Included

- `ThoughtEncoder` with token embeddings, positional encoding, transformer encoder, GRU thought generation, and cross-attention.
- `ThoughtDecoder` with transformer decoder, causal masking, and vocabulary projection head.
- Group-based dataset utilities (`GroupTextDataset`, `collate_group_batch`).
- Training helpers (`training_step`, `train_model`) with reconstruction loss + length penalty.
- Inference helpers (`encode`, `decode_greedy`, `encode_with_compression`).
- `SimpleTokenizer` for local smoke tests and tiny experiments.

## Training script

Use `scripts/train_model.py` with either:

- `.json`: top-level `list[list[str]]`, or
- `.jsonl`: one object per line with `{"texts": ["...", "..."]}`.

Example:

```bash
python scripts/train_model.py --data data/groups.json --epochs 10 --batch-size 8 --output artifacts/thought_vectors.pt
```

## Unit tests

```bash
pytest -q
```

`tests/test_train_two_models.py` trains two small models on a tiny grouped dataset to verify the training loop runs and losses trend down.
