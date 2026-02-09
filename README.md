# Thought Vectors

Implementation scaffold for the Thought Vector System specification.

## Included

- `ThoughtEncoder` with token embeddings, positional encoding, transformer encoder, GRU thought generation, and cross-attention.
- `ThoughtDecoder` with transformer decoder, causal masking, and vocabulary projection head.
- Group-based dataset utilities (`GroupTextDataset`, `collate_group_batch`).
- Training helpers (`training_step`, `train_model`) with reconstruction loss + length penalty.
- Dynamic target compression curriculum (`compute_dynamic_loss_target`) for training.
- Inference helpers (`encode`, `decode_greedy`, `find_minimum_vectors_for_target`, `encode_with_compression`).
- `SimpleTokenizer` for local smoke tests and tiny experiments.

## Training script

Use `scripts/train_model.py` with either:

- `.json`: top-level `list[list[str]]`, or
- `.jsonl`: one object per line with `{"texts": ["...", "..."]}`.

Example:

```bash
python scripts/train_model.py --data data/groups.json --epochs 10 --batch-size 8 --output artifacts/thought_vectors.pt
```

The training loop now logs detailed progress per epoch/batch and supports dynamic loss-target compression controls.

## Preset runner script

To avoid retyping params, edit and run:

```bash
./scripts/train_with_preset.sh
```

## Interactive loop

After training, run:

```bash
python scripts/interact_model.py --checkpoint artifacts/thought_vectors.pt --loss-target 0.6 --max-vectors 16
```

This opens an input loop, reconstructs each input text, and prints:
- selected vector count,
- reconstruction loss,
- per-prefix loss curve used for vector-count selection.

## Unit tests

```bash
pytest -q
```

`tests/test_train_two_models.py` trains two small models on a tiny grouped dataset to verify the training loop runs and losses trend down.
