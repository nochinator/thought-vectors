# Thought Vectors

Implementation scaffold for the Thought Vector System specification.

## Included

- `ThoughtEncoder` with token embeddings, positional encoding, transformer encoder, GRU thought generation, and cross-attention.
- `ThoughtDecoder` with transformer decoder, causal masking, and vocabulary projection head.
- Group-based dataset utilities (`GroupTextDataset`, `collate_group_batch`).
- Data loading for `.json`, `.jsonl`, and `.csv` (`load_groups_from_path`); CSV uses the first column as text.
- Preprocessing (`normalize_apostrophes`, `normalize_text`) for mixed apostrophes and general cleanup.
- Training helpers (`training_step`, `train_model`) with reconstruction loss + length penalty.
- Dynamic target compression curriculum (`compute_dynamic_loss_target`) for training.
- Inference helpers (`encode`, `decode_greedy`, `find_minimum_vectors_for_target`, `encode_with_compression`).
- `SimpleTokenizer` with regex tokenization, lowercasing, punctuation support, and deterministic vocab building for local experiments.

## Training script

Use `scripts/train_model.py` with either:

- `.json`: top-level `list[list[str]]` or list of strings,
- `.jsonl`: one object per line with `{"texts": ["...", "..."]}`,
- `.csv`: first column only is used as training text.

Example:

```bash
python scripts/train_model.py --data data/dataset.csv --epochs 10 --batch-size 8 --sample-every 8 --output artifacts/thought_vectors.pt
```

By default, text preprocessing is enabled (apostrophe normalization + cleanup). Use `--no-preprocess` to disable it.


To continue training from a checkpoint:

```bash
python scripts/train_model.py --data data/dataset.csv --resume-from artifacts/thought_vectors.pt --epochs 5 --output artifacts/thought_vectors.pt
```

When resuming, tokenizer vocabulary is extended with tokens from the new dataset, and model embeddings/LM head are expanded to match (avoids `<unk>` spikes when switching datasets).

For large datasets, bound tokenizer RAM with `--tokenizer-count-memory-limit` and optionally set `--tokenizer-max-vocab-size` / `--tokenizer-min-frequency` to constrain vocab growth during fitting.

If you press `Ctrl+C` during training, the current state is saved to checkpoint before exit.

The training loop logs detailed progress per epoch/batch, compression target details, trainable parameter counts, and a reconstruction sample every 8 batches by default (`--sample-every`).

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
