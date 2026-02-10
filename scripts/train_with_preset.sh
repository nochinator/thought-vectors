#!/usr/bin/env bash
set -euo pipefail

# Edit these values once, then run this script instead of typing the full command.
STSB="../datasets/STSB_train.csv"
SNLI="../datasets/SNLI_train.csv"
MINIPILE="../datasets/minipile.csv"
C41="../datasets/C4subset-1.csv"
C42="../datasets/C4subset-2.csv"
OUTPUT_PATH="artifacts/thought_vectors0.1-3.pt"
MODEL_PATH="artifacts/thought_vectors0.1-2.pt"

python train_model.py \
  --data "$MINIPILE" \
  --resume-from "$MODEL_PATH" \
  --tokenizer-count-memory-limit 24 \
  --epochs 1 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --length-penalty 0.01 \
  --num-thoughts 16 \
  --d-model 512 \
  --layers 4 \
  --heads 8 \
  --dropout 0.1 \
  --max-seq-len 2048 \
  --max-vectors 16 \
  --selection-stride 2 \
  --target-start 3 \
  --target-end 0.2 \
  --target-length-weight 0.65 \
  --target-noise-std 0.07 \
  --target-extreme-prob 0.12 \
  --log-every 1 \
  --output "$OUTPUT_PATH"
