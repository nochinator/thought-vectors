#!/usr/bin/env bash
set -e
cd /home/nochi/vault/projects/AI_construction/legacy
source .venv/bin/activate
cd thought-vectors-main

echo "=== Stage 0: Decoder fine-tune ==="
python -u scripts/train_summarization.py \
  --checkpoint artifacts/vae_compressed_nar.pt \
  --tokenizer-model /tmp/sp_c4_16k.model \
  --article-data /tmp/clean_conv_input.csv \
  --summary-data /tmp/clean_conv_output.csv \
  --thinker-layers 6 --thinker-heads 4 \
  --dropout 0.1 \
  --max-article-len 80 --max-summary-len 80 \
  --batch-size 16 --lr 1e-4 --epochs 10 \
  --freeze-thinker --unfreeze-decoder \
  --log-every 200 --sample-every 400 \
  --output artifacts/thinker_s0_clean.pt

echo "=== Stage 1: Thinker training ==="
python -u scripts/train_stage1_clean.py

echo "=== Done ==="
