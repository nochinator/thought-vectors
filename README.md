# thoughtvec

Text → ordered "thought vectors" → text. An encoder compresses text into N
latent vectors ordered by importance; any prefix of k vectors decodes back to
text, so compression is a decode-time choice. A small predictor estimates
reconstruction quality per prefix length, enabling adaptive compression.
Eventually a "thinker" transformer operates directly in thought-vector space
for latent-space conversation.

Clean-room recreation of the system in `../RESEARCH.md` / `../AI_construction`.
Experiment history: `RESEARCH_LOG.md`.

## Setup (once)

```bash
scripts/setup_env.sh          # venv + torch ROCm + GPU check
.venv/bin/tv-tokenizer train \
  --corpus ../AI_construction/datasets/C4subset-1.csv:4 \
  --corpus ../AI_construction/datasets/minipile.csv:40
.venv/bin/tv-pretokenize --csv ../AI_construction/datasets/C4subset-1.csv \
  --out data/c4_500k --max-rows 500000
```

## Train / eval

```bash
scripts/train.sh --config configs/m1_autoencoder.yaml        # baseline AE
scripts/train.sh --config configs/m2_compression.yaml \
  --init-from checkpoints/m1_autoencoder/best.pt             # compression
.venv/bin/tv-eval --ckpt checkpoints/m2_compression/best.pt --shard data/c4_500k_val
```

All runs go through `scripts/train.sh` (sets the gfx1031 ROCm guards).
Tests: `.venv/bin/pytest`.
