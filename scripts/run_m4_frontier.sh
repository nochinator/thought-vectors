#!/usr/bin/env bash
# M4 frontier thinker run: round-2 winner recipe (WTA4) at 12h.
set -uo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
.venv/bin/tv-train-thinker --config configs/m4_thinker.yaml \
  run.name=m4_frontier \
  train.max_seconds=43200 train.max_steps=225000 \
  train.warmup_steps=2000 train.val_every=2000 train.ckpt_every=5000 \
  train.keep_ckpts=3 \
  thinker.n_hypotheses=4 thinker.w_decoder=0.5 thinker.k_ctx=8 \
  2>&1 | tee logs/m4_frontier.out
if [[ -f checkpoints/m4_frontier/best.pt ]]; then
  .venv/bin/python scripts/eval_thinker.py --ckpt checkpoints/m4_frontier/best.pt \
    --dump logs/m4_frontier.samples.txt 2>&1 | tee -a logs/m4_frontier.out
fi
