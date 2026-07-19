#!/usr/bin/env bash
# ── Round B3 long-run wrapper: auto-resume across gfx1031 crashes ──
#
# Page faults arrive every ~30-40 min under sustained load (see the B1
# results log), so a 12-24h run must resume, not restart: LMTrainer
# checkpoints optimizer state + cumulative elapsed, and the wall-clock LR
# schedule and max_seconds stop both continue where they left off. At
# ckpt_every=1000 steps a crash costs a few minutes, not the run.
#
# Usage: scripts/train_lm_b3.sh <run_name> <max_seconds> [override ...]
# e.g.:  scripts/train_lm_b3.sh b3_lm_48m_24h 86400 lm.d_model=640 ...
set -uo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
NAME=$1; DUR=$2; shift 2

for i in $(seq 1 200); do
  RESUME=()
  [ -f "checkpoints/$NAME/last.pt" ] && RESUME=(--resume "checkpoints/$NAME/last.pt")
  echo "=== [$(date +%H:%M:%S)] $NAME attempt $i ${RESUME[*]:-(fresh)} ==="
  if .venv/bin/tv-train-lm --config configs/b1_lm.yaml "${RESUME[@]}" \
       run.name="$NAME" train.max_seconds="$DUR" "$@"; then
    echo "=== [$(date +%H:%M:%S)] $NAME finished cleanly ==="
    exit 0
  fi
  echo "!!! [$(date +%H:%M:%S)] $NAME crashed (attempt $i); resuming in 15s"
  sleep 15
done
echo "!!! $NAME: exceeded 200 restarts, giving up"
exit 1
