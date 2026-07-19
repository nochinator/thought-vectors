#!/usr/bin/env bash
# ── Round B2: baseline-LM recipe arms (docs/BASELINE_ABLATIONS.md) ──
#
# B1 winners: A class d384x5 (lr 6e-4 >> 3e-4, val_ce 2.72 vs 3.11);
# B class shallow shapes (640x8 / 576x10) at batch 16, lr transfer confirmed
# (512x12: 3.00 @ lr6 vs 4.03 @ lr3). B2 probes the pre-registered recipe
# axes on the winner shapes plus the lr axis B1 opened. Freeze recipe after.
#
#   A class (d384x5, base lr 6e-4):  A_lr10, A_b64, A_seq512, A_w400
#   B class (batch 16, lr 6e-4):     B_640x8_lr6, B_576x10_lr6
#
# Deferred with rationale (see doc): loss-masking variant (thinker parity
# dictates all-non-first-turn targets).
#
# Usage:  scripts/ablate_lm_b2.sh all      (or a list of arm names)
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-2700}

run() {
  local name=$1; shift
  echo "=== [$(date +%H:%M:%S)] B2 arm: $name ==="
  .venv/bin/tv-train-lm --config configs/b1_lm.yaml \
    run.name="$name" train.max_seconds="$DUR" train.lr=6.0e-4 "$@"
}

arm() {
  case $1 in
    A_lr10)   run b2_lm_A_lr10   train.lr=1.0e-3 ;;
    A_b64)    run b2_lm_A_b64    train.batch_size=64 ;;
    A_seq512) run b2_lm_A_seq512 lm.max_seq_len=512 ;;
    A_w400)   run b2_lm_A_w400   train.warmup_steps=400 ;;
    B_640x8_lr6)  run b2_lm_B_640x8_lr6  lm.d_model=640 lm.layers=8  lm.ffn_dim=2560 lm.nhead=8 train.batch_size=16 ;;
    B_576x10_lr6) run b2_lm_B_576x10_lr6 lm.d_model=576 lm.layers=10 lm.ffn_dim=2304 lm.nhead=8 train.batch_size=16 ;;
    # confirm arm: A-class lr winner (1e-3) at the B-class winner shape
    B_640x8_lr10) run b2_lm_B_640x8_lr10 lm.d_model=640 lm.layers=8 lm.ffn_dim=2560 lm.nhead=8 train.batch_size=16 train.lr=1.0e-3 ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

ARMS=("$@")
if [ "${1:-all}" = "all" ]; then
  ARMS=(A_lr10 A_b64 A_seq512 A_w400 B_640x8_lr6 B_576x10_lr6)
fi
# retry each arm once on gfx1031 flakes; keep the bracket alive either way
for a in "${ARMS[@]}"; do
  n="b2_lm_$a"
  for attempt in 1 2; do
    if arm "$a"; then break; fi
    echo "!!! [$(date +%H:%M:%S)] arm $a failed (attempt $attempt)"
    mv "logs/$n" "logs/${n}_crash${attempt}" 2>/dev/null || true
    mv "checkpoints/$n" "checkpoints/${n}_crash${attempt}" 2>/dev/null || true
    if [ "$attempt" = 2 ]; then echo "!!! arm $a skipped after 2 failures"; fi
    sleep 10
  done
done
echo "=== B2 arms complete ==="
for a in "${ARMS[@]}"; do
  n="b2_lm_$a"
  best=$(grep -o '"val_ce": [0-9.]*' "logs/$n/metrics.jsonl" 2>/dev/null | sort -t' ' -k2 -n | head -1 || true)
  echo "$n  best $best"
done
