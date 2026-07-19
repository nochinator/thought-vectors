#!/usr/bin/env bash
# ── Round B1: baseline-LM shape bracket (docs/BASELINE_ABLATIONS.md) ──
#
# Textbook decoder-only LM on the thinker's dialogue shards, equal wall-clock
# (45 min/arm). Two parameter classes; a-priori leading shape per class also
# gets one lr probe. Pick per class by val CE at equal wall-clock PLUS a
# sanity read of logs/<arm>/samples.txt — a shape that word-salads at 45 min
# does not advance regardless of CE.
#
#   15M class (thinker parity):   A_384x5, A_320x8, A_256x12, A_384x5_lr6
#   48M class (system parity):    B_512x12, B_576x10, B_640x8, B_512x12_lr6
#
# Usage:  scripts/ablate_lm_b1.sh all      (or a list of arm names)
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-2700}

run() {
  local name=$1; shift
  echo "=== [$(date +%H:%M:%S)] B1 arm: $name ==="
  .venv/bin/tv-train-lm --config configs/b1_lm.yaml \
    run.name="$name" train.max_seconds="$DUR" "$@"
}

arm() {
  case $1 in
    A_384x5)     run b1_lm_A_384x5   lm.d_model=384 lm.layers=5  lm.ffn_dim=1536 lm.nhead=6 ;;
    A_320x8)     run b1_lm_A_320x8   lm.d_model=320 lm.layers=8  lm.ffn_dim=1280 lm.nhead=5 ;;
    A_256x12)    run b1_lm_A_256x12  lm.d_model=256 lm.layers=12 lm.ffn_dim=1024 lm.nhead=4 ;;
    A_384x5_lr6) run b1_lm_A_384x5_lr6 lm.d_model=384 lm.layers=5 lm.ffn_dim=1536 lm.nhead=6 train.lr=6.0e-4 ;;
    # B class runs batch 16: activations + the B*T*vocab CE logits OOM the
    # 12 GB card at batch 32. Within-class comparisons are unaffected.
    B_512x12)    run b1_lm_B_512x12  lm.d_model=512 lm.layers=12 lm.ffn_dim=2048 lm.nhead=8 train.batch_size=16 ;;
    B_576x10)    run b1_lm_B_576x10  lm.d_model=576 lm.layers=10 lm.ffn_dim=2304 lm.nhead=8 train.batch_size=16 ;;
    B_640x8)     run b1_lm_B_640x8   lm.d_model=640 lm.layers=8  lm.ffn_dim=2560 lm.nhead=8 train.batch_size=16 ;;
    B_512x12_lr6) run b1_lm_B_512x12_lr6 lm.d_model=512 lm.layers=12 lm.ffn_dim=2048 lm.nhead=8 train.batch_size=16 train.lr=6.0e-4 ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

ARMS=("$@")
if [ "${1:-all}" = "all" ]; then
  ARMS=(A_384x5 A_320x8 A_256x12 A_384x5_lr6 B_512x12 B_576x10 B_640x8 B_512x12_lr6)
fi
# gfx1031 flakes (page faults, non-finite grads) can kill an arm mid-run;
# retry each arm once from scratch and keep the bracket alive either way.
for a in "${ARMS[@]}"; do
  n="b1_lm_$a"
  for attempt in 1 2; do
    if arm "$a"; then break; fi
    echo "!!! [$(date +%H:%M:%S)] arm $a failed (attempt $attempt)"
    mv "logs/$n" "logs/${n}_crash${attempt}" 2>/dev/null || true
    mv "checkpoints/$n" "checkpoints/${n}_crash${attempt}" 2>/dev/null || true
    if [ "$attempt" = 2 ]; then echo "!!! arm $a skipped after 2 failures"; fi
    sleep 10
  done
done
echo "=== B1 bracket complete ==="
for a in "${ARMS[@]}"; do
  n="b1_lm_$a"
  best=$(grep -o '"val_ce": [0-9.]*' "logs/$n/metrics.jsonl" | sort -t' ' -k2 -n | head -1 || true)
  echo "$n  best $best"
done
