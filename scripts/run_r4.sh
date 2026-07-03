#!/usr/bin/env bash
# Round 4 overnight bracket: 8 arms × 1h each = ~8h total.
# All M-mixed config (32M, 8L/8H/FFN4096, k_ctx=16, k_out=48, w_thought=1.0, w_decoder=0.5).
# Sequential execution — GPU can only run one training process at a time.

set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0

ARMS=(
  r4_c_control
  r4_contrast
  r4_posweight
  r4_aug
  r4_perslot
  r4_diverse
  r4_tau
  r4_all
)

START_TIME=$(date +%s)

for arm in "${ARMS[@]}"; do
    echo "=== $(date): launching $arm ==="
    mkdir -p "logs/${arm}"
    .venv/bin/tv-train-thinker --config "configs/${arm}.yaml" 2>&1 \
        | tee "logs/${arm}/train.out"
    echo "=== $(date): $arm finished ==="
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "=== ALL DONE in $(( ELAPSED / 60 )) minutes ==="

# Quick summary of final metrics
for arm in "${ARMS[@]}"; do
    echo -n "$arm: "
    tail -1 "logs/${arm}/metrics.jsonl" 2>/dev/null | python3 -c \
        "import json,sys; d=json.load(sys.stdin); print(f'done at step {d.get(\"step\",\"?\")}')"
done
