#!/usr/bin/env bash
# Round 4 — resume: contrast through all (control already done)
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0

ARMS=(r4_contrast r4_posweight r4_aug r4_perslot r4_diverse r4_tau r4_all)

for arm in "${ARMS[@]}"; do
    echo "=== $(date): launching $arm ==="
    mkdir -p "logs/${arm}"
    .venv/bin/tv-train-thinker --config "configs/${arm}.yaml" 2>&1 \
        | tee "logs/${arm}/train.out"
    echo "=== $(date): $arm finished ==="
done

echo "=== ALL DONE ==="
for arm in r4_c_control "${ARMS[@]}"; do
    echo -n "$arm: "
    tail -1 "logs/${arm}/metrics.jsonl" 2>/dev/null | python3 -c \
        "import json,sys; d=json.load(sys.stdin); s=d.get('step','?'); c=d.get('val_cos','?'); ce=d.get('val_dec_ce','?'); print(f'step {s}, val cos {c}, val dec CE {ce}')" 2>/dev/null || echo "FAILED"
done
