#!/usr/bin/env bash
# ── M4 thinker — ROUND 3: "data + slot" and the tail-starvation levers ──────────
#
# Context (RESEARCH_LOG ends 2026-06-13 at the WTA4 frontier; this round was
# designed 2026-06-24 with Nyx). The WTA4 frontier (225K steps) nails the
# grounded opening clause then decays into filler ("a good thing") across the
# BACK HALF of its response slots — distinct-1 0.022. Diagnosis: one-to-many
# mean-collapse, concentrated in the tail slots.
#
# Two structural reasons the tail starves, both attacked here:
#   1. DATA. The frontier trained on data/dialogue (96% SODA, short/formulaic
#      replies — the literal filler attractor). data/dialogue_combined (built
#      2026-06-15, never ablated) cuts SODA 6x, boosts OASST 10x, mean turn
#      23->30 tok. R3_CTL_OLD vs R3_DATA isolates this.
#   2. TOO MANY RESPONSE SLOTS. k_out=32, but a ~30-tok reply is near-perfectly
#      reconstructed by the codec in ~8 thoughts (r=0.25). Slots ~8..31 of the
#      ENCODED target are low-information refinements; regressing to them yields
#      filler, and at inference the decoder renders all 32 -> garbage tail.
#      Levers: shrink k_out (KOUT*), down-weight the noisy tail thought-MSE
#      (SLOT — now reachable in the WTA path), or force importance-ordering so
#      any short prefix decodes (KRP, mirrors the codec's own training).
#
# Protocol (unchanged from round 1/2): equal WALL-CLOCK, one change per arm vs
# the R3_DATA base, log everything. Base recipe = the WTA4 frontier recipe.
#
# Usage:
#   scripts/ablate_thinker_r3.sh R3_DATA R3_KOUT16 R3_SLOT ...   # pick arms
#   scripts/ablate_thinker_r3.sh all                              # full sequence
#   DUR=1800 scripts/ablate_thinker_r3.sh all                     # 30-min slots
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-2400}   # 40-min slots by default (round-1/2 protocol)

# Frontier WTA4 recipe, shared by every arm. Data is combined for all but the
# OLD control. One override per arm is layered on top.
BASE="thinker.n_hypotheses=4 thinker.w_decoder=0.5 thinker.k_ctx=8"
DATA="data.shard_dir=data/dialogue_combined"

overrides_for() {
  case "$1" in
    # ── data isolation ──────────────────────────────────────────────────────
    R3_CTL_OLD) echo "$BASE data.shard_dir=data/dialogue" ;;   # frontier recipe, OLD data — reference
    R3_DATA)    echo "$BASE $DATA" ;;                          # ★ round-3 base: rebalanced data only

    # ── response-slot rightsizing (suspected #1 lever) ──────────────────────
    R3_KOUT16)  echo "$BASE $DATA thinker.k_out=16" ;;         # halve response slots
    R3_KOUT12)  echo "$BASE $DATA thinker.k_out=12" ;;         # ~reply info content
    R3_KOUT8)   echo "$BASE $DATA thinker.k_out=8" ;;          # match k_ctx; aggressive

    # ── the "slot" half: stop the noisy tail from dominating the loss ───────
    R3_SLOT)    echo "$BASE $DATA thinker.slot_weight_decay=0.3" ;;   # early slots weighted 1.0 -> tail 0.3 (now WTA-reachable)
    # importance-ordered output + ADAPTIVE response length: a ~20-tok reply
    # decodes from ~4-8 vectors (predictor-chosen), not all 32 (nochi, 2026-06-24)
    R3_KRP)     echo "$BASE $DATA thinker.k_out_random_prefix=true thinker.k_out_min=4 thinker.out_tau=0.5" ;;
    R3_KRPFULL) echo "$BASE $DATA thinker.k_out_random_prefix=true thinker.k_out_min=4" ;;  # same training, decode all 32 (A/B for out_tau)

    # ── other one-to-many fixes to compare against ──────────────────────────
    R3_DIV)     echo "$BASE $DATA thinker.diversity_weight=0.1" ;;    # explicit anti-collapse (batch-variance)
    R3_WTA8)    echo "$BASE $DATA thinker.n_hypotheses=8" ;;          # more modes (frontier note: real but gradient-split)
    R3_POSW)    echo "$BASE $DATA thinker.pos_weight_decay=0.5" ;;    # position-weighted decoder CE (token-side analog)

    # ── stacked winners (run AFTER the isolated arms rank; edit to taste) ────
    R3_BOTH)    echo "$BASE $DATA thinker.slot_weight_decay=0.3 thinker.k_out_random_prefix=true thinker.k_out_min=4 thinker.out_tau=0.5" ;;

    *) echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

ALL=(R3_CTL_OLD R3_DATA R3_KOUT16 R3_KOUT12 R3_KOUT8 R3_SLOT R3_KRP R3_DIV R3_WTA8 R3_POSW)

args=("$@")
if [[ "${1:-}" == "all" ]]; then args=("${ALL[@]}"); fi

for name in "${args[@]}"; do
  ov=$(overrides_for "$name")
  echo "=== thinker ablation $name (${DUR}s): $ov ==="
  # shellcheck disable=SC2086
  .venv/bin/tv-train-thinker --config configs/m4_thinker.yaml \
    run.name="$name" train.max_seconds="$DUR" $ov \
    2>&1 | tee "logs/$name.out" \
    || echo "RUN $name FAILED — continuing sequence" | tee -a "logs/$name.out"
  if [[ -f "checkpoints/$name/best.pt" ]]; then
    .venv/bin/python scripts/eval_thinker.py --ckpt "checkpoints/$name/best.pt" \
      --dump "logs/$name.samples.txt" 2>&1 | tee -a "logs/$name.out" \
      || echo "EVAL $name FAILED" | tee -a "logs/$name.out"
  else
    echo "no best.pt for $name — eval skipped" | tee -a "logs/$name.out"
  fi
done
echo "=== round-3 sequence done: ${args[*]} ==="
