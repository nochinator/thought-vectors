#!/usr/bin/env bash
# ── M4 thinker — ROUND 7: register disease is a DATA problem (2026-07-03) ──
#
# R6 verdict (RESEARCH_LOG): every arm INCLUDING the do-nothing control worsened
# reg_err_ctx (0.50 -> 0.67..0.83) — continued fine-tuning on dialogue_combined's
# relentlessly upbeat smalltalk IS the disease; no loss/architecture lever fixed
# it.  R7 attacks the data: facebook/empathetic_dialogues (23,074 convs of
# situation-sharing -> commiseration, heavy on negative emotions) mixed into the
# shards (scripts/extract_empathetic.py):
#   data/dialogue_emp1x  combined + emp        (~18% of turns empathetic)
#   data/dialogue_emp2x  combined + emp x2     (~31%; NB dup convs may straddle
#                        train/val — val_cos slightly optimistic, decisions use
#                        external probes only)
#
# Arms (R5/R6 methodology: warm-start thinker+adapted codec from FINAL_12H,
# 45 min, lr 1e-4).  cycle_frac=0.5 is R6's protective lever for continued
# training (kept multiturn intact; lexicon-independent metrics).
#   R7_EMP1X     emp1x + cycle_frac=0.5   (main hypothesis)
#   R7_EMP2X     emp2x + cycle_frac=0.5   (dosage)
#   R7_EMP1X_NC  emp1x, base cycle_frac=0.25 (does data alone fix register?)
# Controls already on disk: FINAL_12H (base), R6_CTL (old data plain),
# R6_CYCLE50 (old data + protection) — all scored with the WIDENED lexicon.
#
# Decision rule: reg_err_ctx < 0.50 (FINAL_12H) with pos_ok not collapsed, no
# regression on self_rep/ctx_sens/ref_f1/round-trip, AND transcript audit +
# chat probe (three metric-gaming incidents so far — audit before believing).
#
# Usage:  scripts/ablate_thinker_r7.sh all      (or a list of arm names)
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-2700}

BASE="thinker.n_hypotheses=4 thinker.w_decoder=1.0 thinker.k_ctx=8 thinker.k_out=8 \
thinker.k_out_random_prefix=true thinker.k_out_min=4 \
thinker.cycle_frac=0.25 thinker.w_cycle=0.5 \
thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05 \
thinker.thinker_init_from=checkpoints/FINAL_12H/best.pt \
train.lr=1.0e-4 train.warmup_steps=200"

overrides_for() {
  case "$1" in
    R7_EMP1X)    echo "$BASE data.shard_dir=data/dialogue_emp1x thinker.cycle_frac=0.5" ;;
    R7_EMP2X)    echo "$BASE data.shard_dir=data/dialogue_emp2x thinker.cycle_frac=0.5" ;;
    R7_EMP1X_NC) echo "$BASE data.shard_dir=data/dialogue_emp1x" ;;
    R8_REV20)    echo "$BASE data.shard_dir=data/dialogue_rev20 thinker.cycle_frac=0.5" ;;
    R8_REV40)    echo "$BASE data.shard_dir=data/dialogue_rev40 thinker.cycle_frac=0.5" ;;
    *) echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

ALL=(R7_EMP1X R7_EMP2X R7_EMP1X_NC)
# R8 reversal-splice arms (see build_reversal_splices.py) run via: scripts/ablate_thinker_r7.sh R8_REV20 R8_REV40

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
    echo "--- multiturn eval $name ---" | tee -a "logs/$name.out"
    .venv/bin/python scripts/eval_multiturn.py --ckpt "checkpoints/$name/best.pt" \
      --device cpu --dump "logs/$name.multiturn.txt" 2>&1 | tee -a "logs/$name.out" \
      || echo "MULTITURN EVAL $name FAILED" | tee -a "logs/$name.out"
    echo "--- register eval $name ---" | tee -a "logs/$name.out"
    .venv/bin/python scripts/eval_register.py --ckpt "checkpoints/$name/best.pt" \
      --device cpu --dump "logs/$name.register.txt" 2>&1 | tee -a "logs/$name.out" \
      || echo "REGISTER EVAL $name FAILED" | tee -a "logs/$name.out"
    echo "--- round-trip guard $name ---" | tee -a "logs/$name.out"
    .venv/bin/python scripts/check_roundtrip.py --ckpt "checkpoints/$name/best.pt" \
      2>&1 | tee -a "logs/$name.out" \
      || echo "ROUND-TRIP FAIL $name — DISQUALIFIED" | tee -a "logs/$name.out"
  else
    echo "no best.pt for $name — eval skipped" | tee -a "logs/$name.out"
  fi
done
echo "=== round-7 bracket done: ${args[*]} ==="
