#!/usr/bin/env bash
# ── M4 thinker — ROUND 6: post-FINAL_12H residual-disease bracket (2026-07-03) ──
#
# FINAL_12H broke content collapse (self_rep 0.19, ctx_sens 0.15, chat gate PASS).
# Residual diseases (RESEARCH_LOG 2026-07-03):
#   (a) context-conditional positivity register errors — reg_err_ctx 0.50 on
#       scripts/eval_register.py (cheerful replies to bad news after a good turn)
#   (b) "i'm a big fan of the ..." stock-phrase attractor (also drags pos_ok to 0.17)
#   (c) pronoun/person confusion
#
# R5 methodology: every arm warm-starts thinker+adapted codec from
# checkpoints/FINAL_12H/best.pt, 45 min at lr 1e-4.  Pre-registered hypotheses:
#   HYP8     WTA winner = most-decodable hypothesis, which skews generic-positive;
#            8 modes may carve out a commiseration mode (targets reg_err_ctx).
#            out_seed [8,k,d] re-inits (shape skip) — everything else warm.
#   CYCLE50 / WCYC1  more of R5's star lever — predictions surviving re-encode
#            should suppress stock phrases like "big fan" (targets pos_ok + b)
#   UNFCODEC unfreeze full codec: encoder can sharpen sentiment separation in
#            thought space (targets reg_err_ctx; round-trip guard is the veto)
#   CTL      maturity control (R5: plain fine-tune made things WORSE — recheck at
#            12h maturity)
#
# Decision rule: promote a lever iff it beats CTL on its target metric, does not
# regress self_rep/ctx_sens/ref_f1/pos_ok/round-trip, AND survives transcript
# audit + chat probe (R5 lesson: ctx_sens was gamed by noise once already).
#
# Usage:  scripts/ablate_thinker_r6.sh all      (or a list of arm names)
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
DATA="data.shard_dir=data/dialogue_combined"

overrides_for() {
  case "$1" in
    R6_CTL)      echo "$BASE $DATA" ;;
    R6_CYCLE50)  echo "$BASE $DATA thinker.cycle_frac=0.5" ;;
    R6_WCYC1)    echo "$BASE $DATA thinker.w_cycle=1.0" ;;
    R6_HYP8)     echo "$BASE $DATA thinker.n_hypotheses=8" ;;
    R6_UNFCODEC) echo "$BASE $DATA thinker.unfreeze=codec" ;;
    *) echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

ALL=(R6_CTL R6_CYCLE50 R6_WCYC1 R6_HYP8 R6_UNFCODEC)

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
echo "=== round-6 bracket done: ${args[*]} ==="
