#!/usr/bin/env bash
# ── M4 thinker — FINAL2 12h run: FINAL recipe, reversal data mix (2026-07-03) ──
#
# Identical recipe to scripts/final_12h.sh (data is the ONLY variable) on
# data/dialogue_rev40 = dialogue_combined + empathetic_dialogues +
# 40k register-REVERSAL splices (scripts/build_reversal_splices.py).
#
# Rationale (RESEARCH_LOG R7/R8): register misrouting is a data absence —
# every training conversation held one mood throughout, so the trunk learned
# conversation-level mood conditioning. Warm-start fine-tunes (R6/R7/R8)
# transferred style but not routing; R8_REV40 produced the first contextual
# commiserations ever seen, so the mix goes in FROM SCRATCH where conditioning
# is learned during the whole trajectory.
#
# Gate record: R8_REV40 chat probe 2026-07-03 (3 fresh convs, temp 0) —
# conversational quality intact on the new data; see RESEARCH_LOG.
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-43200}
NAME=${NAME:-FINAL2_12H}

.venv/bin/tv-train-thinker --config configs/m4_thinker.yaml \
  run.name="$NAME" train.max_seconds="$DUR" \
  thinker.n_hypotheses=4 thinker.k_ctx=8 thinker.k_out=8 \
  thinker.k_out_random_prefix=true thinker.k_out_min=4 \
  thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05 \
  thinker.cycle_frac=0.25 thinker.w_cycle=0.5 thinker.w_decoder=1.0 \
  data.shard_dir=data/dialogue_rev40 \
  2>&1 | tee "logs/$NAME.out"

.venv/bin/python scripts/eval_thinker.py --ckpt "checkpoints/$NAME/best.pt" \
  --dump "logs/$NAME.samples.txt" 2>&1 | tee -a "logs/$NAME.out"
echo "--- multiturn eval $NAME ---" | tee -a "logs/$NAME.out"
.venv/bin/python scripts/eval_multiturn.py --ckpt "checkpoints/$NAME/best.pt" \
  --device cpu --dump "logs/$NAME.multiturn.txt" 2>&1 | tee -a "logs/$NAME.out"
echo "--- register eval $NAME ---" | tee -a "logs/$NAME.out"
.venv/bin/python scripts/eval_register.py --ckpt "checkpoints/$NAME/best.pt" \
  --device cpu --dump "logs/$NAME.register.txt" 2>&1 | tee -a "logs/$NAME.out"
echo "--- register eval (affinity select) $NAME ---" | tee -a "logs/$NAME.out"
.venv/bin/python scripts/eval_register.py --ckpt "checkpoints/$NAME/best.pt" \
  --device cpu --hyp-select affinity --dump "logs/$NAME.register_affinity.txt" \
  2>&1 | tee -a "logs/$NAME.out"
echo "--- round-trip guard $NAME ---" | tee -a "logs/$NAME.out"
.venv/bin/python scripts/check_roundtrip.py --ckpt "checkpoints/$NAME/best.pt" \
  2>&1 | tee -a "logs/$NAME.out" \
  || echo "ROUND-TRIP FAIL $NAME" | tee -a "logs/$NAME.out"
echo "=== FINAL run $NAME done ==="
