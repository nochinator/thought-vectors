#!/usr/bin/env bash
# ── M4 thinker — FINAL 12h run (2026-07-02) ───────────────────────────────────
#
# Recipe: R4_UNFREEZE base (WTA4 · k_out=8 · KRP · decode-full ·
# dialogue_combined · unfreeze=decoder, strong anchor) + the two R5 levers that
# survived the transcript-audited bracket AND the chat-probe hard gate:
#   cycle loss   (cycle_frac=0.25, w_cycle=0.5)  — R5's one clean win: best
#                self_rep, best val_cos, visibly more grammatical chat
#   w_decoder=1.0 — admitted only via R5_STACK confirmation (WDEC alone gamed
#                ctx_sens with noise; stacked with cycle the noise was bounded,
#                the wedding hard-collapse broke 0.909→0.479, and a novel
#                follow-up-question behavior appeared)
#
# From scratch at default LR (3e-4): the levers shape the whole trajectory
# rather than fine-tuning a 2h-mature basin (warm-start was a 45-min-budget
# workaround, not a preference).  Cycle loss costs ~40% throughput — accepted;
# R5 showed the lever beats raw step count.
#
# Gate record: chat probe (4 fresh convs, temp 0 + 0.8, 3-way vs R4_UNFREEZE
# and R5_CYCLE) passed 2026-07-02 — see RESEARCH_LOG.
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-43200}
NAME=${NAME:-FINAL_12H}

.venv/bin/tv-train-thinker --config configs/m4_thinker.yaml \
  run.name="$NAME" train.max_seconds="$DUR" \
  thinker.n_hypotheses=4 thinker.k_ctx=8 thinker.k_out=8 \
  thinker.k_out_random_prefix=true thinker.k_out_min=4 \
  thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05 \
  thinker.cycle_frac=0.25 thinker.w_cycle=0.5 thinker.w_decoder=1.0 \
  data.shard_dir=data/dialogue_combined \
  2>&1 | tee "logs/$NAME.out"

.venv/bin/python scripts/eval_thinker.py --ckpt "checkpoints/$NAME/best.pt" \
  --dump "logs/$NAME.samples.txt" 2>&1 | tee -a "logs/$NAME.out"
echo "--- multiturn eval $NAME ---" | tee -a "logs/$NAME.out"
.venv/bin/python scripts/eval_multiturn.py --ckpt "checkpoints/$NAME/best.pt" \
  --device cpu --dump "logs/$NAME.multiturn.txt" 2>&1 | tee -a "logs/$NAME.out"
echo "--- round-trip guard $NAME ---" | tee -a "logs/$NAME.out"
.venv/bin/python scripts/check_roundtrip.py --ckpt "checkpoints/$NAME/best.pt" \
  2>&1 | tee -a "logs/$NAME.out" \
  || echo "ROUND-TRIP FAIL $NAME" | tee -a "logs/$NAME.out"
echo "=== FINAL run $NAME done ==="
