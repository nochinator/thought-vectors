#!/usr/bin/env bash
# ── M4 thinker — ROUND 5: warm-started fine-tune bracket (2026-07-02) ──────────
#
# Context: R4 concluded 12h-final = R4_UNFREEZE (BASE · WTA4 · k_out=8 · KRP ·
# decode-full · dialogue_combined · unfreeze=decoder, strong anchor).  Residual
# disease: content mean-collapse ("great day / great person" attractor) and
# multi-turn context-ignoring (the identical-reply flag from the qualitative
# probe).  R4 also proved cold 40-min arms are too undertrained to rank
# anti-collapse levers — so R5 arms WARM-START thinker + adapted decoder from
# checkpoints/R4_UNFREEZE/best.pt (2h mature) and fine-tune 45 min at reduced
# LR.  Every arm therefore trains in the low-LR settling regime from minute 0.
#
# All arms keep unfreeze=decoder (so the adapted decoder stays in the ckpt and
# eval loads the right codec) with the round-trip-proven strong anchor.
#
# New metric: scripts/eval_multiturn.py (self_rep / ctx_sens) — the R4 probe
# showed single-turn distinct1 misses multi-turn collapse.  Baseline to beat
# (R4_UNFREEZE/best.pt, CPU, temp 0): self_rep 0.3229, ctx_sens 0.2759.
#
# Usage:
#   scripts/ablate_thinker_r5.sh all
#   DUR=3600 scripts/ablate_thinker_r5.sh R5_CTL R5_CYCLE
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-2700}   # 45-min slots (nochi 2026-07-02: consistent 30-60 min per arm)

BASE="thinker.n_hypotheses=4 thinker.w_decoder=0.5 thinker.k_ctx=8 thinker.k_out=8 \
thinker.k_out_random_prefix=true thinker.k_out_min=4 \
thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05 \
thinker.thinker_init_from=checkpoints/R4_UNFREEZE/best.pt \
train.lr=1.0e-4 train.warmup_steps=200"
DATA="data.shard_dir=data/dialogue_combined"

overrides_for() {
  case "$1" in
    R5_CTL)      echo "$BASE $DATA" ;;                                        # pure maturity: does 45 more min alone move anything?
    R5_TURNDROP) echo "$BASE $DATA thinker.turn_dropout=0.25" ;;              # ★ never ablated — force context use / robustness (targets ctx_sens)
    R5_CYCLE)    echo "$BASE $DATA thinker.cycle_frac=0.25 thinker.w_cycle=0.5" ;;  # ★ re-encode consistency: predicted thoughts must survive decode→re-encode (targets word-salad)
    R5_WDEC)     echo "$BASE $DATA thinker.w_decoder=1.0" ;;                  # stronger token-space grounding
    R5_KCTX16)   echo "$BASE $DATA thinker.k_ctx=16" ;;                       # context starvation hypothesis: 8 vecs/turn may crush the history
    R5_STACK)    echo "$BASE $DATA thinker.cycle_frac=0.25 thinker.w_cycle=0.5 thinker.w_decoder=1.0" ;;  # confirmation arm: CYCLE (self_rep winner) + WDEC (ctx_sens winner) — do they compose or does WDEC's collapse dominate?
    *) echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

ALL=(R5_CTL R5_TURNDROP R5_CYCLE R5_WDEC R5_KCTX16)

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
    echo "--- round-trip guard $name ---" | tee -a "logs/$name.out"
    .venv/bin/python scripts/check_roundtrip.py --ckpt "checkpoints/$name/best.pt" \
      2>&1 | tee -a "logs/$name.out" \
      || echo "ROUND-TRIP FAIL $name — DISQUALIFIED" | tee -a "logs/$name.out"
  else
    echo "no best.pt for $name — eval skipped" | tee -a "logs/$name.out"
  fi
done
echo "=== round-5 bracket done: ${args[*]} ==="
