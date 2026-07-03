#!/usr/bin/env bash
# ── M4 thinker — ROUND 4: content anti-collapse bracket (rank → 12h stack) ──────
#
# Designed 2026-06-24 with nochi, after the round-3 core + out_tau sweep +
# output-rank diagnostic established:
#   • The codec's predictor says true ~30-tok replies need ~4 vectors (not 32);
#     k_out=8 gives headroom over that — folded into the BASE here.
#   • The thinker's OUTPUT RANK (~3.5) ≈ the true reply's (~4.1): the disease is
#     CONTENT mean-collapse ("good thing" attractor), NOT rank/tail collapse.
#     ⇒ attack content collapse directly; KEEP KRP training, DROP out_tau
#     truncation (it monotonically hurt ref_f1).
#   • LR anneals on WALL-CLOCK fraction: the 40-min arms never reached the
#     low-LR settling regime where WTA modes separate. These arms are 2h
#     (DUR=7200) so the anneal + mode-separation actually begin.
#
# BASE = WTA4 frontier recipe + k_out=8 + KRP, decode-FULL (no out_tau), on the
# rebalanced dialogue_combined corpus. One change per arm on top. Rank these,
# then stack the winner(s) into a single 12h run.
#
# INVARIANT (nochi, hard): text→encode→thoughts→decode→text must ALWAYS hold.
# The UNFREEZE arm adapts the decoder; the codec stays in eval() (dropout off,
# ROCm-safe + deterministic) and compress_frac=0.2 anchors prose reconstruction.
# After it trains, check_roundtrip.py PROVES the round-trip held — FAIL ⇒ the arm
# is DISQUALIFIED from the 12h stack regardless of its dialogue metrics.
#
# Usage:
#   scripts/ablate_thinker_r4.sh R4_CTL R4_CONTRAST ...   # pick arms
#   scripts/ablate_thinker_r4.sh all                       # full bracket
#   DUR=3600 scripts/ablate_thinker_r4.sh all              # 1h slots
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-7200}   # 2h slots (nochi, 2026-06-24)

BASE="thinker.n_hypotheses=4 thinker.w_decoder=0.5 thinker.k_ctx=8 thinker.k_out=8 thinker.k_out_random_prefix=true thinker.k_out_min=4"
DATA="data.shard_dir=data/dialogue_combined"

overrides_for() {
  case "$1" in
    R4_CTL)      echo "$BASE $DATA" ;;                                        # control at the new budget
    R4_CONTRAST) echo "$BASE $DATA thinker.contrast_weight=0.2" ;;            # ★ sharp anti-mean-collapse (NTXent)
    R4_DIV)      echo "$BASE $DATA thinker.diversity_weight=0.1" ;;           # soft anti-collapse (batch variance)
    R4_UNFREEZE) echo "$BASE $DATA thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05" ;;  # close TF→free gap; strong anchor holds round-trip (180s guard PASS @ k2 +0.049); round-trip-guarded

    R4_ROLEFLIP) echo "$BASE $DATA thinker.role_flip=true" ;;                 # data variety (user↔bot)
    R4_WTA8)     echo "$BASE $DATA thinker.n_hypotheses=8" ;;                 # more modes (gradient split wider)
    R4_COMBO)    echo "$BASE $DATA thinker.role_flip=true thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05" ;;  # ★ the STACK: prove role_flip × unfreeze clears BOTH singles before 12h (nochi: test the combos)
    R4_BIG30)    echo "$BASE $DATA thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05 thinker.layers=8 thinker.ffn_dim=4096" ;;  # ★ 30M (32.0M, 8L/4096 @ d384) unfreeze single — does 2x size beat 15M-unfreeze at equal 2h? (nochi: do a 30M 2h run)
    R4_BIG30_4H) echo "$BASE $DATA thinker.unfreeze=decoder thinker.compress_frac=0.4 thinker.codec_lr_scale=0.05 thinker.layers=8 thinker.ffn_dim=4096" ;;  # ★ 30M @ 4h (DUR=14400) ≈ 46K steps = MATCHED STEPS + full anneal vs 15M-2h: isolates capacity at equal maturity (nochi 2026-06-25). Separate ckpt so 2h R4_BIG30 is preserved.
    *) echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

ALL=(R4_CTL R4_CONTRAST R4_DIV R4_UNFREEZE R4_ROLEFLIP R4_WTA8)

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
    # round-trip integrity guard — fires for ANY arm that modifies the codec (unfreeze)
    if [[ "$ov" == *"unfreeze"* ]]; then
      echo "--- round-trip guard $name ---" | tee -a "logs/$name.out"
      .venv/bin/python scripts/check_roundtrip.py --ckpt "checkpoints/$name/best.pt" \
        2>&1 | tee -a "logs/$name.out" \
        || echo "ROUND-TRIP FAIL $name — DISQUALIFIED from 12h stack" | tee -a "logs/$name.out"
    fi
  else
    echo "no best.pt for $name — eval skipped" | tee -a "logs/$name.out"
  fi
done
echo "=== round-4 bracket done: ${args[*]} ==="
