#!/usr/bin/env bash
# M4 thinker ablation bracket: equal-wall-clock runs over loss modes and
# architectures, all on the same dialogue shards + frozen m5_frontier codec.
# Usage: scripts/ablate_thinker.sh T0 T1 T2 ... (DUR seconds each, default 2400)
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
DUR=${DUR:-2400}

overrides_for() {
  case "$1" in
    # --- loss-mode bracket (user ideas a, c-mix, d, e) ---
    T0)  echo "" ;;                                            # thought-space MSE+cos only
    T1)  echo "thinker.w_thought=0.0 thinker.w_decoder=1.0" ;; # decoder CE only
    T2)  echo "thinker.w_decoder=0.5" ;;                       # mixed thought + CE
    T3)  echo "thinker.w_reverse=0.5" ;;                       # + reverse aux (annealed)
    T4)  echo "thinker.w_decoder=0.5 thinker.w_reverse=0.5" ;; # mixed + reverse
    T5)  echo "thinker.cycle_frac=0.25 thinker.w_cycle=0.5" ;; # + re-encode consistency
    # --- architecture bracket ---
    P0)  echo "thinker.mode=prefix" ;;                         # AR over thought slots
    P1)  echo "thinker.mode=prefix thinker.w_decoder=0.5" ;;
    K16) echo "thinker.k_ctx=16 thinker.k_out=16" ;;           # tighter thought budget
    K48) echo "thinker.k_ctx=48 thinker.k_out=48" ;;
    L4)  echo "thinker.layers=4" ;;
    L8)  echo "thinker.layers=8" ;;
    # --- context-budget bracket (mean turn ~23 tok; k_ctx=32 costs MORE
    #     vectors than raw text — how few context thoughts suffice?) ---
    K8)    echo "thinker.k_ctx=8" ;;                           # 3:1 ctx compression
    SCHED) echo "thinker.k_ctx_schedule=[32,16,8]" ;;          # budget decays with age
    TAU)   echo "thinker.ctx_tau=0.5" ;;                       # predictor-adaptive per turn
    FLAT)  echo "thinker.flat_context=true" ;;                 # whole history, one encode
    # --- mode-collapse fixes (round 2: deterministic heads averaged the many
    #     valid replies into bland filler — distinct-1 <= 0.04 across T0-T5) ---
    WTA4) echo "thinker.n_hypotheses=4 thinker.w_decoder=0.5 thinker.k_ctx=8" ;;
    WTA8) echo "thinker.n_hypotheses=8 thinker.w_decoder=0.5 thinker.k_ctx=8" ;;
    C8)   echo "thinker.w_decoder=0.5 thinker.k_ctx=8" ;;     # round-2 control: T2 @ k8
    D8)   echo "thinker.w_thought=0.0 thinker.w_decoder=1.0 thinker.k_ctx=8" ;;  # T1 @ k8
    PN1)  echo "thinker.mode=prefix thinker.tf_noise_std=0.1 thinker.w_decoder=0.5 thinker.k_ctx=8" ;;
    PN3)  echo "thinker.mode=prefix thinker.tf_noise_std=0.3 thinker.w_decoder=0.5 thinker.k_ctx=8" ;;
    FD)   echo "thinker.flat_context=true thinker.w_decoder=0.5" ;;  # fastest arch + mixed loss
    # --- phase 2 previews (end-to-end, user idea b) ---
    U1)  echo "thinker.w_decoder=1.0 thinker.w_thought=0.0 thinker.unfreeze=decoder thinker.compress_frac=0.15" ;;
    U2)  echo "thinker.w_decoder=1.0 thinker.w_thought=0.25 thinker.unfreeze=codec thinker.compress_frac=0.25" ;;
    *)   echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

for name in "$@"; do
  ov=$(overrides_for "$name")
  echo "=== thinker ablation $name (${DUR}s): $ov ==="
  # shellcheck disable=SC2086
  .venv/bin/tv-train-thinker --config configs/m4_thinker.yaml \
    run.name="ab_$name" train.max_seconds="$DUR" $ov \
    2>&1 | tee "logs/ab_$name.out" \
    || echo "RUN $name FAILED — continuing bracket" | tee -a "logs/ab_$name.out"
  if [[ -f "checkpoints/ab_$name/best.pt" ]]; then
    .venv/bin/python scripts/eval_thinker.py --ckpt "checkpoints/ab_$name/best.pt" \
      --dump "logs/ab_$name.samples.txt" 2>&1 | tee -a "logs/ab_$name.out" \
      || echo "EVAL $name FAILED" | tee -a "logs/ab_$name.out"
  else
    echo "no best.pt for $name — eval skipped" | tee -a "logs/ab_$name.out"
  fi
done
