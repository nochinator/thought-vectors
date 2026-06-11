#!/usr/bin/env bash
# Equal-wall-clock ablation bracket (see RESEARCH_LOG.md "Ablation bracket").
# Usage: scripts/ablate.sh A C D B384 B512 [E256 ...]
# DUR (seconds per run, default 1500) applies to every run for fairness.
set -euo pipefail
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
DUR=${DUR:-1500}

# W-series: compression-objective ablations, warm-started from the shared
# 60-min base (checkpoints/warm_base/final.pt) for sharper contrasts.
WARM="checkpoints/warm_base/final.pt"

overrides_for() {
  case "$1" in
    W0)   echo "" ;;                                  # control: base config continues
    W1)   echo "reg.word_dropout=0.15" ;;
    W2)   echo "reg.word_dropout=0.30" ;;
    W3)   echo "ksampler.mode=per_sample" ;;
    W4)   echo "ksampler.mode=per_sample reg.word_dropout=0.15" ;;
    W5)   echo "ksampler.full_frac=0.05 ksampler.uniform_frac=0.25 ksampler.ratio_bands=[[0.1,0.25,0.35],[0.25,0.4,0.35],[0.4,0.6,0.2],[0.6,1.0,0.1]]" ;;
    A)    echo "" ;;                                  # d256 4+4 parity anchor point
    C)    echo "train.anchor_full_k_weight=0.5" ;;    # full-k anchor decode
    C4th) echo "train.anchor_full_k_weight=0.5 train.anchor_every=4" ;;
    D)    echo "reg.nar_frac=0.25" ;;                 # NAR-mixed batches
    B384) echo "model.d_model=384 model.nhead=6 model.enc_layers=5 model.dec_layers=5 train.batch_size=32" ;;
    B512) echo "model.d_model=512 model.nhead=8 model.enc_layers=6 model.dec_layers=6 train.batch_size=32" ;;
    # E256 is the frontier dress rehearsal: chosen shape + regularizers on paragraph data.
    E256) echo "data.shard_dir=data/mix_long model.d_model=384 model.nhead=6 model.enc_layers=5 model.dec_layers=5 model.max_seq_len=256 model.num_thoughts=256 train.batch_size=32 reg.kl_beta=0.01 reg.noise_std=0.05 train.predictor_extra_k=1" ;;
    *)    echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

for name in "$@"; do
  ov=$(overrides_for "$name")
  echo "=== ablation $name (${DUR}s): $ov ==="
  extra=()
  if [[ "$name" == W* ]]; then
    extra=(--init-from "$WARM" --config configs/m5_frontier.yaml)
  else
    extra=(--config configs/ablate_base.yaml)
  fi
  # shellcheck disable=SC2086
  .venv/bin/tv-train "${extra[@]}" \
    run.name="ab_${name}" train.max_seconds="$DUR" $ov
  if [[ "$name" == W* ]]; then
    .venv/bin/tv-eval --ckpt "checkpoints/ab_${name}/final.pt" --shard data/mix_uni_val \
      --max-texts 1200 --decode-per-bucket 60 --out "logs/ab_${name}/eval" >/dev/null \
      && echo "eval ab_${name} done"
  fi
done
echo "ABLATIONS_DONE: $*"
