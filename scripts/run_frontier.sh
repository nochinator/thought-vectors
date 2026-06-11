#!/usr/bin/env bash
# Crash-resilient frontier run: relaunch with --resume after ROCm faults
# (gfx1031 sporadically page-faults on long runs; see RESEARCH_LOG 2026-06-11).
# The trainer's max_seconds cap is cumulative across resumes, so this loop
# exits naturally when the 12h budget is spent.
set -u
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0   # known mitigation for RDNA2 "Memory access fault"

RUN=m5_frontier
LOG=logs/$RUN/train.out
mkdir -p "logs/$RUN"
attempt=0

while true; do
  attempt=$((attempt + 1))
  latest=$(ls -t checkpoints/$RUN/step_*.pt 2>/dev/null | head -1)
  if [ -n "${latest}" ]; then
    args=(--resume "$latest")
  elif [ -f checkpoints/warm_base/final.pt ]; then
    args=(--init-from checkpoints/warm_base/final.pt)
  else
    args=()
  fi
  echo "=== attempt $attempt: tv-train ${args[*]:-fresh} ===" >> "$LOG"
  start=$(date +%s)
  .venv/bin/tv-train --config configs/$RUN.yaml "${args[@]}" >> "$LOG" 2>&1
  rc=$?
  ran=$(( $(date +%s) - start ))
  if grep -q "done:" "$LOG"; then
    echo "=== run complete after $attempt attempt(s) ===" >> "$LOG"
    break
  fi
  if [ "$ran" -lt 300 ]; then
    echo "=== crashed after ${ran}s (rc=$rc) — crash loop, giving up ===" >> "$LOG"
    exit 1
  fi
  echo "=== crashed after ${ran}s (rc=$rc) — resuming in 15s ===" >> "$LOG"
  sleep 15
done
