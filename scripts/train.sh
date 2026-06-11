#!/usr/bin/env bash
# All training runs go through this wrapper: ROCm guards for gfx1031.
set -euo pipefail
cd "$(dirname "$0")/.."

export HSA_OVERRIDE_GFX_VERSION=10.3.0
# Do NOT set torch float32_matmul_precision("high") anywhere — NaN after ~20K
# steps on this stack (see RESEARCH_LOG.md). The trainer asserts "highest".
# A hipBLASLt->hipblas fallback warning at startup is expected and harmless.

exec .venv/bin/tv-train "$@"
