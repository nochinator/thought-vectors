#!/usr/bin/env bash
# One-time environment setup.
#   scripts/setup_env.sh          # ROCm build (RX 6700 XT / gfx1031) — what all training ran on
#   scripts/setup_env.sh --cpu    # CPU-only torch — enough to chat with / eval a released checkpoint
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND=rocm
[ "${1:-}" = "--cpu" ] && BACKEND=cpu

if [ ! -d .venv ]; then
    python3 -m venv --without-pip .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
fi
if [ "$BACKEND" = cpu ]; then
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
fi
.venv/bin/pip install sentencepiece numpy pyyaml pytest
.venv/bin/pip install -e .

if [ "$BACKEND" = cpu ]; then
    echo "--- CPU verification ---"
    .venv/bin/python -c 'import torch; x = torch.randn(64, 64); print("torch", torch.__version__, "| matmul ok:", (x @ x).sum().item())'
    echo "CPU-only setup done. Chat/eval work; training scripts assume the ROCm GPU."
else
    echo "--- GPU verification ---"
    HSA_OVERRIDE_GFX_VERSION=10.3.0 .venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip)
print("cuda.is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda")
print("matmul ok, sum:", (x @ x).sum().item())
EOF
fi
