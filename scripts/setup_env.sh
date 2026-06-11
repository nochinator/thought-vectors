#!/usr/bin/env bash
# One-time environment setup for thoughtvec on ROCm (RX 6700 XT / gfx1031).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    python3 -m venv --without-pip .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
fi
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
.venv/bin/pip install sentencepiece numpy pyyaml pytest
.venv/bin/pip install -e .

echo "--- GPU verification ---"
HSA_OVERRIDE_GFX_VERSION=10.3.0 .venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip)
print("cuda.is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda")
print("matmul ok, sum:", (x @ x).sum().item())
EOF
