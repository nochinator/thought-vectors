#!/usr/bin/env bash
set -eu
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONUNBUFFERED=1
cd /home/nochi/vault/projects/AI_construction/legacy/thought-vectors-main
exec ../.venv/bin/python3 -u "$@"
