"""gfx1031 reports an unsupported arch to ROCm kernels; the override must be
set before torch initializes HIP or GPU tests fail with 'invalid device function'."""

import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
