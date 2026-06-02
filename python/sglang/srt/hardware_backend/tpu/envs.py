"""TPU-specific environment variable registry.

This registry is primarily for customizing TPU-specific behavior during
feature development. Once an environment variable stabilizes, we may consider
promoting it to `sglang.srt.environ.Envs` and removing it from here.

Sample usage:
    from sglang.srt.hardware_backend.tpu import envs
    backend = envs.TPU_ATTN_BACKEND
"""

from __future__ import annotations

import os


def _str(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _bool(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# Propagates the resolved `--attention-backend` CLI argument to spawned TP
# workers (which do not inherit the parent's module-level state). If no CLI
# argument is provided and the environment variable is unset, it falls back
# to the default value "pallas_tpu_rpa_v3". Can also be set manually by the
# user to override the backend.
# Possible values:
#   - "pallas_tpu_rpa_v3" (default ragged paged attention kernel)
#   - "pallas_tpu_flash" (paged flash attention kernel)
#   - "torch_native" (fallback native PyTorch attention)
TPU_ATTN_BACKEND: str | None = _str("TPU_ATTN_BACKEND", default="pallas_tpu_rpa_v3")

# Enable static shape precompilation padding to minimize dynamic compiles.
TPU_PRECOMPILATION_ENABLED: bool = _bool("TPU_PRECOMPILATION_ENABLED", default=True)

