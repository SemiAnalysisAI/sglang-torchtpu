"""TPU-specific environment variable registry.

This registry is primarily for customizing TPU-specific behavior during
feature development. Once an environment variable stabilizes, we may consider 
promoting it to `sglang.srt.environ.Envs` and removing it from here.

Sample usage:
    from sglang.srt.hardware_backend.tpu import envs
    if envs.TPU_COMPILATION_MODE:
        ...
"""

from __future__ import annotations

import os

_TRUTHY = frozenset(("1", "true", "yes", "on"))


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in _TRUTHY


def _str(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


# When set to true, enable compilation mode.
TPU_COMPILATION_MODE: bool = _bool("TPU_COMPILATION_MODE", default=False)

# Propagates the resolved `--attention-backend` CLI argument to spawned TP
# workers. Can also be set manually by the user to override the backend.
TPU_ACTIVE_ATTENTION_BACKEND: str | None = _str("TPU_ACTIVE_ATTENTION_BACKEND", default=None)
