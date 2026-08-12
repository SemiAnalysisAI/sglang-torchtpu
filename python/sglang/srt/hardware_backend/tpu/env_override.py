# SPDX-License-Identifier: Apache-2.0
"""TPU env-var overrides.

Call ``apply_tpu_env_overrides()`` from TPU setup before ``torch_tpu`` is
imported. libtpu reads ``LIBTPU_INIT_ARGS`` at load time.
"""

from __future__ import annotations

import os

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

_LIBTPU_INIT_ARGS = "LIBTPU_INIT_ARGS"

# Append instead of setdefault so caller-supplied flags are preserved.
_SPARSE_CORE_OFFLOAD_OVERRIDES = (
    # TODO(wyzhang): drop after the torchtpu fix.
    # SparseCore offload for the all-reduce after embed lookup can return
    # wrong values inside compiled regions on TP=8 dense Qwen.
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
)


def _override_libtpu_flags(flags: tuple[str, ...]) -> list[str]:
    existing = os.environ.get(_LIBTPU_INIT_ARGS, "")
    existing_flags = set(existing.split())
    to_append = [f for f in flags if f not in existing_flags]
    if not to_append:
        return []

    os.environ[_LIBTPU_INIT_ARGS] = f"{existing} {' '.join(to_append)}".strip()
    return to_append


# Opting out of the blanket disable above requires a SIZE THRESHOLD, not a bare
# `=true`. The torchtpu bug is in the all-reduce after the embed lookup, which is a
# *small* collective ([batch, hidden] bf16 -- ~131 KB at a decode batch of 16), while
# the collective worth offloading is the prefill all-reduce ([chunk, hidden] -- ~33.5 MB
# at chunk 4096), measured at 35.8% of prefill chunk time. XLA can distinguish them by
# size, so a threshold between the two keeps the buggy case on the TensorCore while
# offloading the expensive one.
#
# Same mechanism as tpu-inference `eaa02d3d1`, which sets
# `xla_tpu_sparse_core_{all_reduce,all_gather}_offload_min_size_in_bytes`.
_SC_ALL_REDUCE_MIN_SIZE_FLAG = "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"


def _has_sc_all_reduce_size_threshold() -> bool:
    """True when the caller set a positive SC all-reduce offload size threshold."""
    for flag in os.environ.get(_LIBTPU_INIT_ARGS, "").split():
        name, _, value = flag.partition("=")
        if name == _SC_ALL_REDUCE_MIN_SIZE_FLAG:
            try:
                return int(value) > 0
            except ValueError:
                return False
    return False


def apply_tpu_env_overrides() -> None:
    """Apply TPU env overrides."""
    if _has_sc_all_reduce_size_threshold():
        logger.info(
            "TPU env_override: %s is set, so the blanket SparseCore all-reduce "
            "offload disable is skipped -- the threshold is expected to exclude the "
            "small post-embed all-reduce that motivated it.",
            _SC_ALL_REDUCE_MIN_SIZE_FLAG,
        )
        return
    appended = _override_libtpu_flags(_SPARSE_CORE_OFFLOAD_OVERRIDES)
    if appended:
        logger.info(
            "TPU env_override appended %d LIBTPU_INIT_ARGS flag(s): %s",
            len(appended),
            " ".join(appended),
        )
    else:
        logger.debug("TPU env_override: LIBTPU_INIT_ARGS already has SC-off flags")
