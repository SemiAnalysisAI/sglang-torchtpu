# SPDX-License-Identifier: Apache-2.0
"""Environment variables owned by the TPU plugin.

Declare new TPU-only variables here with the descriptor types from
`sglang.srt.environ`. Existing ``SGLANG_TPU_*`` entries stay in core until their
core readers move; never declare one name in both registries.
"""

from __future__ import annotations

from sglang.srt.environ import EnvBool, EnvField, EnvFloat, EnvInt, EnvStr

__all__ = ["TpuEnvs", "envs"]

# Core closes EnvField.__set_name__ after its own Envs, reopen it for this class.
EnvField._allow_set_name = True


class TpuEnvs:
    # Host heap (see host_heap_trim.py)
    SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED = EnvBool(True)
    # Seconds between periodic trims. 0 turns the periodic trim off.
    SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC = EnvFloat(0.0)

    # Kimi-K3 decode megakernel (see kimi_k3_megakernel/). Experimental.
    # Run decode steps through the single-op Pallas megakernel.
    SGLANG_TPU_K3_MEGAKERNEL = EnvBool(False)
    # Checkpoint the megakernel's dense families load from; defaults to the
    # served model path.
    SGLANG_TPU_K3_MEGAKERNEL_CHECKPOINT = EnvStr(None)
    # Tokens of decode state per row; defaults to --context-length.
    SGLANG_TPU_K3_MEGAKERNEL_CONTEXT = EnvInt(None)
    # Decode rows the megakernel state holds (1-8); caps --max-running-requests.
    SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS = EnvInt(8)
    # Kernel tuning recipe passed to the vendored decode program.
    SGLANG_TPU_K3_MEGAKERNEL_RECIPE = EnvStr("tuned")

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
