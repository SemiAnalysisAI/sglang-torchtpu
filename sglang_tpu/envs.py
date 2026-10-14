# SPDX-License-Identifier: Apache-2.0
"""Environment variables owned by the TPU plugin.

Declare new TPU-only variables here with the descriptor types from
`sglang.srt.environ`. Existing ``SGLANG_TPU_*`` entries stay in core until their
core readers move; never declare one name in both registries.
"""

from __future__ import annotations

from sglang.srt.environ import EnvBool, EnvField, EnvFloat

__all__ = ["TpuEnvs", "envs"]

# Core closes EnvField.__set_name__ after its own Envs, reopen it for this class.
EnvField._allow_set_name = True


class TpuEnvs:
    # Host heap (see host_heap_trim.py)
    SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED = EnvBool(True)
    # Seconds between periodic trims. 0 turns the periodic trim off.
    SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC = EnvFloat(0.0)
    # torch.compile (see TpuPlatform.on_warmup_complete)
    # After warmup, evaluate only the guards that tell the compiled graphs
    # apart instead of every guard on every call (~1.4 ms of host time per
    # speculative round). Warmup compiles every serving shape and recompiles
    # are frozen, so the remaining guards still pick the right graph; set 0 to
    # evaluate every guard when debugging a suspected wrong-graph dispatch.
    SGLANG_TPU_OPT_SKIP_GUARD_EVAL = EnvBool(True)
    # KDA (attention_gdn_pallas)
    # Bounded-gate KDA prefill through the sglang-jax Mega kernel
    # (kernels/mega_kda.py) instead of the fused GDN v3 kernel. Off: faster in
    # isolation (0.64 vs 0.79 ms/layer at 8k) but not in a served GLM-5.3-Flash
    # prefill, where the XLA conv glue and the fallback cond cost the gain.
    SGLANG_TPU_OPT_MEGA_KDA_PREFILL = EnvBool(False)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
