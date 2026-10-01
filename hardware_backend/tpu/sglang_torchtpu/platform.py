# SPDX-License-Identifier: Apache-2.0
"""The out-of-tree TPU platform.

Subclasses the fork's in-tree ``TpuSRTPlatform`` rather than reimplementing it:
wave 0 moves no code, so every factory, capability and device method is
inherited unchanged. What this class adds is exactly two things:

1. **Identity.** ``PlatformEnum.OOT`` instead of ``PlatformEnum.TPU``, so the
   27 ``is_out_of_tree()`` gates in the fork's core tree fire. ``is_tpu()`` is
   overridden to stay True so the fork's ~300 ``is_tpu()`` branch sites keep
   working during the migration.
2. **The methods those gates require** that the in-tree platform does not
   implement, either because upstream renamed them or because the gate arrived
   after the fork's last sync.

Selecting this platform is opt-in (``SGLANG_PLATFORM=tpu_oot``) until the two
divergences in the package README are closed.
"""

from __future__ import annotations

from typing import Callable, Optional

from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform
from sglang.srt.platforms.device_mixin import PlatformEnum

# Speculative algorithms the fork validates and tests on TPU. DSPARK is in
# flight (fork PR #707) and is deliberately absent: a capability that lies is
# worse than a missing one.
_SUPPORTED_SPEC_ALGORITHMS = frozenset({"EAGLE", "EAGLE3", "DFLASH"})

# Draft attention backends registered by `speculative/draft_utils.py` for TPU.
_SUPPORTED_DRAFT_ATTENTION_BACKENDS = frozenset(
    {"pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched"}
)
_DEFAULT_DRAFT_ATTENTION_BACKEND = "pallas_tpu_rpa_v3"


class TpuOOTPlatform(TpuSRTPlatform):
    """TPU platform presented to SGLang as an out-of-tree platform."""

    _enum = PlatformEnum.OOT

    def is_tpu(self) -> bool:
        """True despite ``_enum`` being OOT.

        ``DeviceMixin.is_tpu()`` compares against ``PlatformEnum.TPU``, which
        this class no longer declares. The fork's core tree still branches on
        ``is_tpu()`` in ~300 places and 8 of its OOT gates read
        ``is_out_of_tree() or is_tpu()``; both must keep seeing True until
        those sites are migrated.
        """
        return True

    # ---- methods the OOT gates call that the in-tree platform lacks ----

    def get_dsa_kv_pool_cls(self) -> type:
        """Refuse DSA pool construction with a message that names TPU.

        Reached from ``kv_cache_configurator._build_oot_dsa_kv_pool`` (gate at
        ``kv_cache_configurator.py:1318``), an arm that could not fire while the
        platform declared ``PlatformEnum.TPU``.

        There is nothing to return: the fork ships no DSA/NSA pool for TPU --
        ``TpuSRTPlatform.get_nsa_kv_pool_cls`` is itself ``raise
        NotImplementedError("NSA is not supported on TPU.")``. So this is not the
        rename it looks like, and delegating to either spelling only re-raises.
        What the override buys is the diagnostic: without it the failure is a bare
        ``NotImplementedError`` from ``SRTPlatform`` with no indication that the
        model is simply unsupported here.
        """
        raise NotImplementedError(
            "DSA / sparse attention (DeepSeek V3.2 family) has no KV pool "
            "implementation on TPU; neither get_dsa_kv_pool_cls nor the fork's "
            "get_nsa_kv_pool_cls resolves to a pool class. Run this model on a "
            "platform that supports DSA."
        )

    def get_full_graph_backend_cls(self) -> Optional[type]:
        """TPU drives whole-graph compilation through its own graph runner.

        Upstream default is ``None``; stated explicitly because the fork's
        platform predates the hook (upstream #37969, merged after our snapshot)
        and a silent inherited default would read as untested.
        """
        return None

    # ---- speculative decoding (upstream #38740, merged after our snapshot) ----
    # These gates do not exist in the fork's core tree yet. They are implemented
    # ahead of the sync that brings them in: two of the three fail loudly at
    # argument resolution and the third fails quietly by rejecting our own
    # draft backends.

    def supports_speculative_algorithm(self, algorithm: str) -> bool:
        return str(algorithm).upper() in _SUPPORTED_SPEC_ALGORITHMS

    def supports_speculative_draft_attention_backend(
        self, algorithm: str, backend: str
    ) -> bool:
        if not self.supports_speculative_algorithm(algorithm):
            return False
        return backend in _SUPPORTED_DRAFT_ATTENTION_BACKENDS

    def get_default_speculative_draft_attention_backend(self, algorithm: str) -> str:
        if not self.supports_speculative_algorithm(algorithm):
            raise ValueError(
                f"Speculative algorithm {algorithm!r} is not supported on TPU; "
                f"supported: {sorted(_SUPPORTED_SPEC_ALGORITHMS)}."
            )
        return _DEFAULT_DRAFT_ATTENTION_BACKEND

    def get_speculative_cache_locs_fn(self) -> Optional[Callable]:
        """No OOT override: the fork still routes cache locs via ``_is_tpu``.

        Returning a function here while the in-tree branches are also live would
        give two implementations of the same decision. Revisit when
        ``speculative/`` migrates (proposal section 13).
        """
        return None
