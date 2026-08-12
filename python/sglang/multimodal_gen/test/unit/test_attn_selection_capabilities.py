# SPDX-License-Identifier: Apache-2.0
"""A2: attention selection receives layer capabilities; TPU routes
causal/GQA to SDPA at selection time instead of raising at construction."""

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
)
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform

PALLAS = "tpu_pallas.TpuPallasAttentionBackend"
SDPA = "sdpa.SDPABackend"


class TestTpuAttentionSelection:
    def _select(self, **kwargs):
        return TpuPlatform.get_attn_backend_cls_str(None, 128, torch.bfloat16, **kwargs)

    def test_default_is_pallas(self):
        assert self._select().endswith(PALLAS)

    def test_causal_routes_to_sdpa(self):
        assert self._select(causal=True).endswith(SDPA)

    def test_gqa_routes_to_sdpa(self):
        assert self._select(num_kv_heads=8).endswith(SDPA)

    def test_explicit_pallas_with_causal_still_routes_to_sdpa(self):
        got = TpuPlatform.get_attn_backend_cls_str(
            AttentionBackendEnum.TPU_PALLAS_FLASH, 128, torch.bfloat16, causal=True
        )
        assert got.endswith(SDPA)

    def test_all_platform_overrides_accept_capability_kwargs(self):
        # Every override must tolerate the new keywords (the selector always
        # passes them); the base signature is the contract.
        import inspect

        for cls in Platform.__subclasses__():
            fn = inspect.unwrap(cls.get_attn_backend_cls_str.__func__)
            params = inspect.signature(fn).parameters
            assert "causal" in params and "num_kv_heads" in params, cls.__name__
