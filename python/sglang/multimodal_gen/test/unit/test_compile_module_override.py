# SPDX-License-Identifier: Apache-2.0
"""Platform compile_module_override seam: default None -> torch.compile path;
a platform-supplied forward installs once and skips torch.compile."""

from unittest.mock import patch

import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms.interface import Platform
from sglang.multimodal_gen.runtime.utils.torch_compile import CompiledModuleRegistry


class _Mod(nn.Module):
    def forward(self, x):
        return x + 1


class TestCompileModuleOverride:
    def test_default_is_none(self):
        assert Platform.compile_module_override(_Mod()) is None

    def test_install_forward_once(self):
        registry = CompiledModuleRegistry()
        module = _Mod()
        sentinel = lambda x: x  # noqa: E731
        assert registry.install_forward_once(module, sentinel)
        assert module.forward is sentinel
        assert registry.is_compiled(module)
        # Second install is a no-op.
        assert not registry.install_forward_once(module, lambda x: x - 1)
        assert module.forward is sentinel

    def test_tpu_dynamo_default_returns_none(self):
        from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform

        with patch(
            "sglang.multimodal_gen.envs.SGLANG_DIFFUSION_TPU_COMPILE_TRACER",
            "dynamo",
        ):
            assert TpuPlatform.compile_module_override(_Mod()) is None

    def test_tpu_unknown_tracer_raises(self):
        import pytest

        from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform

        with patch(
            "sglang.multimodal_gen.envs.SGLANG_DIFFUSION_TPU_COMPILE_TRACER",
            "typo",
        ):
            with pytest.raises(ValueError, match="dynamo, direct"):
                TpuPlatform.compile_module_override(_Mod())
