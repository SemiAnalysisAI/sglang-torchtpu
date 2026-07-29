"""Unit tests for the TPU torch.compile backend configuration.

The TPU compile policies must not use the registered ``"tpu"`` backend
string (== ``TpuBackend()`` defaults, serialization ON): the stock
serialized-executable cache key degenerates to a per-process constant, so
any process compiling more than one graph collides on reload
(``e0102 RuntimeProgramInputMismatch``). ``resolve_tpu_compile_backend``
returns an explicit ``TpuBackend(enable_serialization=False)`` instead.
CPU-runnable: torch_tpu is faked via sys.modules.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.hardware_backend.tpu.patches import resolve_tpu_compile_backend
from sglang.test.test_utils import CustomTestCase


class _FakeTpuBackend:
    def __init__(self, debug=False, dynamism=False, enable_serialization=True):
        self.debug = debug
        self.dynamism = dynamism
        self.enable_serialization = enable_serialization


def _fake_compile_module() -> types.ModuleType:
    module = types.ModuleType("torch_tpu._internal.compile")
    module.TpuBackend = _FakeTpuBackend
    return module


class TestResolveTpuCompileBackend(CustomTestCase):
    def test_default_disables_serialization(self):
        with patch.dict(
            sys.modules, {"torch_tpu._internal.compile": _fake_compile_module()}
        ):
            backend = resolve_tpu_compile_backend(None)
        self.assertIsInstance(backend, _FakeTpuBackend)
        self.assertFalse(backend.enable_serialization)

    def test_tpu_string_is_replaced(self):
        with patch.dict(
            sys.modules, {"torch_tpu._internal.compile": _fake_compile_module()}
        ):
            backend = resolve_tpu_compile_backend("tpu")
        self.assertIsInstance(backend, _FakeTpuBackend)
        self.assertFalse(backend.enable_serialization)

    def test_explicit_override_passes_through(self):
        self.assertEqual(resolve_tpu_compile_backend("inductor"), "inductor")


class TestJaxEnvScrub(CustomTestCase):
    def test_apply_tpu_env_overrides_removes_jax_owned_env(self):
        import os

        from sglang.srt.hardware_backend.tpu.env_override import (
            apply_tpu_env_overrides,
        )

        with patch.dict(
            os.environ,
            {"TPU_ML_PLATFORM": "JAX", "TPU_ML_PLATFORM_VERSION": "1"},
        ):
            apply_tpu_env_overrides()
            self.assertNotIn("TPU_ML_PLATFORM", os.environ)
            self.assertNotIn("TPU_ML_PLATFORM_VERSION", os.environ)


if __name__ == "__main__":
    unittest.main()
