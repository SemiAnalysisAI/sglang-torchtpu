"""CPU eligibility/import coverage for the opt-in fused GLM expert pair."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[5]
_HELPER = (
    _ROOT / "python/sglang/srt/hardware_backend/tpu/glm_fp8_fused_experts.py"
)
_ENVIRON = _ROOT / "python/sglang/srt/environ.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGlmFp8FusedExpertsGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load_file("glm_fused_experts_guard_test", _HELPER)

    def test_default_off_and_explicit_environment_opt_in(self):
        environ = load_file("glm_fused_experts_env_test", _ENVIRON)
        name = "SGLANG_TPU_GLM_FUSED_EXPERTS"
        with mock.patch.dict(os.environ):
            os.environ.pop(name, None)
            self.assertFalse(environ.envs.SGLANG_TPU_GLM_FUSED_EXPERTS.get())
            os.environ[name] = "1"
            self.assertTrue(environ.envs.SGLANG_TPU_GLM_FUSED_EXPERTS.get())
            os.environ[name] = "0"
            self.assertFalse(environ.envs.SGLANG_TPU_GLM_FUSED_EXPERTS.get())

    def test_cpu_import_and_ineligible_paths_do_not_import_jax_or_torch(self):
        # Use a fresh interpreter: CI may already have imported these modules.
        # Loading this file directly isolates the helper's import contract
        # from the public sglang package's unrelated initialization imports.
        script = """
import importlib.abc
import importlib.util
import sys

class RejectDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('jax', 'torch', 'torch_tpu'):
            raise AssertionError('Unexpected device import: ' + fullname)

sys.meta_path.insert(0, RejectDeviceImports())
spec = importlib.util.spec_from_file_location('isolated_fused_experts', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not module.supports_glm_fused_experts(False, fp32_scales=False)
assert not module.supports_glm_fused_experts(True, fp32_scales=True)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(_HELPER)],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_qualified_hardware_and_each_unsupported_field(self):
        values = dict(
            chip_version=SimpleNamespace(value="7x"),
            mxu_column_size=256,
            num_lanes=128,
            fp8_ops_per_second=2300000000000000,
            vmem_capacity_bytes=67108864,
        )

        def eligible(info, *, bs1=True, fp32=False):
            return self.helper.supports_glm_fused_experts(
                bs1, fp32_scales=fp32, tpu_info=SimpleNamespace(**info)
            )

        self.assertTrue(eligible(values))
        self.assertFalse(eligible(values, bs1=False))
        self.assertFalse(eligible(values, fp32=True))
        for field, unsupported in (
            ("chip_version", SimpleNamespace(value="6e")),
            ("mxu_column_size", 128),
            ("num_lanes", 64),
            ("fp8_ops_per_second", 0),
            ("vmem_capacity_bytes", 33554432),
        ):
            with self.subTest(field=field):
                changed = {**values, field: unsupported}
                self.assertFalse(eligible(changed))
                del changed[field]
                self.assertFalse(eligible(changed))


if __name__ == "__main__":
    unittest.main()
