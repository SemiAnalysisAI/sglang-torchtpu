"""CPU tests for SiluAndMul.forward_tpu and SGLANG_TPU_ACTIVATION_FP32."""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels import fused_op
from sglang.srt.environ import envs, exportable_env_vars
from sglang.srt.layers.activation import SiluAndMul

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _gate_up(scale: float, tokens: int = 64, d: int = 512) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return (torch.randn(tokens, 2 * d, generator=g) * scale).to(torch.bfloat16)


def _op(fp32: bool) -> SiluAndMul:
    with envs.SGLANG_TPU_ACTIVATION_FP32.override(fp32):
        return SiluAndMul()


class TestSiluAndMulTpuFp32(CustomTestCase):
    def test_off_by_default_and_bit_identical_to_native(self):
        self.assertIs(envs.SGLANG_TPU_ACTIVATION_FP32.default, False)
        op = SiluAndMul()
        x = _gate_up(100.0)
        self.assertTrue(torch.equal(op.forward_tpu(x), op.forward_native(x)))

    def test_on_rounds_silu_once_and_keeps_the_rest_in_model_dtype(self):
        # The RL trainer computes silu(gate.float()).to(dtype) * up; both sides
        # must use the same formula, so this pins it bitwise.
        x = _gate_up(100.0)
        d = x.shape[-1] // 2
        out = _op(True).forward_tpu(x)
        expected = F.silu(x[..., :d].float()).to(torch.bfloat16) * x[..., d:]
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, expected))

    def test_tpu_dispatch_selects_forward_tpu(self):
        # The in-tree TPU platform, then the out-of-tree plugin's "tpu" dispatch key.
        for platform_key, oot_key in (("tpu", None), ("", "tpu")):
            with (
                mock.patch.object(fused_op, "_platform_key", lambda: platform_key),
                mock.patch.object(fused_op, "_oot_dispatch_key", lambda: oot_key),
            ):
                op = _op(True)
                self.assertEqual(op.dispatch_forward(), op.forward_tpu)

    def test_setting_is_exported_with_server_env_vars(self):
        # External-engine RL trainers read this setting from /server_info
        # (SGLANG_EXPOSE_OWN_ENV_VARS), which only exports variables in Envs.
        with envs.SGLANG_TPU_ACTIVATION_FP32.override(True):
            exported = exportable_env_vars()
        self.assertIn("SGLANG_TPU_ACTIVATION_FP32", exported)


if __name__ == "__main__":
    unittest.main()
