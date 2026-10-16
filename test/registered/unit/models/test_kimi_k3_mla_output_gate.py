import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import mla_output_gate
from sglang.kernels.ops.attention.mla_output_gate import covered
from sglang.kernels.ops.gemm import kimi_k3 as kimi_k3_gemm
from sglang.kernels.ops.gemm.kimi_k3 import kimi_k3_tiny_gemm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKimiK3MlaOutputGate(CustomTestCase):
    def test_cpu_tensors_use_fallback(self):
        x = torch.zeros(2, 8, dtype=torch.bfloat16)
        gate = torch.zeros_like(x)

        self.assertFalse(covered(x, gate))

    def test_non_cuda_platforms_use_fallback_even_when_is_cuda_true(self):
        class _FakeCudaTensor:
            is_cuda = True
            dtype = torch.bfloat16
            shape = (2, 8)

            def is_contiguous(self):
                return True

            def numel(self):
                return 16

        x = _FakeCudaTensor()
        gate = _FakeCudaTensor()
        for flag in ("_is_tpu", "_is_hip", "_is_npu"):
            with (
                self.subTest(flag=flag),
                patch.object(mla_output_gate, "_is_tpu", flag == "_is_tpu"),
                patch.object(mla_output_gate, "_is_hip", flag == "_is_hip"),
                patch.object(mla_output_gate, "_is_npu", flag == "_is_npu"),
            ):
                self.assertFalse(covered(x, gate))

    def test_tiny_gemm_falls_back_to_linear_on_cpu_and_non_cuda_platforms(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        w = torch.randn(1536, 128, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, w)
        with patch(
            "sglang.kernels.ops.gemm.tiny_gemm.tiny_gemm_bf16",
            side_effect=AssertionError("CUDA tiny_gemm should not be called"),
        ):
            torch.testing.assert_close(kimi_k3_tiny_gemm(x, w), expected)
            for flag in ("_is_tpu", "_is_hip", "_is_npu"):
                with (
                    self.subTest(flag=flag),
                    patch.object(kimi_k3_gemm, flag, True),
                ):
                    torch.testing.assert_close(kimi_k3_tiny_gemm(x, w), expected)


if __name__ == "__main__":
    unittest.main()
