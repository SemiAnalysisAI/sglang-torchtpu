"""Manual TPU tests for dense FP8 linear."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.fp8_linear import (
    apply_tpu_fp8_linear,
    prebuild_tpu_fp8_linear_op,
)
from sglang.test.test_utils import CustomTestCase


class TestTpuFp8Linear(CustomTestCase):
    def test_fp8_linear_runs_eager_and_compiled(self):
        prebuild_tpu_fp8_linear_op()

        device = torch.device("tpu")
        x = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        weight_fp32 = torch.randn(128, 32, dtype=torch.float32)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        weight_scale = (
            weight_fp32.abs().amax(dim=0).clamp(min=1e-12) / fp8_max
        ).float()
        weight = (
            (weight_fp32 / weight_scale)
            .clamp(min=-fp8_max, max=fp8_max)
            .to(torch.float8_e4m3fn)
            .to(device)
        )
        weight_scale = weight_scale.to(device)

        eager_out = apply_tpu_fp8_linear(x, weight, weight_scale)

        def run_fp8_linear(x, weight, weight_scale):
            return apply_tpu_fp8_linear(x, weight, weight_scale)

        compiled = torch.compile(
            run_fp8_linear,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        compiled_out = compiled(x, weight, weight_scale)

        self.assertEqual(eager_out.shape, (4, 32))
        self.assertEqual(eager_out.dtype, torch.bfloat16)
        torch.testing.assert_close(compiled_out.cpu(), eager_out.cpu())


if __name__ == "__main__":
    unittest.main()
