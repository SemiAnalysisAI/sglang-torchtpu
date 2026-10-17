# SPDX-License-Identifier: Apache-2.0
"""The single-kernel small-M W8A8 linear must be bit-identical to the XLA path."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu import fp8_linear
from sglang_tpu.kernels import fp8_linear_small_m

# Representative (K, N) shapes of dense FP8 linears.
SHAPES = [(4096, 512), (256, 4096), (4096, 2048), (2048, 4096), (1536, 4096)]


class TestFp8LinearSmallM(CustomTestCase):
    def test_kernel_matches_xla_path(self):
        fp8_linear.register_fp8_linear_small_m_op()
        op = fp8_linear._SMALL_M_OP
        self.assertIsNotNone(op)
        gen = torch.Generator().manual_seed(3)
        # 5 tokens: a row block that is not a sublane multiple, with no pad.
        for tokens in (1, 5, 16):
            for k, n in SHAPES:
                with self.subTest(tokens=tokens, k=k, n=n):
                    x = torch.randn(tokens, k, generator=gen).bfloat16().to("tpu")
                    w = (
                        (torch.randn(k, n, generator=gen) * 0.5)
                        .to(torch.float8_e4m3fn)
                        .to("tpu")
                    )
                    ws = (torch.rand(n, generator=gen) * 0.01 + 1e-3).to("tpu")
                    try:
                        fp8_linear._SMALL_M_OP = None
                        expected = fp8_linear.apply_tpu_fp8_linear(x, w, ws).cpu()
                    finally:
                        fp8_linear._SMALL_M_OP = op
                    actual = fp8_linear.apply_tpu_fp8_linear(x, w, ws).cpu()
                    self.assertEqual(actual.shape, (tokens, n))
                    self.assertTrue(torch.equal(actual, expected))

    def test_unsupported_shapes_keep_xla_path(self):
        self.assertTrue(fp8_linear._small_m_supports(16, 4096, 512))
        self.assertFalse(fp8_linear._small_m_supports(17, 4096, 512))
        self.assertFalse(fp8_linear._small_m_supports(1, 4096, 100))
        self.assertFalse(fp8_linear._small_m_supports(1, 4000, 512))
        # The widest N tile whose double-buffered FP8 block fits the VMEM budget.
        self.assertEqual(fp8_linear_small_m._tile_n(16384, 1024), 512)
        self.assertFalse(fp8_linear._small_m_supports(1, 131072, 128))


if __name__ == "__main__":
    unittest.main()
