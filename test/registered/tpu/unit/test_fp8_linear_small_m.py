# SPDX-License-Identifier: Apache-2.0
"""The single-kernel small-M W8A8 linear must be bit-identical to the XLA path."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu import fp8_linear
from sglang.test.ci.ci_register import register_tpu_ci

try:
    import torch_tpu  # noqa: F401
except ImportError:
    torch_tpu = None

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

# GLM-5.3-Flash TP8 dense FP8 linears: shared experts, MLA, dense MLP.
SHAPES = [(4096, 512), (256, 4096), (4096, 2048), (2048, 4096), (1536, 4096)]


@unittest.skipUnless(torch_tpu is not None, "Requires torch_tpu")
class TestFp8LinearSmallM(unittest.TestCase):
    def test_kernel_matches_xla_path(self):
        fp8_linear.register_fp8_linear_small_m_op()
        op = fp8_linear._SMALL_M_OP
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

    def test_large_batches_keep_xla_path(self):
        self.assertFalse(fp8_linear._small_m_supports(17, 4096, 512))
        self.assertFalse(fp8_linear._small_m_supports(1, 4096, 100))


if __name__ == "__main__":
    unittest.main()
