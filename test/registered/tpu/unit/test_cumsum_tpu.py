# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TPU exact integer/bool cumsum fallback."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import unittest

import torch

from sglang.srt.hardware_backend.tpu.env_override import apply_tpu_env_overrides
from sglang.srt.hardware_backend.tpu.patches import _register_aten_fallbacks
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(hasattr(torch, "tpu"), "requires torch_tpu")
class TestCumsumTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        apply_tpu_env_overrides()
        _register_aten_fallbacks()

    def test_int64_and_int32_cumsum_exact_beyond_fp32_mantissa(self):
        # 2**24 + 1 = 16,777,217 cannot be represented in fp32 without rounding.
        base = (1 << 24) + 1
        for dtype in (torch.int64, torch.int32):
            x = torch.tensor([[base, 1, 1, 1]], dtype=dtype, device="tpu")
            expected = torch.tensor([[base, base + 1, base + 2, base + 3]], dtype=dtype)
            torch.testing.assert_close(torch.cumsum(x, dim=1).cpu(), expected)
            torch.testing.assert_close(x.cumsum(dim=-1).cpu(), expected)

    def test_int64_int32_and_bool_cumsum_shapes_and_inplace(self):
        x64 = torch.ones((2, 512), dtype=torch.int64, device="tpu")
        expected64 = torch.arange(1, 513, dtype=torch.int64).expand(2, 512)
        torch.testing.assert_close(torch.cumsum(x64, dim=1).cpu(), expected64)

        x32_inplace = torch.ones((2, 64), dtype=torch.int32, device="tpu")
        x32_inplace.cumsum_(dim=1)
        expected32 = torch.arange(1, 65, dtype=torch.int32).expand(2, 64)
        torch.testing.assert_close(x32_inplace.cpu(), expected32)

        b = torch.tensor([[True, False, True, True]], device="tpu")
        self.assertEqual(torch.cumsum(b, dim=1).cpu().tolist(), [[1, 1, 2, 3]])
        self.assertEqual(
            torch.cumsum(b, dim=1, dtype=torch.int32).cpu().tolist(),
            [[1, 1, 2, 3]],
        )


if __name__ == "__main__":
    unittest.main()
