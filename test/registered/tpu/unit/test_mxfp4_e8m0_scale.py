"""MXFP4 e8m0 scales decode to the exact fp32 power of two on TPU.

Needs a TPU; skipped otherwise.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest

import torch

from sglang.srt.layers.quantization.mxfp4_tpu_util import _e8m0_to_fp32
from sglang.test.test_utils import CustomTestCase


@unittest.skipIf(importlib.util.find_spec("torch_tpu") is None, "needs torch_tpu")
class TestE8m0ScaleDecode(CustomTestCase):
    def test_every_code_is_bit_exact_on_tpu(self):
        """Bug regression: torch.exp2 on TPU is approximate, so decoded scales
        could be an ULP off the power of two the checkpoint encodes."""
        codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
        # float64 ldexp is exact; 2**-127 rounds to the fp32 subnormal and
        # 2**128 to inf, as e8m0 0 and 255 decode.
        expected = torch.ldexp(
            torch.ones(256, dtype=torch.float64), codes.to(torch.int32) - 127
        ).to(torch.float32)

        actual = _e8m0_to_fp32(codes.to("tpu")).cpu()

        self.assertEqual(actual.dtype, torch.float32)
        mismatched = (actual.view(torch.int32) != expected.view(torch.int32)).nonzero()
        self.assertEqual(mismatched.numel(), 0, f"codes off: {mismatched.flatten()}")


if __name__ == "__main__":
    unittest.main()
