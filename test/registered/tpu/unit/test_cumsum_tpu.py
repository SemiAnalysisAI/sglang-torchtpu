# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TPU int64/int32/bool cumsum fallback."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.patches import _register_aten_fallbacks
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(hasattr(torch, "tpu"), "requires torch_tpu")
class TestCumsumTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _register_aten_fallbacks()

    def test_int64_and_bool_cumsum_on_tpu(self):
        x = torch.ones((2, 512), dtype=torch.int64, device="tpu")
        expected = torch.arange(1, 513, dtype=torch.int64).expand(2, 512)
        torch.testing.assert_close(torch.cumsum(x, dim=1).cpu(), expected)
        torch.testing.assert_close(x.cumsum(dim=-1).cpu(), expected)

        b = torch.tensor([[True, False, True, True]], device="tpu")
        self.assertEqual(torch.cumsum(b, dim=1).cpu().tolist(), [[1, 1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
