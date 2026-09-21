# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TPU int64/int32/bool cumsum patch."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu import patches
from sglang.test.test_utils import CustomTestCase


class TestCumsumTpu(CustomTestCase):
    def setUp(self):
        patches._apply_cumsum_patch()

    def test_int64_cumsum_past_bf16_limit_simulated(self):
        x = torch.ones((2, 600), dtype=torch.int64)
        expected = torch.arange(1, 601, dtype=torch.int64).expand(2, 600)
        with patch.object(patches, "_is_tpu_tensor", return_value=True):
            torch.testing.assert_close(torch.cumsum(x, dim=1), expected)
            torch.testing.assert_close(x.cumsum(dim=-1), expected)

    def test_bool_and_dtype_and_out_kwargs(self):
        b = torch.tensor([[True, False, True, True]])
        with patch.object(patches, "_is_tpu_tensor", return_value=True):
            res = torch.cumsum(b, dim=1)
            self.assertEqual(res.dtype, torch.int64)
            self.assertEqual(res.tolist(), [[1, 1, 2, 3]])
            out = torch.empty((1, 4), dtype=torch.int32)
            torch.cumsum(b, dim=1, dtype=torch.int32, out=out)
            self.assertEqual(out.tolist(), [[1, 1, 2, 3]])

    @unittest.skipUnless(hasattr(torch, "tpu"), "requires torch_tpu")
    def test_live_tpu_int64_cumsum(self):
        x = torch.ones((2, 512), dtype=torch.int64, device="tpu")
        res = torch.cumsum(x, dim=1).cpu()
        self.assertEqual(res[0, 256].item(), 257)
        self.assertEqual(res[0, 511].item(), 512)


if __name__ == "__main__":
    unittest.main()
