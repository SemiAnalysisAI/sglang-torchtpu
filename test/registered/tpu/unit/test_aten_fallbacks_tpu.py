# SPDX-License-Identifier: Apache-2.0
"""srt's torch_tpu ATen fallbacks match the CPU kernels."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import unittest

import torch

from sglang.srt.hardware_backend.tpu.env_override import apply_tpu_env_overrides
from sglang.srt.hardware_backend.tpu.patches import (
    _register_aten_fallbacks,
    _register_int64_cumsum_fallback,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(hasattr(torch, "tpu"), "requires torch_tpu")
class TestAtenFallbacksTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        apply_tpu_env_overrides()
        _register_aten_fallbacks()

    def test_logaddexp(self):
        gen = torch.Generator().manual_seed(0)
        a = torch.randn(2, 8, 1024, generator=gen) * 30
        b = torch.randn(2, 8, 1024, generator=gen) * 30
        a[0, 0, :2] = float("-inf")
        b[0, 0, :4] = float("-inf")
        got = torch.logaddexp(a.to("tpu"), b.to("tpu")).cpu()
        torch.testing.assert_close(got, torch.logaddexp(a, b), rtol=1e-5, atol=1e-5)

    def test_logaddexp_broadcasts(self):
        a = torch.randn(4, 1, 256)
        b = torch.randn(1, 16, 256)
        got = torch.logaddexp(a.to("tpu"), b.to("tpu")).cpu()
        torch.testing.assert_close(got, torch.logaddexp(a, b), rtol=1e-5, atol=1e-5)


@unittest.skipUnless(hasattr(torch, "tpu"), "requires torch_tpu")
class TestCumsumTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        apply_tpu_env_overrides()
        _register_int64_cumsum_fallback()

    def _check(self, ref, dim, **kwargs):
        got = torch.cumsum(ref.to("tpu"), dim, **kwargs).cpu()
        torch.testing.assert_close(got, torch.cumsum(ref, dim, **kwargs))

    def test_int64_result_is_exact(self):
        # 2**24 + 1 is not representable in float32.
        base = (1 << 24) + 1
        for dtype in (torch.int64, torch.int32):
            ref = torch.tensor([[base, 1, 1, 1], [-base, 7, -3, 1]], dtype=dtype)
            with self.subTest(dtype=dtype):
                # dtype=None promotes integral inputs to int64.
                self._check(ref, 1)
                self._check(ref, -1, dtype=torch.int64)

    def test_bool_and_narrow_integer_inputs(self):
        gen = torch.Generator().manual_seed(0)
        mask = torch.rand(3, 1000, generator=gen) > 0.3
        self._check(mask, -1)
        self._check(mask, 1, dtype=torch.int32)
        small = torch.randint(-100, 100, (2, 777), generator=gen, dtype=torch.int16)
        self._check(small, 1)
        self._check(small, 0)

    def test_int32_and_float_results_keep_the_native_kernel(self):
        gen = torch.Generator().manual_seed(1)
        ints = torch.randint(0, 1 << 20, (4, 3000), generator=gen, dtype=torch.int32)
        self._check(ints, 1, dtype=torch.int32)
        floats = torch.randn(4, 3000, generator=gen)
        got = torch.cumsum(floats.to("tpu"), 1).cpu()
        torch.testing.assert_close(got, torch.cumsum(floats, 1), rtol=1e-4, atol=1e-3)

    def test_inplace_and_scalar(self):
        for dtype in (torch.int64, torch.int32):
            with self.subTest(dtype=dtype):
                x = torch.ones((2, 64), dtype=dtype, device="tpu")
                x.cumsum_(dim=1)
                self.assertEqual(x.dtype, dtype)
                torch.testing.assert_close(
                    x.cpu(), torch.arange(1, 65, dtype=dtype).expand(2, 64)
                )
        self._check(torch.tensor(5), 0)

    def test_result_does_not_alias_the_input(self):
        for shape, dim in (((), 0), ((3, 1), 1)):
            with self.subTest(shape=shape):
                x = torch.full(shape, 4, dtype=torch.int64, device="tpu")
                y = x.cumsum(dim)
                y += 10
                self.assertTrue(torch.all(x.cpu() == 4))


if __name__ == "__main__":
    unittest.main()
