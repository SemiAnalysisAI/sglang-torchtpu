"""Correctness tests for the TPU ``aten::cumsum`` fallback kernel.

torch_tpu cannot lower an int64 ``cumsum`` — it aborts the process on a libtpu
``CHECK`` ("Unsupported conversion from vmreg/vreg to U64", the same s64 lowering
bug that breaks ``masked_scatter_``'s scan). ``hardware_backend/tpu/patches.py``
registers a ``PrivateUse1`` kernel that computes in int32 and casts back.

Because cumsum's default dtype for any integral input *is* int64, the abort was
reachable from ordinary code: ``mask.cumsum(-1)`` on a token mask was enough to
kill a Qwen-VL prefill.

These tests are differential: the TPU kernel's output must equal PyTorch's own
CPU ``cumsum`` for every case, across dtypes, dims and shapes. They also pin the
two properties the implementation depends on — that non-int64 work stays on the
device (no host round trip, no recursion) and that dtype promotion matches
PyTorch. Requires a real TPU device; skipped otherwise.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase


class TestCumsumTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.hardware_backend.tpu import patches
            from sglang.srt.hardware_backend.tpu.env_override import (
                apply_tpu_env_overrides,
            )

            # Serving applies these in _prepare_tpu_env() before importing
            # torch_tpu; this test reaches torch_tpu directly, so it opts in
            # before import and before the first device op initializes PJRT.
            apply_tpu_env_overrides()

            import torch_tpu  # noqa: F401

            cls.dev = "tpu"
            torch.zeros(1, device=cls.dev)  # probe: a real TPU is present
            patches._register_aten_fallbacks()  # installs the kernel under test
        except Exception as e:  # no torch_tpu / no device -> skip, don't fail
            raise unittest.SkipTest(f"TPU not available: {e}")

    def _assert_matches_cpu(self, src, dim, dtype=None):
        """TPU cumsum must match CPU cumsum bit-for-bit, dtype included."""
        ref = src.cumsum(dim, dtype=dtype)
        got = src.to(self.dev).cumsum(dim, dtype=dtype).cpu()
        self.assertEqual(
            got.dtype,
            ref.dtype,
            f"dtype drift for {src.dtype} dim={dim} dtype={dtype}",
        )
        self.assertTrue(
            torch.equal(got, ref),
            f"TPU != CPU for {tuple(src.shape)}/{src.dtype} dim={dim} dtype={dtype}",
        )

    def test_int64_default_dtype(self):
        # The Qwen-VL prefill shape: cumsum of an all-ones int64 attention mask.
        # This is the case that aborted the process before the fallback.
        self._assert_matches_cpu(torch.ones(1, 1024, dtype=torch.int64), -1)

    def test_integral_inputs_promote_to_int64(self):
        # cumsum promotes every integral input to int64 unless told otherwise;
        # the fallback computes in int32, so the cast back must restore it.
        for dt in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32):
            src = torch.ones(64, dtype=dt)
            self._assert_matches_cpu(src, 0)
            self.assertEqual(src.to(self.dev).cumsum(0).dtype, torch.int64)

    def test_explicit_dtypes(self):
        src = torch.randint(0, 7, (4, 64), dtype=torch.int64)
        for dtype in (torch.int32, torch.int64, torch.float32):
            self._assert_matches_cpu(src, -1, dtype=dtype)

    def test_floating_inputs_are_untouched(self):
        for dt in (torch.float32, torch.bfloat16):
            self._assert_matches_cpu(torch.arange(128, dtype=dt) / 8, 0)

    def test_dims_and_shapes(self):
        torch.manual_seed(0)
        src = torch.randint(0, 5, (3, 7, 11), dtype=torch.int64)
        for dim in range(src.dim()):
            self._assert_matches_cpu(src, dim)
            self._assert_matches_cpu(src, dim - src.dim())  # negative indexing
        self._assert_matches_cpu(torch.zeros(0, dtype=torch.int64), 0)  # empty
        self._assert_matches_cpu(torch.tensor([5], dtype=torch.int64), 0)  # single

    def test_non_multiple_of_1024_length(self):
        # libtpu's associative-scan emitter miscompiles lengths that are not a
        # multiple of 1024 (torch_tpu#2898); env_override disables it. A wrong
        # value here means that override stopped being applied.
        torch.manual_seed(0)
        self._assert_matches_cpu(torch.randint(0, 9, (4, 333), dtype=torch.int64), -1)

    def test_non_int64_stays_on_device(self):
        # The fallback redispatches through aten::cumsum.out rather than going to
        # host. A regression to a CPU fallback (like _upsample_bicubic2d) would
        # still return a tensor with device.type == "tpu", so forbid Tensor.cpu
        # during the call to prove no D2H transfer occurs.
        src = torch.ones(1, 512, dtype=torch.int32, device=self.dev)
        src_i64 = torch.ones(1, 512, dtype=torch.int64, device=self.dev)
        with mock.patch.object(
            torch.Tensor,
            "cpu",
            side_effect=AssertionError("Unexpected D2H transfer to CPU"),
        ):
            self.assertEqual(src.cumsum(-1, dtype=torch.int32).device.type, "tpu")
            self.assertEqual(src.float().cumsum(-1).device.type, "tpu")
            self.assertEqual(src_i64.cumsum(-1).device.type, "tpu")

    def test_fuzz(self):
        torch.manual_seed(0)
        for _ in range(40):
            ndim = int(torch.randint(1, 4, (1,)).item())
            dims = torch.randint(1, 20, (ndim,)).tolist()
            src = torch.randint(-50, 50, dims, dtype=torch.int64)
            self._assert_matches_cpu(src, int(torch.randint(0, ndim, (1,)).item()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
