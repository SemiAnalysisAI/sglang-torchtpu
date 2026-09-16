"""Correctness tests for the TPU ``aten::cumsum`` / ``aten::cumsum_`` kernels.

torch_tpu cannot lower an int64 cumulative scan — it aborts the process on a
libtpu ``CHECK`` ("Unsupported conversion from vmreg/vreg to U64", the same s64
lowering bug that breaks ``masked_scatter_``'s scan). ``hardware_backend/tpu/
patches.py`` registers ``PrivateUse1`` kernels that keep the result *exact*:
either a provably lossless int32 scan (when the input dtype and the scanned
length bound the running sum) or an int64 log-step scan built from elementwise
adds.

Because cumsum's default dtype for any integral input *is* int64, the abort was
reachable from ordinary code — ``mask.cumsum(-1)`` on a token mask, or a plain
boolean advanced index like ``hidden[mask]``, which lowers to the same scan.
The latter is what actually killed Qwen-Image's text-encoding stage.

These tests are differential: the TPU result must equal PyTorch's own CPU
result for every case, across dtypes, dims, shapes and — unlike a narrowing
implementation — magnitudes that do not fit in int32. They also pin the two
properties the implementation depends on: that work stays on the device (no
host round trip, no recursion) and that dtype promotion matches PyTorch.

Not covered, by construction: ``torch.cumsum(x, dim, out=an_int64_tensor)``.
``aten::cumsum.out`` is the native primitive the kernels delegate to and cannot
be replaced without losing it, so an int64 ``out=`` still aborts the process —
which is also why there is no test for it here (it would take the runner down
with it). Requires a real TPU device; skipped otherwise.
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
            patches._register_aten_fallbacks()  # installs the kernels under test
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
        # narrow dtypes take the provably-safe int32 path and must still come
        # back as int64.
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

    # --- exactness beyond int32 -------------------------------------------
    # A narrowing implementation passes every test above and still silently
    # returns garbage here: 2**40 truncated to int32 is exactly 0, so the
    # corruption looks like a plausible all-zero position-id tensor.

    def test_values_that_do_not_fit_int32_are_exact(self):
        src = torch.full((4,), 2**40, dtype=torch.int64)
        self._assert_matches_cpu(src, 0)
        self._assert_matches_cpu(src, 0, dtype=torch.int64)

    def test_sum_overflowing_int32_is_exact(self):
        # Each value fits in int32; the running sum does not.
        src = torch.full((8,), 2**30, dtype=torch.int32)
        self._assert_matches_cpu(src, 0)

    def test_negative_and_mixed_magnitudes(self):
        src = torch.tensor([-(2**40), 2**41, -7, 2**31, -(2**33)], dtype=torch.int64)
        self._assert_matches_cpu(src, 0)

    def test_large_values_on_inner_dim(self):
        torch.manual_seed(0)
        src = torch.randint(-(2**45), 2**45, (3, 129), dtype=torch.int64)
        for dim in (0, 1, -1):
            self._assert_matches_cpu(src, dim)

    # --- overloads ---------------------------------------------------------

    def test_inplace_matches_cpu(self):
        # cumsum_ is a separate ATen overload; before it was registered this
        # aborted the process for int64.
        for src in (
            torch.ones(1, 1024, dtype=torch.int64),
            torch.full((6,), 2**40, dtype=torch.int64),
            torch.arange(64, dtype=torch.float32) / 4,
        ):
            ref = src.clone()
            ref.cumsum_(-1)
            got = src.to(self.dev)
            returned = got.cumsum_(-1)
            self.assertIs(returned, got, "cumsum_ must return self")
            self.assertTrue(
                torch.equal(got.cpu(), ref), f"in-place mismatch for {src.dtype}"
            )

    def test_out_variant_with_non_int64_out(self):
        # The out= overload stays native. Non-int64 out is the case production
        # uses (metadata offsets) and must keep working.
        src = torch.arange(1, 65, dtype=torch.int32, device=self.dev)
        out = torch.empty(64, dtype=torch.int32, device=self.dev)
        torch.cumsum(src, 0, dtype=torch.int32, out=out)
        self.assertEqual(out.cpu()[-1].item(), 64 * 65 // 2)

    # --- the shape that actually broke production --------------------------

    def test_bool_mask_advanced_index(self):
        # `hidden[mask]` lowers to a scan; this, not an explicit cumsum call,
        # is what aborted Qwen-Image in _extract_masked_hidden(). Keep it
        # pinned so an overload-coverage regression is caught here.
        torch.manual_seed(0)
        hidden = torch.randn(1, 67, 8)
        mask = torch.zeros(1, 67, dtype=torch.bool)
        mask[0, ::3] = True

        ref = hidden[mask]
        got = hidden.to(self.dev)[mask.to(self.dev)].cpu()
        self.assertEqual(got.shape, ref.shape)
        self.assertTrue(torch.equal(got, ref))

        # ... and the split that follows it in the pipeline.
        lengths = mask.sum(dim=1)
        self.assertEqual(lengths.tolist(), mask.to(self.dev).sum(dim=1).cpu().tolist())

    # --- implementation properties -----------------------------------------

    def test_stays_on_device(self):
        # Neither path may go to host: the bounded path redispatches to
        # aten::cumsum.out and the unbounded one is elementwise adds. Forbid
        # Tensor.cpu during the calls to prove no D2H transfer happens.
        i32 = torch.ones(1, 512, dtype=torch.int32, device=self.dev)
        i64 = torch.ones(1, 512, dtype=torch.int64, device=self.dev)
        big = torch.full((1, 512), 2**40, dtype=torch.int64, device=self.dev)
        with mock.patch.object(
            torch.Tensor,
            "cpu",
            side_effect=AssertionError("Unexpected D2H transfer to CPU"),
        ):
            self.assertEqual(i32.cumsum(-1, dtype=torch.int32).device.type, "tpu")
            self.assertEqual(i32.float().cumsum(-1).device.type, "tpu")
            self.assertEqual(i64.cumsum(-1).device.type, "tpu")
            self.assertEqual(big.cumsum(-1).device.type, "tpu")
            self.assertEqual(big.clone().cumsum_(-1).device.type, "tpu")

    def test_fuzz(self):
        torch.manual_seed(0)
        for i in range(40):
            ndim = int(torch.randint(1, 4, (1,)).item())
            dims = torch.randint(1, 20, (ndim,)).tolist()
            # Alternate between values int32 could hold and values it could not.
            hi = 50 if i % 2 else 2**45
            src = torch.randint(-hi, hi, dims, dtype=torch.int64)
            self._assert_matches_cpu(src, int(torch.randint(0, ndim, (1,)).item()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
