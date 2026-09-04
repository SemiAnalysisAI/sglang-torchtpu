"""Tests for padding the multimodal scatter source on TPU.

The TPU write-back of vision embeddings goes through masked_scatter_, whose
destination and mask are already token-bucketed but whose source length is the
raw multimodal token count. That lone dynamic operand made torch_tpu compile a
fresh XLA graph per token count. Padding the source to the destination length
removes it -- which is only sound because masked_scatter_ ignores rows beyond
mask.sum(). These tests pin down that soundness argument.

    python -m pytest test/registered/unit/managers/test_mm_scatter_source_padding.py -v
"""

import unittest

import torch

from sglang.srt.managers.mm_utils import _pad_scatter_source
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BUCKET = 12
HIDDEN = 4


def _tpu_masked_scatter(dest, mask, source):
    """The TPU kernel override from hardware_backend/tpu/patches.py.

    Reproduced here because the real one is registered for PrivateUse1 and the
    padding argument has to hold for it, not just for the native op.
    """
    src_flat = source.to(dest.dtype).reshape(-1)
    flat_mask = mask.to(torch.bool).broadcast_to(dest.shape).reshape(-1)
    idx = (torch.cumsum(flat_mask.to(torch.int32), 0, dtype=torch.int32) - 1).clamp(
        min=0
    )
    return torch.where(flat_mask, src_flat[idx], dest.reshape(-1)).reshape(dest.shape)


class TestPadScatterSource(CustomTestCase):
    def test_padding_never_changes_the_scatter_result(self):
        """The soundness argument for padding, checked on both implementations.

        If a future change made the tail reachable -- say by padding to fewer
        rows than mask.sum(), or by a kernel that consumed source positionally
        past the mask -- the padded and unpadded results would diverge here.
        """
        torch.manual_seed(0)
        for num_mm_tokens in (1, 3, 7, BUCKET):
            with self.subTest(num_mm_tokens=num_mm_tokens):
                dest = torch.randn(BUCKET, HIDDEN)
                mask = torch.zeros(BUCKET, 1, dtype=torch.bool)
                mask[torch.randperm(BUCKET)[:num_mm_tokens]] = True
                src = torch.randn(num_mm_tokens, HIDDEN)
                padded = _pad_scatter_source(src, BUCKET)

                self.assertEqual(padded.shape, (BUCKET, HIDDEN))

                native = dest.clone()
                native.masked_scatter_(mask.expand_as(native), src)
                native_padded = dest.clone()
                native_padded.masked_scatter_(mask.expand_as(native_padded), padded)
                self.assertTrue(torch.equal(native, native_padded))

                tpu = _tpu_masked_scatter(dest, mask.expand_as(dest), src)
                tpu_padded = _tpu_masked_scatter(dest, mask.expand_as(dest), padded)
                self.assertTrue(torch.equal(tpu, tpu_padded))
                self.assertTrue(torch.equal(tpu, native))

    def test_source_at_or_above_the_target_is_returned_untouched(self):
        """Guards the no-op branch: padding must never shrink or copy needlessly."""
        exact = torch.ones(BUCKET, HIDDEN)
        self.assertIs(_pad_scatter_source(exact, BUCKET), exact)

        longer = torch.ones(BUCKET + 5, HIDDEN)
        self.assertIs(_pad_scatter_source(longer, BUCKET), longer)

    def test_pad_rows_are_zero_and_prefix_is_preserved(self):
        src = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)

        padded = _pad_scatter_source(src, BUCKET)

        self.assertTrue(torch.equal(padded[:3], src))
        self.assertTrue(torch.equal(padded[3:], torch.zeros(BUCKET - 3, HIDDEN)))


if __name__ == "__main__":
    unittest.main()
