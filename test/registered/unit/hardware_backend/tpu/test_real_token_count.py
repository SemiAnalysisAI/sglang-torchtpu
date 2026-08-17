"""Unit tests for the TPU real-token-count channel and the MoE padded-row skip.

Two halves of one contract:

  1. ``hardware_backend.tpu.real_token_count`` publishes the forward's real token
     count across the ``torch.compile`` boundary. ``TestRealTokenCount`` pins the
     properties that keep it from forcing a serving-time recompile: the buffer's
     object identity is stable across steps, and the value is updated in place.
  2. ``fp8_moe._retire_dead_moe_rows`` turns that count into an expert
     assignment. ``TestRetireDeadMoeRows`` pins which rows get retired and that
     they land on an out-of-range sentinel, not on expert 0.

CPU tensors and pure logic only -- no TPU. The kernel-side numerics (that a
sentinel row really does join no group) live in
``test/registered/tpu/test_pallas_fused_moe.py``.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.hardware_backend.tpu.fp8_moe import _retire_dead_moe_rows
from sglang.srt.hardware_backend.tpu.real_token_count import (
    publish_real_token_count,
    real_token_count,
    reset_real_token_count,
)
from sglang.test.test_utils import CustomTestCase

_CPU = torch.device("cpu")


class TestRealTokenCount(CustomTestCase):
    def setUp(self):
        reset_real_token_count()

    def tearDown(self):
        reset_real_token_count()

    def test_unpublished_reads_none(self):
        self.assertIsNone(real_token_count())

    def test_publish_then_read(self):
        publish_real_token_count(device=_CPU, real_q=3328, enabled=True)
        count = real_token_count()
        self.assertEqual(count.tolist(), [3328])
        self.assertEqual(count.dtype, torch.int32)

    def test_buffer_identity_is_stable_across_steps(self):
        """The whole point of the module: a new tensor per step would recompile."""
        publish_real_token_count(device=_CPU, real_q=3328, enabled=True)
        first = real_token_count()
        for real_q in (3072, 2816, 4096):
            publish_real_token_count(device=_CPU, real_q=real_q, enabled=True)
            self.assertIs(real_token_count(), first)
            self.assertEqual(first.tolist(), [real_q])

    def test_disabled_reads_none_and_keeps_buffer(self):
        publish_real_token_count(device=_CPU, real_q=3328, enabled=True)
        buffer = real_token_count()
        publish_real_token_count(device=_CPU, real_q=64, enabled=False)
        self.assertIsNone(real_token_count())
        # Re-enabling must reuse the same buffer, so an extend step that follows
        # decode steps does not hand the graph a new tensor object.
        publish_real_token_count(device=_CPU, real_q=2816, enabled=True)
        self.assertIs(real_token_count(), buffer)
        self.assertEqual(buffer.tolist(), [2816])

    def test_disabled_does_not_allocate(self):
        publish_real_token_count(device=_CPU, real_q=64, enabled=False)
        self.assertIsNone(real_token_count())

    def test_reset_drops_the_buffer(self):
        publish_real_token_count(device=_CPU, real_q=3328, enabled=True)
        reset_real_token_count()
        self.assertIsNone(real_token_count())


class TestRetireDeadMoeRows(CustomTestCase):
    """``_retire_dead_moe_rows`` over a 4-row batch, topk 2, 8 local experts."""

    SENTINEL = 8

    def _rows(self):
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32)
        topk_weights = torch.ones(4, 2, dtype=torch.float32)
        return topk_ids, topk_weights

    def _retire(self, topk_ids, topk_weights, real_tokens):
        return _retire_dead_moe_rows(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            sentinel_expert_id=self.SENTINEL,
            real_tokens=real_tokens,
        )

    def test_no_count_keeps_every_row(self):
        topk_ids, topk_weights = self._rows()
        ids, weights = self._retire(topk_ids, topk_weights, None)
        self.assertEqual(ids.tolist(), topk_ids.tolist())
        self.assertEqual(weights.tolist(), topk_weights.tolist())

    def test_padded_rows_go_to_the_sentinel(self):
        topk_ids, topk_weights = self._rows()
        ids, weights = self._retire(
            topk_ids, topk_weights, torch.tensor([2], dtype=torch.int32)
        )
        self.assertEqual(ids.tolist(), [[0, 1], [2, 3], [8, 8], [8, 8]])
        self.assertEqual(
            weights.tolist(), [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        )

    def test_sentinel_is_out_of_range_for_the_group_one_hot(self):
        """The retired id must not name a real expert, or its group does the work.

        ``_fused_moe_jax`` builds ``group_sizes`` as
        ``one_hot(topk_ids, route_num_experts).sum(0)``; an id equal to that width
        one-hots to all zeros, which is what keeps the grouped matmul away.
        """
        topk_ids, topk_weights = self._rows()
        ids, _ = self._retire(
            topk_ids, topk_weights, torch.tensor([2], dtype=torch.int32)
        )
        group_sizes = torch.nn.functional.one_hot(
            ids.reshape(-1).long(), num_classes=self.SENTINEL + 1
        ).sum(0)
        # Four retired slots, all outside the real expert range.
        self.assertEqual(
            group_sizes[: self.SENTINEL].tolist(), [1, 1, 1, 1, 0, 0, 0, 0]
        )
        self.assertEqual(int(group_sizes[self.SENTINEL]), 4)

    def test_negative_ids_are_retired_too(self):
        """The router's own -1 marker keeps its old meaning, new destination."""
        topk_ids, topk_weights = self._rows()
        topk_ids[1] = -1
        ids, weights = self._retire(topk_ids, topk_weights, None)
        self.assertEqual(ids[1].tolist(), [self.SENTINEL, self.SENTINEL])
        self.assertEqual(weights[1].tolist(), [0.0, 0.0])

    def test_full_batch_retires_nothing(self):
        topk_ids, topk_weights = self._rows()
        ids, weights = self._retire(
            topk_ids, topk_weights, torch.tensor([4], dtype=torch.int32)
        )
        self.assertEqual(ids.tolist(), topk_ids.tolist())
        self.assertEqual(weights.tolist(), topk_weights.tolist())

    def test_count_is_read_as_a_tensor_not_a_python_int(self):
        """Baking the count into the graph as an int is what forces recompiles."""
        topk_ids, topk_weights = self._rows()
        real_tokens = torch.tensor([4], dtype=torch.int32)
        ids, _ = self._retire(topk_ids, topk_weights, real_tokens)
        self.assertEqual(ids.tolist(), topk_ids.tolist())
        # Same tensor object, new value -- as publish_real_token_count does it.
        real_tokens.fill_(1)
        ids, weights = self._retire(topk_ids, topk_weights, real_tokens)
        self.assertEqual(ids[1:].tolist(), [[8, 8], [8, 8], [8, 8]])
        self.assertEqual(weights[0].tolist(), [1.0, 1.0])

    def test_dtypes_are_preserved(self):
        topk_ids, topk_weights = self._rows()
        topk_weights = topk_weights.to(torch.bfloat16)
        ids, weights = self._retire(
            topk_ids, topk_weights, torch.tensor([2], dtype=torch.int32)
        )
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(weights.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
