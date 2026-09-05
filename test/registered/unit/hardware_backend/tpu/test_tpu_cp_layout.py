"""Unit tests for the TPU prefill context-parallel token layout (numpy only)."""

import unittest

import numpy as np

from sglang.srt.hardware_backend.tpu.cp_layout import (
    build_tpu_cp_layout,
    local_row_count,
    max_local_rows,
    owned_blocks,
    warmup_request_lengths,
    zigzag_block_sizes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestZigzagBlocks(unittest.TestCase):
    def test_block_sizes_sum_to_length_and_front_load_the_remainder(self):
        self.assertEqual(zigzag_block_sizes(35, 4), [5, 5, 5, 4, 4, 4, 4, 4])
        self.assertEqual(zigzag_block_sizes(64, 8), [4] * 16)
        for length in (16, 17, 100, 4095):
            self.assertEqual(sum(zigzag_block_sizes(length, 8)), length)

    def test_owned_blocks_pair_early_and_late(self):
        self.assertEqual(owned_blocks(8, 0), (0, 15))
        self.assertEqual(owned_blocks(8, 3), (3, 12))
        self.assertEqual(owned_blocks(4, 3), (3, 4))

    def test_local_rows_partition_the_tokens_across_ranks(self):
        extend_lens = [40, 24, 17]
        total = sum(local_row_count(extend_lens, 4, rank) for rank in range(4))
        self.assertEqual(total, sum(extend_lens))
        self.assertEqual(
            max_local_rows(extend_lens, 4),
            max(local_row_count(extend_lens, 4, rank) for rank in range(4)),
        )

    def test_local_rows_are_equal_when_lengths_divide_evenly(self):
        # 64 -> 4 tokens per block, 128 -> 8: every rank holds 2 blocks of each.
        extend_lens = [64, 128]
        counts = {local_row_count(extend_lens, 8, rank) for rank in range(8)}
        self.assertEqual(counts, {24})


class TestBuildLayout(unittest.TestCase):
    CP = 4
    EXTEND = [40, 24]
    PREFIX = [0, 100]
    PAD_B = 4
    BUCKET = 32

    def _layout(self, rank):
        return build_tpu_cp_layout(
            extend_lens=self.EXTEND,
            prefix_lens=self.PREFIX,
            cp_size=self.CP,
            cp_rank=rank,
            pad_b=self.PAD_B,
            local_rows_padded=self.BUCKET,
        )

    def test_every_token_is_owned_by_exactly_one_rank(self):
        seen = np.zeros(sum(self.EXTEND), dtype=np.int64)
        for rank in range(self.CP):
            layout = self._layout(rank)
            real = layout.local_rows[: layout.local_len]
            self.assertTrue((layout.local_rows[layout.local_len :] == -1).all())
            np.add.at(seen, real, 1)
        self.assertTrue((seen == 1).all())

    def test_local_order_is_request_major_with_adjacent_blocks(self):
        layout = self._layout(1)
        blocks0 = zigzag_block_sizes(self.EXTEND[0], self.CP)
        starts0 = np.concatenate([[0], np.cumsum(blocks0)])
        early, late = owned_blocks(self.CP, 1)
        expected = np.concatenate(
            [
                np.arange(starts0[early], starts0[early + 1]),
                np.arange(starts0[late], starts0[late + 1]),
            ]
        )
        self.assertEqual(
            layout.local_rows[: expected.shape[0]].tolist(), expected.tolist()
        )

    def test_gathered_rows_are_identical_on_every_rank(self):
        reference = self._layout(0)
        for rank in range(1, self.CP):
            layout = self._layout(rank)
            self.assertTrue((layout.gathered_req == reference.gathered_req).all())
            self.assertTrue((layout.gathered_pos == reference.gathered_pos).all())
        self.assertEqual(reference.gathered_req.shape[0], self.CP * self.BUCKET)
        # Every (request, offset) pair appears once; padding rows are -1 twice.
        valid = reference.gathered_req >= 0
        pairs = set(
            zip(
                reference.gathered_req[valid].tolist(),
                reference.gathered_pos[valid].tolist(),
            )
        )
        self.assertEqual(len(pairs), sum(self.EXTEND))
        self.assertTrue((reference.gathered_pos[~valid] == -1).all())

    def test_pseudo_rows_follow_the_owned_blocks(self):
        for rank in range(self.CP):
            layout = self._layout(rank)
            early, late = owned_blocks(self.CP, rank)
            for i, length in enumerate(self.EXTEND):
                blocks = zigzag_block_sizes(length, self.CP)
                prefix = self.PREFIX[i]
                self.assertEqual(
                    layout.pseudo_kv_lens[2 * i], prefix + sum(blocks[: early + 1])
                )
                self.assertEqual(layout.pseudo_q_lens[2 * i], blocks[early])
                self.assertEqual(
                    layout.pseudo_kv_lens[2 * i + 1], prefix + sum(blocks[: late + 1])
                )
                self.assertEqual(layout.pseudo_q_lens[2 * i + 1], blocks[late])
            # Padded requests are zero rows at the tail.
            self.assertTrue((layout.pseudo_kv_lens[2 * len(self.EXTEND) :] == 0).all())
            self.assertTrue((layout.pseudo_q_lens[2 * len(self.EXTEND) :] == 0).all())
            self.assertEqual(
                int(layout.pseudo_q_lens.sum()),
                local_row_count(self.EXTEND, self.CP, rank),
            )

    def test_rank_zero_owns_every_last_token(self):
        layout = self._layout(0)
        self.assertEqual(layout.last_owner_mask.tolist(), [1, 1, 0, 0])
        for i, length in enumerate(self.EXTEND):
            row = layout.last_rows[i]
            token = layout.local_rows[row]
            self.assertEqual(token, sum(self.EXTEND[: i + 1]) - 1)
        for rank in range(1, self.CP):
            other = self._layout(rank)
            self.assertEqual(other.last_owner_mask.tolist(), [0] * self.PAD_B)
            self.assertEqual(other.last_rows.tolist(), [0] * self.PAD_B)

    def test_rejects_short_requests_and_overfull_buckets(self):
        with self.assertRaisesRegex(ValueError, "below the zigzag minimum"):
            build_tpu_cp_layout(
                extend_lens=[7],
                prefix_lens=[0],
                cp_size=4,
                cp_rank=0,
                pad_b=1,
                local_rows_padded=8,
            )
        with self.assertRaisesRegex(ValueError, "exceed the request bucket"):
            build_tpu_cp_layout(
                extend_lens=[16, 16],
                prefix_lens=[0, 0],
                cp_size=4,
                cp_rank=0,
                pad_b=1,
                local_rows_padded=8,
            )
        with self.assertRaisesRegex(ValueError, "exceed the local token bucket"):
            build_tpu_cp_layout(
                extend_lens=[64],
                prefix_lens=[0],
                cp_size=4,
                cp_rank=0,
                pad_b=1,
                local_rows_padded=8,
            )


class TestWarmupRequestLengths(unittest.TestCase):
    def test_lengths_fill_the_local_bucket_on_every_rank(self):
        for local_rows, cp_size, max_reqs in ((64, 4, 8), (2048, 8, 64), (16, 8, 1)):
            lengths = warmup_request_lengths(
                local_rows=local_rows, cp_size=cp_size, max_reqs=max_reqs
            )
            self.assertLessEqual(len(lengths), max_reqs)
            for rank in range(cp_size):
                self.assertEqual(local_row_count(lengths, cp_size, rank), local_rows)
            for length in lengths:
                self.assertEqual(length % (2 * cp_size), 0)

    def test_rejects_odd_or_empty_buckets(self):
        with self.assertRaises(ValueError):
            warmup_request_lengths(local_rows=3, cp_size=4, max_reqs=1)
        with self.assertRaises(ValueError):
            warmup_request_lengths(local_rows=8, cp_size=4, max_reqs=0)


if __name__ == "__main__":
    unittest.main()
