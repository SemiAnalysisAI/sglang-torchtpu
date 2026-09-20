# SPDX-License-Identifier: Apache-2.0
"""Run table and CPU fallback of the in-place K/V row-run write."""

import unittest

import numpy as np
import torch

from sglang.srt.hardware_backend.tpu.kv_row_write import (
    kv_write_segments,
    write_rows_torch,
    write_run_bound,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKvWriteSegments(unittest.TestCase):
    def test_runs_break_at_slot_gaps_and_padding(self):
        # Slots 5-8 are one run; 9 skipped; 10-11; padding rows; 30 alone.
        slots = np.array([5, 6, 7, 8, 10, 11, 0, 0, 30])
        valid = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1], dtype=bool)
        segments, count = kv_write_segments(slots, valid, max_segments=8)
        self.assertEqual(count, 3)
        self.assertEqual(segments.shape, (3, 8))
        self.assertEqual(
            segments[:, :3].T.tolist(), [[0, 5, 4], [4, 10, 2], [8, 30, 1]]
        )
        self.assertEqual(int(segments[:, 3:].abs().sum()), 0)

    def test_padding_between_consecutive_slots_splits_the_run(self):
        slots = np.array([5, 6, 7, 8])
        valid = np.array([1, 1, 0, 1], dtype=bool)
        segments, count = kv_write_segments(slots, valid)
        self.assertEqual(count, 2)
        self.assertEqual(segments[:, :2].T.tolist(), [[0, 5, 2], [3, 8, 1]])

    def test_long_runs_split_into_staging_pieces(self):
        slots = np.arange(100, 100 + 3000)
        segments, count = kv_write_segments(slots, np.ones(3000, dtype=bool))
        self.assertEqual(count, 3)
        self.assertEqual(
            segments[:, :3].T.tolist(),
            [[0, 100, 1024], [1024, 1124, 1024], [2048, 2148, 952]],
        )

    def test_no_valid_rows_gives_no_runs(self):
        segments, count = kv_write_segments(
            np.zeros(4, dtype=np.int64), np.zeros(4, dtype=bool)
        )
        self.assertEqual(count, 0)
        self.assertEqual(int(segments.abs().sum()), 0)

    def test_run_bound_covers_pages_block_edges_and_pieces(self):
        # 4096 rows on 128-token pages in 16 blocks: 32 pages, 32 block edges,
        # 4 staging pieces, plus one.
        self.assertEqual(write_run_bound(4096, page_size=128, blocks=16), 69)

    def test_too_many_runs_is_rejected(self):
        slots = np.arange(0, 20, 2)
        with self.assertRaises(ValueError):
            kv_write_segments(slots, np.ones(10, dtype=bool), max_segments=4)


class TestWriteRowsFallback(unittest.TestCase):
    def test_cpu_write_matches_scatter(self):
        torch.manual_seed(0)
        kv = torch.zeros(6, 4, 2, 2, 8, dtype=torch.bfloat16)
        rows = torch.randn(10, 2, 2, 8).to(torch.bfloat16)
        slots = np.array([9, 10, 11, 20, 21, 0, 0, 3, 4, 5])
        valid = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
        segments, count = kv_write_segments(slots, valid)
        write_rows_torch(torch.tensor([count], dtype=torch.int32), segments, rows, kv)
        expected = torch.zeros_like(kv).flatten(0, 1)
        expected[torch.from_numpy(slots[valid])] = rows[torch.from_numpy(valid)]
        self.assertTrue(torch.equal(kv.flatten(0, 1), expected))


if __name__ == "__main__":
    unittest.main()
