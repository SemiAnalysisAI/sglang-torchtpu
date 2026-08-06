# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the batched-RPA stacked decode schedule."""

import unittest

import numpy as np

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    should_use_split_k_run,
    should_use_stacked_rpa,
)
from sglang.srt.layers.attention.pallas.batched_rpa.kernel import (
    combine_stacked_partials,
)
from sglang.srt.layers.attention.pallas.batched_rpa.schedule import (
    split_k_chunk_bounds,
    split_k_run_is_profitable,
    stacked_lane_has_work,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestStackedRpaFinalization(CustomTestCase):
    BATCH_SIZE = 8
    NUM_K_CASES = (1, 8, 9, 15, 16, 17)

    def test_stacked_selection_preserves_large_batch_parallelism(self):
        common = dict(
            decode=True,
            enabled=True,
            has_sinks=False,
            has_sliding_window=False,
        )
        self.assertTrue(should_use_stacked_rpa(num_tokens=1, **common))
        self.assertFalse(should_use_stacked_rpa(num_tokens=2, **common))
        self.assertFalse(should_use_stacked_rpa(num_tokens=8, **common))
        self.assertFalse(
            should_use_stacked_rpa(num_tokens=1, **(common | {"decode": False}))
        )
        self.assertFalse(should_use_split_k_run(num_tokens=1, **common))
        self.assertTrue(should_use_split_k_run(num_tokens=2, **common))
        self.assertTrue(should_use_split_k_run(num_tokens=8, **common))
        self.assertFalse(
            should_use_split_k_run(num_tokens=8, **(common | {"has_sinks": True}))
        )
        self.assertFalse(
            should_use_split_k_run(
                num_tokens=8, **(common | {"has_sliding_window": True})
            )
        )
        self.assertFalse(
            should_use_split_k_run(num_tokens=8, **(common | {"decode": False}))
        )

    def test_lane_validity_keeps_every_lane_with_work(self):
        for num_k in self.NUM_K_CASES:
            actual = [
                bool(stacked_lane_has_work(num_k, self.BATCH_SIZE, lane))
                for lane in range(self.BATCH_SIZE)
            ]
            expected = [
                lane < min(num_k, self.BATCH_SIZE) for lane in range(self.BATCH_SIZE)
            ]
            self.assertEqual(actual, expected, f"num_k={num_k}")

    def test_uneven_split_chunks_leave_cleared_gaps_before_merge(self):
        # 240K cached tokens at bkv=3840 produce 62 full blocks. Six lanes get
        # eight blocks and two get seven, requiring one cleared gap on each of
        # the shorter lanes before the common merge row.
        bounds = [
            tuple(
                int(value)
                for value in split_k_chunk_bounds(0, 62, self.BATCH_SIZE, lane)
            )
            for lane in range(self.BATCH_SIZE)
        ]
        lengths = [end - start for start, end in bounds]
        self.assertEqual(bounds[0], (0, 8))
        self.assertEqual(bounds[-1], (55, 62))
        self.assertEqual(lengths, [8] * 6 + [7] * 2)
        self.assertEqual(
            [bounds[i][1] for i in range(7)],
            [bounds[i + 1][0] for i in range(7)],
        )
        self.assertEqual(
            [max(lengths) - length for length in lengths], [0] * 6 + [1] * 2
        )

    def test_split_planner_requires_long_skewed_profitable_runs(self):
        common = dict(
            max_active_blocks=63,
            comparable_long_count=2,
            run_len=63,
            split_len=62,
            predicted_unsplit=63,
            predicted_split=10,
            batch_size=self.BATCH_SIZE,
            min_long_blocks=18,
            min_benefit=4,
        )
        self.assertTrue(bool(split_k_run_is_profitable(**common)))
        self.assertFalse(
            bool(
                split_k_run_is_profitable(
                    **(common | {"comparable_long_count": self.BATCH_SIZE})
                )
            )
        )
        self.assertFalse(
            bool(
                split_k_run_is_profitable(
                    **(common | {"max_active_blocks": 9, "run_len": 9})
                )
            )
        )
        self.assertFalse(
            bool(split_k_run_is_profitable(**(common | {"predicted_split": 60})))
        )

    def test_partial_merge_matches_global_softmax(self):
        import jax.numpy as jnp

        for num_k in self.NUM_K_CASES:
            scores = np.linspace(-1.5, 2.0, num_k, dtype=np.float32)
            values = np.linspace(0.25, 1.75, num_k, dtype=np.float32)
            m_all = []
            l_all = []
            acc_all = []

            for lane in range(self.BATCH_SIZE):
                lane_scores = scores[lane :: self.BATCH_SIZE]
                lane_values = values[lane :: self.BATCH_SIZE]
                if bool(stacked_lane_has_work(num_k, self.BATCH_SIZE, lane)):
                    lane_m = lane_scores.max()
                    weights = np.exp(lane_scores - lane_m)
                    lane_l = weights.sum()
                    lane_acc = (weights * lane_values).sum()
                else:
                    lane_m, lane_l, lane_acc = -np.inf, 0.0, 0.0
                m_all.append([[[lane_m]]])
                l_all.append([[[lane_l]]])
                acc_all.append([[[lane_acc]]])

            _, merged_l, merged_acc = combine_stacked_partials(
                jnp.asarray(m_all), jnp.asarray(l_all), jnp.asarray(acc_all)
            )
            expected_weights = np.exp(scores - scores.max())
            expected = np.sum(expected_weights * values) / expected_weights.sum()
            actual = float((merged_acc / merged_l).reshape(()))
            self.assertAlmostEqual(
                actual, float(expected), places=5, msg=f"num_k={num_k}"
            )


if __name__ == "__main__":
    unittest.main()
