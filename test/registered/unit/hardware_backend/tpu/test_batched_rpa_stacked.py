# SPDX-License-Identifier: Apache-2.0

"""Unit tests for selecting lane-packed batched-RPA decode exports."""

import unittest

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    should_use_split_k_run,
    should_use_stacked_rpa,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestStackedRpaSelection(CustomTestCase):
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


if __name__ == "__main__":
    unittest.main()
