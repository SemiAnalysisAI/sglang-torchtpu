# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.input_logprobs import _copy_top_logprobs, get_top_logprobs_chunk

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestInputTopLogprobs(unittest.TestCase):
    def test_chunks_and_width_buckets_preserve_values_and_stable_ties(self):
        for rows, k in ((0, 1), (1, 0), (127, 1), (129, 3), (511, 5), (512, 17)):
            with self.subTest(rows=rows, k=k):
                logits = torch.arange(rows * 17).reshape(rows, 17).remainder(7).float()
                expected_values, expected_indices = torch.sort(
                    logits, descending=True, stable=True
                )
                values, indices = _copy_top_logprobs(logits, k, None, None)
                self.assertEqual(values, expected_values[:, :k].tolist())
                self.assertEqual(indices, expected_indices[:, :k].tolist())

    def test_normalization_and_split_sequence_assembly(self):
        logits = torch.arange(129 * 17).reshape(129, 17).remainder(7).float()
        row_max = logits.max(dim=1).values
        row_log_sum = torch.logsumexp(logits - row_max[:, None], dim=1)
        values, indices = torch.sort(logits, descending=True, stable=True)
        normalized = (values - row_max[:, None]) - row_log_sum[:, None]
        for normalizer in (None, (row_max, row_log_sum)):
            for precomputed in (None, (values[:, :5], indices[:, :5])):
                with self.subTest(
                    normalized=normalizer is not None,
                    precomputed=precomputed is not None,
                ):
                    actual_values, actual_indices = _copy_top_logprobs(
                        logits, 5, normalizer, precomputed
                    )
                    expected = normalized if normalizer is not None else values
                    self.assertEqual(actual_values, expected[:, :5].tolist())
                    self.assertEqual(actual_indices, indices[:, :5].tolist())
        actual_values, actual_indices = [[[-99.0]]], [[[16]]]
        split = get_top_logprobs_chunk(
            logits, [3, 5], [130, 10], actual_values, actual_indices, 1
        )
        self.assertEqual(split, 0)
        self.assertEqual(actual_values, [[[-99.0], *values[:, :3].tolist()], []])
        self.assertEqual(actual_indices, [[[16], *indices[:, :3].tolist()], []])


if __name__ == "__main__":
    unittest.main()
