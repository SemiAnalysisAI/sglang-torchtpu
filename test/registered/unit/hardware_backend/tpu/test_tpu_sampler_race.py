import unittest

import torch

from sglang.srt.hardware_backend.tpu.sampler import sample_exponential_race
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")


class TestExponentialRaceSampler(unittest.TestCase):
    def test_matches_the_target_distribution(self):
        torch.manual_seed(0)
        p = torch.tensor([0.5, 0.3, 0.15, 0.05, 0.0])
        n = 40000
        draws = sample_exponential_race(p.expand(n, -1).contiguous())
        self.assertEqual(draws.dtype, torch.int32)
        freq = torch.bincount(draws.long(), minlength=5).float() / n
        # 3-sigma band for n=40000 is about +-0.0075 on the largest entry
        torch.testing.assert_close(freq[:4], p[:4], atol=0.012, rtol=0)
        self.assertEqual(
            int(freq[4] * n), 0
        )  # zero-probability entries are never drawn

    def test_rows_are_independent_and_shape_preserving(self):
        torch.manual_seed(1)
        p = torch.zeros(6, 10)
        p[torch.arange(6), torch.arange(6)] = 1.0  # one-hot rows
        draws = sample_exponential_race(p)
        self.assertEqual(tuple(draws.shape), (6,))
        self.assertEqual(draws.tolist(), list(range(6)))

    def test_all_zero_row_does_not_produce_nan(self):
        draws = sample_exponential_race(torch.zeros(2, 8))
        self.assertEqual(tuple(draws.shape), (2,))
        self.assertTrue(bool(((draws >= 0) & (draws < 8)).all()))


if __name__ == "__main__":
    unittest.main()
