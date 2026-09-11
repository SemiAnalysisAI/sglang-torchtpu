import unittest

import torch

from sglang.srt.layers.moe.topk import _tpu_grouped_topk_group_mask
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference_group_mask(scores, num_expert_group, topk_group):
    n = scores.shape[0]
    group_scores = scores.view(n, num_expert_group, -1).topk(2, dim=-1)[0].sum(-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    return group_mask


class TestTpuGroupedTopkMask(CustomTestCase):
    def test_matches_reference_on_random_scores(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            # DeepSeek-R1: 256 experts, 8 groups, keep 4. Also 8-of-16 shapes.
            for experts, groups, keep in ((256, 8, 4), (256, 16, 8), (64, 4, 1)):
                scores = torch.randn(64, experts).to(dtype)
                with self.subTest(dtype=dtype, groups=groups, keep=keep):
                    actual = _tpu_grouped_topk_group_mask(scores, groups, keep)
                    expected = _reference_group_mask(scores, groups, keep)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(bool((actual.sum(-1) == keep).all()))
                    # bf16 rounding ties group scores; topk breaks ties
                    # arbitrarily, the TPU mask by lower index. Compare the
                    # kept scores, which must agree either way.
                    gs = scores.view(64, groups, -1).topk(2, dim=-1)[0].sum(-1)
                    kept = lambda m: torch.where(m.bool(), gs, -torch.inf).sort(-1)[0]
                    torch.testing.assert_close(
                        kept(actual), kept(expected), rtol=0, atol=0
                    )
                    if dtype is torch.float32:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tie_at_group_max_counts_twice(self):
        # Group 0 has two equal maxima, so its top-2 sum is 2*m, like topk(2).
        scores = torch.zeros(1, 16)
        scores[0, 0] = scores[0, 1] = 5.0  # group 0: 5 + 5 = 10
        scores[0, 4] = 9.0  # group 1: 9 + 0 = 9
        actual = _tpu_grouped_topk_group_mask(scores, 4, 1)
        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    def test_equal_group_scores_keep_lower_index_and_exact_count(self):
        scores = torch.zeros(2, 384)
        actual = _tpu_grouped_topk_group_mask(scores, 8, 2)
        expected = torch.zeros(2, 8)
        expected[:, :2] = 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_groups_selected_ignores_scores(self):
        for groups in (1, 2, 8):
            for dtype in (torch.bfloat16, torch.float32):
                scores = torch.randn(3, 384, dtype=dtype)
                scores[0, :3] = torch.tensor(
                    [float("nan"), float("inf"), float("-inf")], dtype=dtype
                )
                scores[2].fill_(float("-inf"))
                actual = _tpu_grouped_topk_group_mask(scores, groups, groups)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual, torch.ones(3, groups, dtype=dtype), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
