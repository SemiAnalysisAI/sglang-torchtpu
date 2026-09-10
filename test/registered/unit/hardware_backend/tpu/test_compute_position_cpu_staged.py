"""The CPU-staged compute_position matches upstream and prefers host lists."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu import patches
from sglang.srt.model_executor import forward_batch_info
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference(prefix, seq):
    return torch.cat([torch.arange(p, p + s) for p, s in zip(prefix, seq)]).to(
        torch.int64
    )


class TestComputePositionCpuStaged(unittest.TestCase):
    def setUp(self):
        original = forward_batch_info.compute_position_torch
        self.addCleanup(setattr, forward_batch_info, "compute_position_torch", original)
        patches._compute_position_torch_patched = False
        self.addCleanup(setattr, patches, "_compute_position_torch_patched", False)
        patches._patch_compute_position_torch()
        self.fn = forward_batch_info.compute_position_torch

    def test_matches_reference_from_host_lists(self):
        prefix, seq = [0, 5, 100, 7], [3, 1, 8, 2]
        pos, start = self.fn(
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(seq, dtype=torch.int32),
            prefix,
            seq,
        )
        self.assertTrue(torch.equal(pos, _reference(prefix, seq)))
        self.assertTrue(
            torch.equal(start, torch.tensor([0, 3, 4, 12], dtype=torch.int32))
        )

    def test_falls_back_to_tensors_without_host_lists(self):
        prefix, seq = [2, 0], [4, 6]
        pos, _ = self.fn(
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(seq, dtype=torch.int32),
        )
        self.assertTrue(torch.equal(pos, _reference(prefix, seq)))

    def test_host_lists_are_used_without_reading_the_tensors(self):
        prefix_t = torch.tensor([9, 9], dtype=torch.int32)  # deliberately wrong
        seq_t = torch.tensor([9, 9], dtype=torch.int32)
        with patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("device read")
        ):
            pos, _ = self.fn(prefix_t, seq_t, [0, 4], [2, 2])
        self.assertTrue(torch.equal(pos, torch.tensor([0, 1, 4, 5])))

    def test_empty_batch(self):
        pos, start = self.fn(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            [],
            [],
        )
        self.assertEqual(pos.numel(), 0)
        self.assertEqual(start.numel(), 0)


if __name__ == "__main__":
    unittest.main()
