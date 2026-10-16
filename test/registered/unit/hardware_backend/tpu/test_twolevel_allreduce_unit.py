"""CPU-side logic of the two-level Pallas all-reduce and the K3 merged MoE reduce.

The kernel itself runs on TPU in test_twolevel_allreduce.py; these cover the
host split, the eligibility gate, the fallbacks to the original collective and
the default (concatenating) merged_moe_all_reduce.
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import twolevel_allreduce as tl
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHostSplit(CustomTestCase):
    def test_derived_from_nnodes(self):
        self.assertEqual(tl.local_size(32, 4), 8)
        self.assertEqual(tl.local_size(8, 1), 8)
        self.assertEqual(tl.local_size(16, 2), 8)

    def test_no_split_when_ranks_do_not_divide(self):
        self.assertEqual(tl.local_size(30, 4), 0)
        self.assertEqual(tl.local_size(32, 0), 0)

    def test_override(self):
        with tpu_envs.SGLANG_TPU_TWOLEVEL_ALLREDUCE_LOCAL_SIZE.override(4):
            self.assertEqual(tl.local_size(8, 1), 4)
        with tpu_envs.SGLANG_TPU_TWOLEVEL_ALLREDUCE_LOCAL_SIZE.override(3):
            self.assertEqual(tl.local_size(8, 1), 0)


class TestEligibility(CustomTestCase):
    def test_single_token_bf16_lane_aligned(self):
        self.assertTrue(tl.eligible(torch.zeros(1, 7168, dtype=torch.bfloat16)))

    def test_rejects_others(self):
        for t in (
            torch.zeros(4, 7168, dtype=torch.bfloat16),  # more than one token
            torch.zeros(1, 7168, dtype=torch.float32),
            torch.zeros(1, 7100, dtype=torch.bfloat16),  # not a 128 multiple
            torch.zeros(7168, dtype=torch.bfloat16),
            torch.zeros(1, tl.MAX_ELEMS + 128, dtype=torch.bfloat16),
        ):
            with self.subTest(shape=tuple(t.shape), dtype=t.dtype):
                self.assertFalse(tl.eligible(t))


class TestFallbacks(CustomTestCase):
    def test_all_reduce_calls_original_when_not_registered(self):
        original = mock.Mock(return_value="orig")
        group = mock.Mock(world_size=32)
        x = torch.zeros(1, 7168, dtype=torch.bfloat16)
        with mock.patch.object(tl, "_READY", False):
            self.assertEqual(tl.all_reduce_around(original, group, x), "orig")
        original.assert_called_once_with(group, x)

    def test_merged_calls_original_when_not_registered(self):
        original = mock.Mock(return_value=("a", "b"))
        latent = torch.zeros(1, 3584, dtype=torch.bfloat16)
        shared = torch.zeros(1, 7168, dtype=torch.bfloat16)
        with mock.patch.object(tl, "_READY", False):
            self.assertEqual(
                tl.merged_moe_all_reduce_around(original, latent, shared), ("a", "b")
            )

    def test_registration_off_by_default(self):
        with mock.patch.object(tl, "register") as reg:
            tl.register_from_parallel_state()
        reg.assert_not_called()


class TestMergedMoeAllReduceDefault(CustomTestCase):
    def test_concat_reduce_split(self):
        from sglang.srt.models import kimi_k3

        latent = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
        shared = torch.arange(10, 16, dtype=torch.float32).reshape(1, 6)
        with mock.patch.object(
            kimi_k3, "tensor_model_parallel_all_reduce", side_effect=lambda t: t * 2
        ) as ar:
            got_latent, got_shared = kimi_k3.merged_moe_all_reduce(latent, shared)
        ar.assert_called_once()  # one collective for both
        torch.testing.assert_close(got_latent, latent * 2)
        torch.testing.assert_close(got_shared, (shared * 2).to(torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
