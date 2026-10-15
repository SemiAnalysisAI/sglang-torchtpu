"""The SWA member of a hybrid pool and a draft pool never stage raiden
retraction backups.

The SWA payload is row-filtered on load and a staged handle restores whole
pages, so it takes the row path; the full member stages. The staging budget is
sized for the target pool, so a draft pool would book a second one. Runs on CPU
with no TPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool


def _hybrid_pool() -> SWAKVPool:
    return SWAKVPool(
        size=32,
        size_swa=32,
        page_size=4,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=128,
        swa_attention_layer_ids=[0, 2],
        full_attention_layer_ids=[1, 3],
        device="cpu",
        token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
    )


class TestRetractionBackupEligibility(CustomTestCase):
    def test_swa_member_never_stages(self):
        pool = _hybrid_pool()
        self.assertEqual(pool.swa_kv_pool.allocation_label, "SWA")
        with mock.patch.dict(os.environ, {"SGLANG_RAIDEN_RETRACT_HOST_GB": "1"}):
            self.assertIsNone(pool.swa_kv_pool._raiden_backup())

    def test_draft_pool_never_stages(self):
        draft = _hybrid_pool().full_kv_pool
        allocator = TokenToKVPoolAllocator(
            32, torch.bfloat16, "cpu", draft, need_sort=False
        )
        allocator.register_draft_kv_pools((draft,))
        with mock.patch.dict(os.environ, {"SGLANG_RAIDEN_RETRACT_HOST_GB": "1"}):
            self.assertIsNone(draft._raiden_backup())


if __name__ == "__main__":
    unittest.main()
