"""Unit tests for SWATokenToKVPoolAllocator.free_swa (TPU device-alignment).

CPU-runnable: free_swa only touches ``full_to_swa_index_mapping`` (kept
host-resident on TPU, ``mapping_device="cpu"``) and ``swa_attn_allocator.free``,
so a tiny host-side harness exercises the exact freeing/zeroing logic without a
TPU or a full KV pool.
"""

import unittest

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _RecordingAllocator:
    """Captures the indices handed to free() so the test can assert on them."""

    def __init__(self):
        self.freed = None

    def free(self, indices):
        self.freed = indices


class _FreeSwaHarness:
    """Minimal stand-in exposing exactly what free_swa reads/writes, with a
    host-resident (CPU) full_to_swa_index_mapping, as TPU keeps it."""

    free_swa = SWATokenToKVPoolAllocator.free_swa

    def __init__(self, mapping):
        self.full_to_swa_index_mapping = mapping
        self.swa_attn_allocator = _RecordingAllocator()


class TestFreeSwa(CustomTestCase):
    @staticmethod
    def _mapping():
        # full index -> swa slot; 0 == unmapped (no SWA slot), >0 == real slot.
        return torch.tensor([0, 5, 0, 7, 9, 0], dtype=torch.int64)

    def test_frees_mapped_slots_and_zeros_mapping(self):
        harness = _FreeSwaHarness(self._mapping())
        free_index = torch.tensor([1, 3, 4], dtype=torch.int64)  # -> slots 5,7,9
        harness.free_swa(free_index)
        self.assertEqual(harness.swa_attn_allocator.freed.tolist(), [5, 7, 9])
        # freed rows zeroed; the untouched rows keep their mapping.
        self.assertEqual(
            harness.full_to_swa_index_mapping.tolist(), [0, 0, 0, 0, 0, 0]
        )

    def test_filters_unmapped_sentinel(self):
        harness = _FreeSwaHarness(self._mapping())
        free_index = torch.tensor([0, 1, 2], dtype=torch.int64)  # slots 0,5,0
        harness.free_swa(free_index)
        # only the >0 slot (5) is freed; the 0 sentinels are filtered out.
        self.assertEqual(harness.swa_attn_allocator.freed.tolist(), [5])
        # rows 0,1,2 are zeroed regardless of whether they had a slot.
        self.assertEqual(
            harness.full_to_swa_index_mapping.tolist(), [0, 0, 0, 7, 9, 0]
        )

    def test_no_double_free_after_zeroing(self):
        harness = _FreeSwaHarness(self._mapping())
        idx = torch.tensor([1, 3], dtype=torch.int64)  # slots 5,7
        harness.free_swa(idx)
        self.assertEqual(harness.swa_attn_allocator.freed.tolist(), [5, 7])
        # freeing the same full indices again finds them zeroed -> nothing freed.
        harness.free_swa(idx)
        self.assertEqual(harness.swa_attn_allocator.freed.tolist(), [])


if __name__ == "__main__":
    unittest.main()
