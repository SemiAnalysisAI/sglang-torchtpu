"""Correctness tests for the TPU fixed-capacity free-list allocator.

`TpuPagedTokenToKVPoolAllocator` (need_sort=False) keeps `free_pages` at a
constant shape `[num_pages]` with the valid free pages compacted at the front
`[0:_num_free]`, using `torch.roll` + a fixed-size front write for alloc/free
(fixed XLA shapes on TPU). These tests run on CPU and prove the fixed-shape logic
preserves exact allocator semantics:
  * the valid free-page list (order included) matches a simple deque reference
    that pops the front on alloc and prepends sorted-unique pages on free
    (exactly the base `need_sort=False` behavior);
  * out_indices, conservation (no double-alloc, no leak), backup/restore, grouped
    free, out-of-space, and clear all behave.
"""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang.test.test_utils import CustomTestCase


def _make(num_pages, page_size):
    return TpuPagedTokenToKVPoolAllocator(
        size=num_pages * page_size,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


def _front(a):
    return a.free_pages[: a._num_free].tolist()


class TestFixedFreeList(CustomTestCase):
    def test_alloc_free_matches_deque_reference(self):
        for page_size in (1, 4):
            a = _make(num_pages=20, page_size=page_size)
            ref = list(range(1, 21))  # free pages, front = index 0 (matches clear())
            self.assertEqual(_front(a), ref)
            torch.manual_seed(0)
            allocated_pages = []
            for _ in range(200):
                if ref and (len(allocated_pages) == 0 or torch.rand(1).item() < 0.5):
                    # alloc k pages
                    k = int(torch.randint(1, min(4, len(ref)) + 1, (1,)).item())
                    oi = a.alloc(k * page_size)
                    exp_pages = ref[:k]
                    ref = ref[k:]
                    self.assertEqual(_front(a), ref)
                    exp_idx = [
                        p * page_size + off
                        for p in exp_pages
                        for off in range(page_size)
                    ]
                    self.assertEqual(oi.tolist(), exp_idx)
                    allocated_pages.extend(exp_pages)
                elif allocated_pages:
                    # free k allocated pages
                    k = int(torch.randint(1, len(allocated_pages) + 1, (1,)).item())
                    pages = allocated_pages[:k]
                    allocated_pages = allocated_pages[k:]
                    idx = torch.tensor(
                        [p * page_size for p in pages], dtype=torch.int64
                    )
                    a.free(idx)
                    # base need_sort=False: prepend torch.unique (sorted) pages
                    ref = sorted(set(pages)) + ref
                    self.assertEqual(_front(a), ref)
                # conservation: free + allocated == all pages
                self.assertEqual(
                    sorted(_front(a) + allocated_pages), list(range(1, 21))
                )
                # no duplicates in the free list
                self.assertEqual(len(set(_front(a))), len(_front(a)))

    def test_alloc_extend_consumes_front_and_conserves(self):
        # page_size=1 (spec config): each new token consumes one front page.
        a = _make(num_pages=64, page_size=1)
        ref = list(range(1, 65))
        bs = 1
        prefix = torch.tensor([0], dtype=torch.int64)
        seq = torch.tensor([5], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64)
        oi = a.alloc_extend(
            prefix, prefix.clone(), seq, seq.clone(), last_loc, extend_num_tokens=5
        )
        self.assertIsNotNone(oi)
        # 5 new pages consumed from the front
        self.assertEqual(_front(a), ref[5:])
        self.assertEqual(a._num_free, 64 - 5)
        # out_indices are exactly the 5 front pages (page_size=1 -> idx == page)
        self.assertEqual(sorted(oi.tolist()), ref[:5])
        # free them back -> conservation
        a.free(oi.to(torch.int64))
        self.assertEqual(sorted(_front(a)), list(range(1, 65)))

    def test_out_of_space_returns_none(self):
        a = _make(num_pages=4, page_size=1)
        self.assertIsNone(a.alloc(5))  # more than available
        oi = a.alloc(4)
        self.assertEqual(sorted(oi.tolist()), [1, 2, 3, 4])
        self.assertEqual(a.available_size(), 0)
        self.assertIsNone(a.alloc(1))  # empty

    def test_grouped_free(self):
        a = _make(num_pages=10, page_size=1)
        oi = a.alloc(6)  # pages 1..6
        a.free_group_begin()
        a.free(torch.tensor([1, 2], dtype=torch.int64))
        a.free(torch.tensor([3], dtype=torch.int64))
        # nothing freed yet (grouped)
        self.assertEqual(a._num_free, 4)
        a.free_group_end()  # flushes [1,2,3]
        self.assertEqual(a._num_free, 7)
        self.assertEqual(sorted(_front(a)), [1, 2, 3, 7, 8, 9, 10])

    def test_backup_restore_is_independent(self):
        a = _make(num_pages=10, page_size=1)
        a.alloc(3)
        st = a.backup_state()
        a.alloc(2)
        a.free(torch.tensor([1, 2], dtype=torch.int64))
        a.restore_state(st)
        self.assertEqual(a._num_free, 7)
        self.assertEqual(_front(a), list(range(4, 11)))

    def test_clear_resets(self):
        a = _make(num_pages=8, page_size=1)
        a.alloc(5)
        a.clear()
        self.assertEqual(a._num_free, 8)
        self.assertEqual(_front(a), list(range(1, 9)))
        self.assertEqual(a.available_size(), 8)


if __name__ == "__main__":
    unittest.main()
