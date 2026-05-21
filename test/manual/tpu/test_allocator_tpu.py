"""Unit tests for :class:`TpuPagedTokenToKVPoolAllocator`.

These tests guard the CPU-resident free-page bookkeeping rewrite — the
core of this allocator subclass. They run on CPU (no torch_tpu / TPU
hardware needed), because the goal is to validate the host-side data
structures and the stable-shape device formula, not to measure compile
behaviour.

Why this guard matters: the old implementation kept ``free_pages`` as a
device tensor that shrank every page boundary, which made
``alloc_decode``'s output shape a function of pool occupancy and
triggered a PJRT recompile every ``page_size`` decode tokens. The new
implementation keeps ``free_pages`` as a host-side numpy array and
H2D-copies fixed-shape ``(bs,)`` tensors into the device formula. A
future refactor that accidentally re-introduces the device-shaped
input would be silent under unit tests of the parent allocator; these
tests catch it by asserting the exact indices the formula produces.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _make_allocator(size: int = 32, page_size: int = 4):
    """Build a CPU-resident allocator. ``size`` must be a multiple of
    ``page_size``. ``kvcache=None`` is fine — the base class only stores
    the reference and never calls into it from the allocator path."""
    import numpy as np

    from sglang.srt.hardware_backend.tpu.allocator_tpu import (
        TpuPagedTokenToKVPoolAllocator,
    )

    return TpuPagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


class TestAllocatorPoolLifecycle(CustomTestCase):
    """Pool initialisation, accounting, and clear()."""

    def test_pool_starts_full_with_slot_zero_reserved(self):
        a = _make_allocator(size=32, page_size=4)
        # num_pages = 8; slot 0 reserved for the dummy-padding page;
        # so 8 usable pages with IDs 1..8.
        self.assertEqual(a.num_pages, 8)
        self.assertEqual(len(a.free_pages), 8)
        self.assertEqual(int(a.free_pages.min()), 1)
        self.assertEqual(int(a.free_pages.max()), 8)
        # available_size reports usable tokens, not pages.
        self.assertEqual(a.available_size(), 8 * 4)

    def test_clear_restores_full_pool(self):
        a = _make_allocator(size=32, page_size=4)
        # Drain a couple of pages, then clear() must restore IDs 1..8.
        out = a.alloc(8)  # 2 pages worth of tokens
        self.assertIsNotNone(out)
        self.assertEqual(len(a.free_pages), 6)
        a.clear()
        self.assertEqual(len(a.free_pages), 8)
        self.assertEqual(int(a.free_pages.min()), 1)


class TestAllocDecode(CustomTestCase):
    """``alloc_decode`` is the hot path the rewrite optimises. Output
    indices follow the device formula::

        need_new_pages[i] == 0  ->  out_indices[i] = last_loc[i] + 1
        need_new_pages[i] == 1  ->  out_indices[i] = new_page_id * page_size
    """

    def test_no_new_pages_continues_current_page(self):
        a = _make_allocator(size=32, page_size=4)
        # Two requests, both mid-page (seq_lens % 4 != 1). The allocator
        # should NOT consume any free pages — out_indices = last_loc + 1.
        seq_lens = torch.tensor([3, 7], dtype=torch.int32)
        last_loc = torch.tensor([5, 11], dtype=torch.int32)
        free_before = len(a.free_pages)

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        self.assertTrue(torch.equal(out, torch.tensor([6, 12], dtype=torch.int32)))
        self.assertEqual(len(a.free_pages), free_before)

    def test_all_new_pages_assigns_fresh_page_per_req(self):
        a = _make_allocator(size=32, page_size=4)
        # Both reqs at a page boundary (seq_lens % 4 == 1).
        seq_lens = torch.tensor([5, 9], dtype=torch.int32)
        last_loc = torch.tensor([99, 99], dtype=torch.int32)
        first_three_free = list(map(int, a.free_pages[:3]))

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        # Each req gets the NEXT distinct page (in order). Pool was
        # [1, 2, 3, ...] so req 0 -> page 1, req 1 -> page 2.
        expected = torch.tensor(
            [first_three_free[0] * 4, first_three_free[1] * 4],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(out, expected))
        # Consumed exactly two pages; rest stay.
        self.assertEqual(len(a.free_pages), 8 - 2)
        self.assertEqual(int(a.free_pages[0]), first_three_free[2])

    def test_mixed_decode_some_cross_boundary(self):
        a = _make_allocator(size=32, page_size=4)
        # Req 0 mid-page (last_loc+1), req 1 at boundary (fresh page).
        seq_lens = torch.tensor([3, 5], dtype=torch.int32)
        last_loc = torch.tensor([10, 99], dtype=torch.int32)
        first_free = int(a.free_pages[0])

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        self.assertEqual(int(out[0]), 11)  # last_loc[0] + 1, no new page
        self.assertEqual(int(out[1]), first_free * 4)  # new page * page_size
        # Exactly one page consumed.
        self.assertEqual(len(a.free_pages), 7)

    def test_returns_none_when_pool_exhausted(self):
        a = _make_allocator(size=8, page_size=4)  # only 2 usable pages
        # Drain both.
        a.alloc_decode(
            torch.tensor([5, 5], dtype=torch.int32),
            torch.tensor([5, 5], dtype=torch.int32),
            torch.tensor([99, 99], dtype=torch.int32),
        )
        # Now ask for one more page — should return None, not crash.
        out = a.alloc_decode(
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([99], dtype=torch.int32),
        )
        self.assertIsNone(out)


class TestFreeRoundTrip(CustomTestCase):
    """``free`` accepts a device tensor of token indices and returns
    the corresponding page IDs to the pool."""

    def test_free_returns_pages_to_pool(self):
        a = _make_allocator(size=32, page_size=4)
        # Use alloc() to drain two pages (8 contiguous tokens).
        out = a.alloc(8)
        self.assertIsNotNone(out)
        self.assertEqual(len(a.free_pages), 6)

        # Free those 8 tokens; the two pages they came from should
        # re-enter the pool.
        a.free(out)
        self.assertEqual(len(a.free_pages), 8)

    def test_free_dedupes_same_page(self):
        a = _make_allocator(size=32, page_size=4)
        # All tokens belong to page 1 — freeing them should return
        # exactly one page to the pool.
        tok = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
        before = len(a.free_pages)
        a.free(tok)
        self.assertEqual(len(a.free_pages), before + 1)

    def test_free_empty_is_noop(self):
        a = _make_allocator(size=32, page_size=4)
        before = len(a.free_pages)
        a.free(torch.empty(0, dtype=torch.int64))
        self.assertEqual(len(a.free_pages), before)


class TestAllocExtend(CustomTestCase):
    """``alloc_extend`` is the prefill / extend path. Less optimisation-
    critical than ``alloc_decode`` but still has to compute the right
    number of new pages from host tensors."""

    def test_extend_consumes_expected_pages(self):
        a = _make_allocator(size=32, page_size=4)
        # One request, prefix=0 tokens, target seq_lens=5 -> two pages
        # (one full + one partial); extend_num_tokens=5.
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([5], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)

        out = a.alloc_extend(
            prefix_lens, prefix_lens.clone(),
            seq_lens, seq_lens.clone(),
            last_loc,
            extend_num_tokens=5,
        )
        # extend_num_tokens output slots.
        self.assertEqual(out.shape, (5,))
        # Two pages consumed (the formula in the kernel is roundup-based).
        self.assertEqual(len(a.free_pages), 8 - 2)

    def test_extend_returns_none_when_pool_too_small(self):
        a = _make_allocator(size=8, page_size=4)  # 2 usable pages
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([100], dtype=torch.int32)  # >> 2 pages
        last_loc = torch.tensor([-1], dtype=torch.int32)
        out = a.alloc_extend(
            prefix_lens, prefix_lens.clone(),
            seq_lens, seq_lens.clone(),
            last_loc,
            extend_num_tokens=100,
        )
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
