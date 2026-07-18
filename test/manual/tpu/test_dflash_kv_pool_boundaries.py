"""CPU-runnable paged-KV allocator boundary / leak / reuse tests for DFlash.

DFlash allocates a fixed draft block per request, runs the target verify, then
keeps only the accepted prefix and frees the rest **at page granularity** (the
paged allocator can only release whole pages from the tail). Two things must
hold for correctness across many decode steps:

  1. The DFlash page-granular keep/free accounting (``_compute_paged_keep_slots``)
     retains exactly the accepted prefix rounded up to a page, clamped to the
     block, for zero / partial / full / mixed accepts.
  2. Feeding those keep counts into a *real* ``PagedTokenToKVPoolAllocator``
     leaks no pages (``available_size`` accounting stays exact), reuses freed
     pages validly (reallocated indices are in-range and unique), and fully
     reclaims back to the baseline once everything is freed.

Both the allocator (``alloc``/``free``/``clear``/``available_size`` are pure
torch; the kvcache handle is only stored, never touched by allocation) and
``_compute_paged_keep_slots`` run on CPU with no JAX/TPU device and no server
backend init, so this file is deterministic in a constrained environment.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.speculative.dflash_info import _compute_paged_keep_slots
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 64


def _make_allocator(num_pages: int) -> PagedTokenToKVPoolAllocator:
    # need_sort=False keeps free() deterministic (prepends freed pages); kvcache
    # is None because alloc/free never dereference it.
    return PagedTokenToKVPoolAllocator(
        size=num_pages * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


class TestComputePagedKeepSlots(CustomTestCase):
    """The DFlash page-granular keep/free accounting math (``_compute_paged_keep_slots``).

    keep_slots = clamp(min(round_up(prefix+commit, page), prefix+block) - prefix,
                       0, block). Prefixes are page-aligned in paged mode.
    """

    def _keep(self, prefix, commit, block):
        return _compute_paged_keep_slots(
            prefix_lens=torch.tensor(prefix, dtype=torch.int32),
            commit_lens=torch.tensor(commit, dtype=torch.int32),
            draft_token_num=block,
            page_size=PAGE_SIZE,
        )

    def test_single_page_block_reject_vs_accept(self):
        # block == one page: any commit>0 retains the whole page; commit==0 frees it.
        keep = self._keep([64, 64, 64], [0, 1, 64], block=64)
        self.assertEqual(keep.tolist(), [0, 64, 64])

    def test_two_page_block_partial_and_full(self):
        # block == two pages, prefix page-aligned at 64:
        #   commit 0  -> keep 0   (free both pages)
        #   commit 1  -> keep 64  (keep 1 page, free 1) — page-granular partial
        #   commit 64 -> keep 64  (still 1 page; 64 fills exactly page 1)
        #   commit 65 -> keep 128 (spills into page 2 -> keep both)
        #   commit128 -> keep 128 (full accept)
        keep = self._keep([64] * 5, [0, 1, 64, 65, 128], block=128)
        self.assertEqual(keep.tolist(), [0, 64, 64, 128, 128])

    def test_kept_plus_freed_equals_block_and_is_page_aligned(self):
        block = 128
        prefix = [0, 64, 128, 256]
        commit = [0, 30, 64, 128]
        keep = self._keep(prefix, commit, block)
        freed = block - keep
        # Every kept/freed count is a whole number of pages, and they partition
        # the block exactly (no lost or double-counted slots).
        self.assertTrue(torch.all(keep % PAGE_SIZE == 0), keep.tolist())
        self.assertTrue(torch.all(freed % PAGE_SIZE == 0), freed.tolist())
        self.assertTrue(torch.all(keep + freed == block))

    def test_clamped_to_block_and_nonnegative(self):
        # Absurd commit far beyond the block must clamp to the block, never exceed.
        keep = self._keep([64, 64], [10_000, 0], block=64)
        self.assertEqual(keep.tolist(), [64, 0])
        self.assertTrue(torch.all(keep >= 0))
        self.assertTrue(torch.all(keep <= 64))


class TestPagedAllocatorLeakReuse(CustomTestCase):
    """Drive a real ``PagedTokenToKVPoolAllocator`` through the DFlash
    alloc -> keep/free -> reuse cycle and assert no page leak or corruption.
    """

    def test_alloc_free_roundtrip_is_exact(self):
        alloc = _make_allocator(num_pages=16)
        base = alloc.available_size()
        loc = alloc.alloc(4 * PAGE_SIZE)
        self.assertIsNotNone(loc)
        self.assertEqual(alloc.available_size(), base - 4 * PAGE_SIZE)
        alloc.free(loc)
        self.assertEqual(alloc.available_size(), base)  # no leak

    def test_dflash_reject_partial_full_no_leak(self):
        # bs=4 requests, each allocates a 2-page draft block after a 1-page prefix.
        # commit lengths exercise reject / partial / full page-granular frees.
        block = 2 * PAGE_SIZE
        bs = 4
        alloc = _make_allocator(num_pages=32)
        base = alloc.available_size()

        keep = _compute_paged_keep_slots(
            prefix_lens=torch.full((bs,), PAGE_SIZE, dtype=torch.int32),
            commit_lens=torch.tensor([0, 1, 64, 128], dtype=torch.int32),
            draft_token_num=block,
            page_size=PAGE_SIZE,
        )
        self.assertEqual(keep.tolist(), [0, 64, 64, 128])

        # Allocate one contiguous block per request; request i owns slots
        # [i*block : (i+1)*block] of the returned flat index tensor.
        loc = alloc.alloc(bs * block)
        self.assertIsNotNone(loc)
        self.assertEqual(alloc.available_size(), base - bs * block)

        # Free each request's rejected tail (block - keep), page-aligned.
        kept_slots = []
        freed_tails = []
        for i in range(bs):
            seg = loc[i * block : (i + 1) * block]
            k = int(keep[i].item())
            kept_slots.append(seg[:k])
            freed_tails.append(seg[k:])
        freed = torch.cat(freed_tails)
        alloc.free(freed)

        total_keep = int(keep.sum().item())
        # Exactly the freed tail returned; kept slots still outstanding -> no leak.
        self.assertEqual(alloc.available_size(), base - total_keep)

        # Reuse: a fresh allocation of the freed capacity must succeed and hand
        # back only valid, unique, currently-free pages (no double-allocation of
        # a still-kept slot).
        outstanding = torch.cat([s for s in kept_slots if s.numel() > 0])
        reused = alloc.alloc(bs * block - total_keep)
        self.assertIsNotNone(reused)
        self.assertEqual(len(torch.unique(reused)), reused.numel())  # unique
        self.assertTrue(torch.all(reused >= 1))  # page 0 reserved
        self.assertLess(int(reused.max().item()), 32 * PAGE_SIZE)  # in range
        # No reused slot collides with a still-outstanding kept slot.
        overlap = torch.isin(reused, outstanding).any()
        self.assertFalse(bool(overlap), "reused a still-kept KV slot (corruption)")

        # Free everything -> full reclamation back to baseline (no leak).
        alloc.free(outstanding)
        alloc.free(reused)
        self.assertEqual(alloc.available_size(), base)

    def test_repeated_cycles_do_not_leak(self):
        # Many decode steps: repeated alloc/partial-free must never drift the
        # accounting (guards against a slow per-step page leak).
        block = 2 * PAGE_SIZE
        alloc = _make_allocator(num_pages=8)
        base = alloc.available_size()
        for _ in range(50):
            loc = alloc.alloc(block)
            self.assertIsNotNone(loc)
            # Keep 1 page, free 1 page (partial accept), then free the kept page.
            alloc.free(loc[PAGE_SIZE:])
            alloc.free(loc[:PAGE_SIZE])
        self.assertEqual(alloc.available_size(), base)


if __name__ == "__main__":
    unittest.main()
