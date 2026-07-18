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
from sglang.srt.speculative.dflash_info import (
    _compute_paged_keep_slots,
    compact_verify_out_cache_loc,
)
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
                       0, block). Prefixes are arbitrary in steady state (seq_lens
                       advances by the accepted commit length each step), so this
                       is exercised for both page-aligned and non-page-aligned
                       prefixes.
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


def _build_paged_extend(alloc, prefix_len, block, page=PAGE_SIZE):
    """Model DFlash's paged-extend ``out_cache_loc`` layout using ONLY plain
    (page-aligned) ``alloc`` calls -- no ``alloc_extend``/Triton.

    In paged mode a request's committed prefix occupies ``ceil(prefix/page)``
    pages; the last page is partial when ``prefix % page != 0``. A draft block of
    ``block`` new tokens is laid out as:
      1. the unused suffix ("tail") of that partial prefix page -- reused in place,
         NOT newly allocated, because the page is already owned by the request;
      2. newly allocated full pages for the remaining ``block - len(tail)`` tokens.
    ``out_cache_loc = concat(tail, new_page_slots)[:block]``.

    Returns ``(out_cache_loc, all_alloc, prefix_slots)`` where ``all_alloc`` is
    every slot the request holds (prefix pages + new pages).
    """
    prefix_pages = (prefix_len + page - 1) // page
    prefix_slots = alloc.alloc(prefix_pages * page)
    assert prefix_slots is not None
    tail = prefix_slots[prefix_len:]  # unused suffix of the last (partial) prefix page
    remaining = block - int(tail.numel())
    if remaining > 0:
        new_pages = (remaining + page - 1) // page
        new_slots = alloc.alloc(new_pages * page)
        assert new_slots is not None
    else:
        new_slots = prefix_slots.new_empty((0,))
    out_cache_loc = torch.cat([tail, new_slots])[:block]
    all_alloc = torch.cat([prefix_slots, new_slots])
    return out_cache_loc, all_alloc, prefix_slots


class TestPagedExtendNonAlignedPrefix(CustomTestCase):
    """Real-allocator leak/reuse for the STEADY-STATE DFlash path, where the
    committed prefix (``batch.seq_lens``) is arbitrary -- usually NOT page-aligned.

    Models the true paged-extend layout (partial-prefix-page tail + new full
    pages), frees the rejected tail exactly as ``dflash_info.verify`` does
    (``out_cache_loc[keep_slots:]`` with page-granular reclamation), and checks
    the page-boundary lengths {63,64,65,127,128,129} the plan calls for.
    """

    NUM_PAGES = 64

    def _run_case(self, prefix, commit, block):
        page = PAGE_SIZE
        alloc = _make_allocator(num_pages=self.NUM_PAGES)
        base = alloc.available_size()

        out_cache_loc, all_alloc, prefix_slots = _build_paged_extend(
            alloc, prefix, block
        )
        self.assertEqual(out_cache_loc.numel(), block)

        keep = int(
            _compute_paged_keep_slots(
                prefix_lens=torch.tensor([prefix], dtype=torch.int32),
                commit_lens=torch.tensor([commit], dtype=torch.int32),
                draft_token_num=block,
                page_size=page,
            )[0].item()
        )

        # Invariant: the partial prefix page's tail slots are always kept (freeing
        # them would corrupt the still-resident prefix).
        tail_len = ((prefix + page - 1) // page) * page - prefix
        self.assertGreaterEqual(
            keep, tail_len, f"would free the partial prefix page (p={prefix},c={commit})"
        )

        to_free = out_cache_loc[keep:]
        prefix_pages = set((prefix_slots // page).tolist())
        freed_pages = (
            set((to_free // page).unique().tolist()) if to_free.numel() else set()
        )
        # Never free a page that holds prefix data.
        self.assertTrue(
            freed_pages.isdisjoint(prefix_pages),
            f"freed a prefix page (p={prefix}, c={commit})",
        )

        alloc.free(to_free)

        # Exact page accounting: resident pages == ceil((prefix+commit)/page)
        # clamped by the allocated draft window ceil((prefix+block)/page).
        expected_resident = min(
            -(-(prefix + commit) // page), -(-(prefix + block) // page)
        )
        self.assertEqual(
            alloc.available_size(),
            base - expected_resident * page,
            f"page leak/over-free (p={prefix}, c={commit})",
        )

        # Slots the request still holds (prefix + kept draft), = non-freed pages.
        page_of = all_alloc // page
        if freed_pages:
            freed_t = torch.tensor(sorted(freed_pages), dtype=all_alloc.dtype)
            outstanding = all_alloc[~torch.isin(page_of, freed_t)]
        else:
            outstanding = all_alloc

        # Reuse: reallocating the freed capacity must hand back exactly the freed
        # pages (need_sort=False prepends them), unique, page-0-reserved, and
        # never colliding with a still-outstanding prefix/kept slot.
        freed_capacity = len(freed_pages) * page
        if freed_capacity > 0:
            reused = alloc.alloc(freed_capacity)
            self.assertIsNotNone(reused)
            self.assertEqual(len(torch.unique(reused)), reused.numel())
            self.assertTrue(torch.all(reused >= 1))  # page 0 reserved for padding
            self.assertEqual(
                set((reused // page).tolist()),
                freed_pages,
                "reuse handed back pages other than the freed ones",
            )
            self.assertFalse(
                bool(torch.isin(reused, outstanding).any()),
                f"reused a still-outstanding slot (p={prefix}, c={commit})",
            )
            alloc.free(reused)

        # Free everything the request holds -> full reclamation, no leak.
        alloc.free(outstanding)
        self.assertEqual(alloc.available_size(), base)

    def test_page_boundary_prefixes_reject_partial_full(self):
        block = 2 * PAGE_SIZE  # 128: makes commit==65 a valid past-page-edge case
        for prefix in [63, 64, 65, 127, 128, 129]:
            for commit in [0, 1, 64, 65, block]:
                with self.subTest(prefix=prefix, commit=commit):
                    self._run_case(prefix, commit, block)


class TestCompactVerifyOutCacheLoc(CustomTestCase):
    """int64 gather-coordinate hygiene + correctness for verify()'s KV compaction
    (``compact_verify_out_cache_loc``), the exact helper ``DFlashVerifyInput.verify``
    calls to split a verify block into freed vs kept KV slots.
    """

    def _ocl(self, bs, dtn, dtype=torch.int64):
        # Distinct positive slot ids so free/keep sets are checkable by value.
        return torch.arange(PAGE_SIZE, PAGE_SIZE + bs * dtn, dtype=dtype)

    def test_page_size_1_frees_all_uncommitted(self):
        dtn = 4
        ocl = self._ocl(2, dtn)  # req0: 64..67, req1: 68..71
        commit = torch.tensor([2, 0], dtype=torch.int32)
        free, keep = compact_verify_out_cache_loc(ocl, commit, dtn, page_size=1)
        self.assertEqual(keep.tolist(), [64, 65])  # req0's first 2
        self.assertEqual(sorted(free.tolist()), [66, 67, 68, 69, 70, 71])
        # kept ∪ freed partitions the whole block, no overlap.
        self.assertEqual(
            sorted(free.tolist() + keep.tolist()), list(range(64, 72))
        )

    def test_paged_keeps_page_granular_tail(self):
        dtn = 4
        ocl = self._ocl(2, dtn)
        commit = torch.tensor([2, 0], dtype=torch.int32)
        # keep_slots rounds the kept region up to a page, so req0 keeps ALL 4 draft
        # slots (page-granular) while committing only 2; req1 keeps none.
        keep_slots = torch.tensor([4, 0], dtype=torch.int64)
        free, keep = compact_verify_out_cache_loc(
            ocl, commit, dtn, page_size=64, keep_slots=keep_slots
        )
        self.assertEqual(keep.tolist(), [64, 65])  # committed prefix only
        self.assertEqual(sorted(free.tolist()), [68, 69, 70, 71])  # req1's page
        # Paged mode frees FEWER slots than page_size==1 (66,67 stay resident).
        self.assertNotIn(66, free.tolist())

    def test_int64_coordinates_even_from_int32_input(self):
        # The lock-in: torch_tpu gather rejects mixed i32/i64, so the helper must
        # coerce coordinates to int64 regardless of the input dtype.
        dtn = 4
        commit = torch.tensor([3, 1], dtype=torch.int32)
        for page_size, ks in ((1, None), (64, torch.tensor([4, 4], dtype=torch.int64))):
            free, keep = compact_verify_out_cache_loc(
                self._ocl(2, dtn, dtype=torch.int32),
                commit,
                dtn,
                page_size=page_size,
                keep_slots=ks,
            )
            self.assertEqual(keep.dtype, torch.int64, f"keep dtype (page={page_size})")
            self.assertEqual(free.dtype, torch.int64, f"free dtype (page={page_size})")

    def test_paged_requires_keep_slots(self):
        with self.assertRaises(ValueError):
            compact_verify_out_cache_loc(
                self._ocl(1, 4), torch.tensor([1], dtype=torch.int32), 4, page_size=64
            )

    def test_freed_slots_feed_real_allocator_without_leak(self):
        # End-to-end with a real allocator: the freed coordinates from the helper
        # release exactly the expected pages and reclaim fully.
        dtn = PAGE_SIZE  # one page per draft slot-row group
        bs = 2
        alloc = _make_allocator(num_pages=16)
        base = alloc.available_size()
        block_locs = alloc.alloc(bs * dtn)  # 2 pages
        commit = torch.tensor([dtn, 0], dtype=torch.int32)  # req0 full, req1 reject
        keep_slots = _compute_paged_keep_slots(
            prefix_lens=torch.tensor([PAGE_SIZE, PAGE_SIZE], dtype=torch.int32),
            commit_lens=commit,
            draft_token_num=dtn,
            page_size=PAGE_SIZE,
        )
        free, keep = compact_verify_out_cache_loc(
            block_locs, commit, dtn, page_size=PAGE_SIZE, keep_slots=keep_slots
        )
        alloc.free(free)
        # req0 keeps its page, req1's page is freed -> exactly one page resident.
        self.assertEqual(alloc.available_size(), base - PAGE_SIZE)
        alloc.free(keep)
        self.assertEqual(alloc.available_size(), base)


if __name__ == "__main__":
    unittest.main()
