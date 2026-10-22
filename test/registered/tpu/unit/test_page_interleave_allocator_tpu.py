"""Unit tests for :class:`TpuPageInterleavePoolAllocator`, the page-interleaved
full-attention allocator of a TPU decode-context-parallel group, and for the
SWA composite that pairs it with a stock sliding-window allocator.

Host-only logic: the allocators keep their free lists on the CPU, so every case
runs on ``device="cpu"``.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import swa as swa_module
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.platforms import current_platform
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
    TpuPageInterleavePoolAllocator,
)
from sglang_tpu.kv_free_fence import KVFreeFence, _page_pools

PS = 8
N = 2
PAGES_PER_RANK = 16


def _make_allocator(pages_per_rank=PAGES_PER_RANK, need_sort=False):
    return TpuPageInterleavePoolAllocator(
        size=pages_per_rank * PS,
        physical_page_size=PS,
        shard_size=N,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _t(values):
    return torch.tensor(values, dtype=torch.int64)


def _extend(alloc, prefix_lens, seq_lens, last_locs):
    return alloc.alloc_extend(
        prefix_lens=_t(prefix_lens),
        prefix_lens_cpu=_t(prefix_lens),
        seq_lens=_t(seq_lens),
        seq_lens_cpu=_t(seq_lens),
        last_loc=_t(last_locs),
        extend_num_tokens=sum(s - p for p, s in zip(prefix_lens, seq_lens)),
    )


def _grow(alloc, locs, prefix_len, seq_len):
    """Extend one request whose tokens so far are ``locs`` (a flat tensor)."""
    last = int(locs[-1]) if locs.numel() else -1
    out = _extend(alloc, [prefix_len], [seq_len], [last])
    assert out is not None
    return torch.cat([locs, out.to(torch.int64)])


def _owner(loc):
    return (loc // PS) % N


def _local_page(loc):
    return (loc // PS) // N


class TestPlacement(CustomTestCase):
    """Position page j of a request is logical page l with l % N == j % N,
    stored at local physical page l // N, whatever the allocation history."""

    def test_fresh_request_pages_alternate_by_position(self):
        alloc = _make_allocator()
        locs = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, 5 * PS + 3)
        positions = torch.arange(locs.numel())
        self.assertTrue(torch.equal(_owner(locs), (positions // PS) % N))
        # Tokens of one position page are contiguous in logical space.
        self.assertTrue(torch.equal(locs % PS, positions % PS))

    def test_unaligned_extends_continue_the_run(self):
        alloc = _make_allocator()
        locs = torch.empty(0, dtype=torch.int64)
        length = 0
        for ext in (3, PS, 2 * PS + 1, 1, PS - 1, 4 * PS):
            locs = _grow(alloc, locs, length, length + ext)
            length += ext
        positions = torch.arange(length)
        self.assertTrue(torch.equal(_owner(locs), (positions // PS) % N))
        self.assertEqual(len(torch.unique(locs)), length)

    def test_batch_extend_matches_per_request_extend(self):
        single = _make_allocator()
        batched = _make_allocator()
        prefix, seq = [0, 3, PS * 2 + 5], [PS + 1, PS * 3, PS * 2 + 5]
        last_locs = []
        for p in prefix:
            # Give the prefixes real pages in both allocators.
            if p:
                a = _grow(single, torch.empty(0, dtype=torch.int64), 0, p)
                b = _grow(batched, torch.empty(0, dtype=torch.int64), 0, p)
                self.assertTrue(torch.equal(a, b))
                last_locs.append(int(a[-1]))
            else:
                last_locs.append(-1)
        out = _extend(batched, prefix, seq, last_locs)
        parts = [
            _extend(single, [p], [s], [l]) for p, s, l in zip(prefix, seq, last_locs)
        ]
        self.assertTrue(torch.equal(out, torch.cat(parts)))

    def test_decode_opens_the_next_position_page(self):
        alloc = _make_allocator()
        locs = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, PS)
        for step in range(3 * PS):
            seq_len = PS + step + 1
            out = alloc.alloc_decode(_t([seq_len]), _t([seq_len]), locs[-1:])
            self.assertIsNotNone(out)
            locs = torch.cat([locs, out.to(torch.int64)])
        positions = torch.arange(locs.numel())
        self.assertTrue(torch.equal(_owner(locs), (positions // PS) % N))
        self.assertEqual(len(torch.unique(locs)), locs.numel())

    def test_decode_batch_mixes_continuing_and_opening_requests(self):
        alloc = _make_allocator()
        a = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, PS)  # opens page 1
        b = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, 3)  # continues
        c = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, 2 * PS)  # opens page 2
        out = alloc.alloc_decode(
            _t([PS + 1, 4, 2 * PS + 1]),
            _t([PS + 1, 4, 2 * PS + 1]),
            torch.stack([a[-1], b[-1], c[-1]]),
        )
        self.assertEqual(int(out[1]), int(b[-1]) + 1)
        self.assertEqual(_owner(out[0]).item(), 1)
        self.assertEqual(_owner(out[2]).item(), 0)
        self.assertEqual(int(out[0]) % PS, 0)
        self.assertEqual(int(out[2]) % PS, 0)

    def test_rotation_bases_other_than_zero_are_refused(self):
        alloc = _make_allocator()
        with self.assertRaisesRegex(ValueError, "rotation base 0"):
            alloc.alloc_extend(
                prefix_lens=_t([0]),
                prefix_lens_cpu=_t([0]),
                seq_lens=_t([PS]),
                seq_lens_cpu=_t([PS]),
                last_loc=_t([-1]),
                extend_num_tokens=PS,
                rotation_bases=[1],
            )
        self.assertIsNotNone(_extend(alloc, [0], [PS], [-1]))


class TestAccounting(CustomTestCase):
    def test_sizes_and_predicate(self):
        alloc = _make_allocator()
        self.assertIsInstance(alloc, PageInterleavePoolAllocator)
        self.assertEqual(page_interleave_shard_size(alloc), N)
        self.assertEqual(alloc.page_size, PS)
        self.assertEqual(alloc.size, N * PAGES_PER_RANK * PS)
        self.assertEqual(alloc.num_pages, N * PAGES_PER_RANK)
        self.assertEqual(alloc.class_free_page_counts(), [PAGES_PER_RANK] * N)
        self.assertEqual(alloc.available_size(), N * PAGES_PER_RANK * PS)
        self.assertEqual(alloc.aggregate_free_size(), N * PAGES_PER_RANK * PS)
        free = alloc.get_all_free_pages()
        self.assertEqual(free.numel(), N * PAGES_PER_RANK)
        # Logical page 0..N-1 (physical page 0 of every class) is reserved.
        self.assertEqual(int(free.min()), N)

    def test_available_size_is_the_min_class_floor(self):
        alloc = _make_allocator()
        # Three single-page requests: classes filled 3 / 0.
        for _ in range(3):
            _grow(alloc, torch.empty(0, dtype=torch.int64), 0, PS)
        self.assertEqual(
            alloc.class_free_page_counts(), [PAGES_PER_RANK - 3, PAGES_PER_RANK]
        )
        self.assertEqual(alloc.available_size(), N * PS * (PAGES_PER_RANK - 3))
        self.assertEqual(alloc.aggregate_free_size(), PS * (2 * PAGES_PER_RANK - 3))

    def test_free_returns_pages_to_their_class(self):
        alloc = _make_allocator()
        locs = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, 5 * PS)
        self.assertEqual(
            alloc.class_free_page_counts(), [PAGES_PER_RANK - 3, PAGES_PER_RANK - 2]
        )
        alloc.free(locs[: 2 * PS])
        self.assertEqual(
            alloc.class_free_page_counts(), [PAGES_PER_RANK - 2, PAGES_PER_RANK - 1]
        )
        alloc.free_segment(locs[2 * PS :], start_pos=2 * PS)
        self.assertEqual(alloc.class_free_page_counts(), [PAGES_PER_RANK] * N)
        alloc.debug_check_classes()
        # The stock pool's double-free guard still fires.
        with self.assertRaisesRegex(RuntimeError, "already free"):
            alloc.free(locs[:PS])

    def test_free_group_defers_inside_the_class_pools(self):
        alloc = _make_allocator()
        locs = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, 4 * PS)
        alloc.free_group_begin()
        alloc.free(locs)
        self.assertEqual(alloc.class_free_page_counts(), [PAGES_PER_RANK - 2] * N)
        alloc.free_group_end()
        self.assertEqual(alloc.class_free_page_counts(), [PAGES_PER_RANK] * N)

    def test_refuses_when_one_class_is_short_without_taking_pages(self):
        alloc = _make_allocator(pages_per_rank=2)
        # Four single-page requests exhaust class 0 (positions page 0).
        for _ in range(2):
            _grow(alloc, torch.empty(0, dtype=torch.int64), 0, PS)
        self.assertEqual(alloc.class_free_page_counts(), [0, 2])
        self.assertEqual(alloc.available_size(), 0)
        self.assertIsNone(_extend(alloc, [0], [PS], [-1]))
        self.assertIsNone(_extend(alloc, [0], [2 * PS], [-1]))
        self.assertEqual(alloc.class_free_page_counts(), [0, 2])

    def test_clear_resets_every_class(self):
        alloc = _make_allocator()
        _grow(alloc, torch.empty(0, dtype=torch.int64), 0, 3 * PS)
        alloc.clear()
        self.assertEqual(alloc.class_free_page_counts(), [PAGES_PER_RANK] * N)

    def test_kv_free_fence_attaches_to_the_class_pools(self):
        alloc = _make_allocator()
        pools = _page_pools(alloc)
        self.assertEqual(pools, alloc.class_pools)
        self.assertTrue(
            all(isinstance(p, TpuPagedTokenToKVPoolAllocator) for p in pools)
        )
        fence = KVFreeFence(synchronize_device=lambda: None)
        for pool in pools:
            fence.attach(pool)
        locs = _grow(alloc, torch.empty(0, dtype=torch.int64), 0, 2 * PS)
        fence.on_batch_formed()
        result = SimpleNamespace(copy_done=None)
        fence.on_forward_launched(result)
        alloc.free(locs)
        # Held behind the in-flight forward: counted available, not in the free lists.
        self.assertEqual(sum(p.held_pages.num_pages for p in pools), 2)
        self.assertEqual(alloc.class_free_page_counts(), [PAGES_PER_RANK] * N)
        fence.on_forward_done(result)
        self.assertEqual(sum(p.held_pages.num_pages for p in pools), 0)


class _FakeSwaPool(BaseSWAKVPool):
    """The composite only stores the pool and registers the mapping with it."""

    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.swa_req_ring_size = None
        self.full_to_swa_index_mapping = None

    def register_mapping(self, mapping):
        self.full_to_swa_index_mapping = mapping

    def translate_loc_from_full_to_swa(self, kv_indices):
        return self.full_to_swa_index_mapping[kv_indices.to(torch.int64)]

    def _unused(self, *args, **kwargs):
        raise NotImplementedError("the allocator never reads KV buffers")

    get_key_buffer = get_value_buffer = get_kv_buffer = _unused
    get_state_buf_infos = set_kv_buffer = _unused


def _as_tpu():
    """The composite picks its sub-allocators from the platform; select the
    TPU ones here (host-resident, so they run on the CPU)."""
    return (
        patch.object(swa_module, "_is_tpu", True),
        patch.object(
            swa_module,
            "TpuPagedTokenToKVPoolAllocator",
            TpuPagedTokenToKVPoolAllocator,
            create=True,
        ),
        patch.object(
            current_platform,
            "get_page_interleave_allocator_cls",
            return_value=TpuPageInterleavePoolAllocator,
        ),
    )


class TestSwaComposite(CustomTestCase):
    """SWATokenToKVPoolAllocator(full_shard_size=2): a classed full side over 2x
    logical slots and a stock SWA side over its own rows at the physical page."""

    FULL_ROWS = PAGES_PER_RANK * PS
    SWA_ROWS = 6 * PS

    def setUp(self):
        # The composite reads _is_tpu at call time too (host-resident mapping).
        for patcher in _as_tpu():
            patcher.start()
            self.addCleanup(patcher.stop)

    def _make(self):
        return swa_module.SWATokenToKVPoolAllocator(
            N * self.FULL_ROWS,
            self.SWA_ROWS,
            page_size=PS,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=_FakeSwaPool(),
            need_sort=False,
            full_shard_size=N,
        )

    def test_geometry(self):
        alloc = self._make()
        self.assertIsInstance(alloc.full_attn_allocator, TpuPageInterleavePoolAllocator)
        self.assertIsInstance(alloc.swa_attn_allocator, TpuPagedTokenToKVPoolAllocator)
        self.assertNotIsInstance(alloc.swa_attn_allocator, PageInterleavePoolAllocator)
        self.assertEqual(alloc.page_size, PS)
        self.assertEqual(alloc.size_full, N * self.FULL_ROWS)
        self.assertEqual(alloc.size_swa, self.SWA_ROWS)
        self.assertEqual(alloc.full_available_size(), N * self.FULL_ROWS)
        self.assertEqual(alloc.swa_available_size(), self.SWA_ROWS)
        # Indexed by the full LOGICAL location, trailing -1 included.
        self.assertEqual(
            alloc.full_to_swa_index_mapping.numel(), N * self.FULL_ROWS + PS + 1
        )
        # Seams keyed on the predicate stay off for the composite.
        self.assertEqual(page_interleave_shard_size(alloc), 1)
        self.assertEqual(page_interleave_shard_size(alloc.full_attn_allocator), N)

    def test_prealloc_extend_then_decode_keep_the_mapping(self):
        alloc = self._make()
        # Decode-instance preallocation: full KV for the whole request, SWA for
        # the window tail only (page-aligned tail start).
        fill_len = 4 * PS + 3
        tail = 2 * PS + 3
        locs = alloc.alloc_extend_swa_tail(
            prefix_lens=_t([0]),
            prefix_lens_cpu=_t([0]),
            seq_lens=_t([fill_len]),
            seq_lens_cpu=_t([fill_len]),
            last_loc=_t([-1]),
            extend_num_tokens=fill_len,
            swa_tail_len=tail,
        ).to(torch.int64)
        positions = torch.arange(fill_len)
        self.assertTrue(torch.equal(_owner(locs), (positions // PS) % N))
        mapping = alloc.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[locs[:-tail]] == 0))
        swa = mapping[locs[-tail:]]
        self.assertTrue(torch.all(swa > 0))
        self.assertEqual(alloc.swa_available_size(), self.SWA_ROWS - 3 * PS)
        # Verify-step extends and plain decodes page both sides in step.
        out = alloc.alloc_extend(
            prefix_lens=_t([fill_len]),
            prefix_lens_cpu=_t([fill_len]),
            seq_lens=_t([fill_len + PS]),
            seq_lens_cpu=_t([fill_len + PS]),
            last_loc=locs[-1:],
            extend_num_tokens=PS,
        ).to(torch.int64)
        locs = torch.cat([locs, out])
        self.assertTrue(torch.all(mapping[out] > 0))
        dec = alloc.alloc_decode(
            _t([fill_len + PS + 1]), _t([fill_len + PS + 1]), locs[-1:]
        ).to(torch.int64)
        locs = torch.cat([locs, dec])
        positions = torch.arange(locs.numel())
        self.assertTrue(torch.equal(_owner(locs), (positions // PS) % N))
        self.assertEqual(len(torch.unique(locs)), locs.numel())
        self.assertEqual(
            len(torch.unique(mapping[locs[-(tail + PS + 1) :]])), tail + PS + 1
        )
        # Freeing returns both sides and clears the mapping.
        alloc.free(locs)
        self.assertEqual(alloc.full_available_size(), N * self.FULL_ROWS)
        self.assertEqual(alloc.swa_available_size(), self.SWA_ROWS)
        self.assertTrue(torch.all(mapping[:-1] == 0))

    def test_resize_is_refused(self):
        alloc = self._make()
        with self.assertRaisesRegex(NotImplementedError, "page-interleaved"):
            alloc.resize(
                SimpleNamespace(full_max_total_num_tokens=8, swa_max_total_num_tokens=8)
            )

    def test_default_composite_is_unchanged(self):
        alloc = swa_module.SWATokenToKVPoolAllocator(
            self.FULL_ROWS,
            self.SWA_ROWS,
            page_size=PS,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=_FakeSwaPool(),
            need_sort=False,
        )
        self.assertIs(type(alloc.full_attn_allocator), TpuPagedTokenToKVPoolAllocator)
        self.assertEqual(
            alloc.full_to_swa_index_mapping.numel(), self.FULL_ROWS + PS + 1
        )


if __name__ == "__main__":
    unittest.main()
