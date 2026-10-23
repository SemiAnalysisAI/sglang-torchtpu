# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.allocator_tpu import host_return_allocators
from sglang_tpu.kv_free_fence import KVFreeFence
from sglang_tpu.unified_kv_pool import (
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
    merge_swa_kv_heads,
    num_full_groups,
    num_shared_kv_buffers,
    num_swa_groups,
    split_swa_output,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuUnifiedSWAKVPoolStorage(unittest.TestCase):
    """Tests for PR 1: shared physical buffers, SWA group page mapping, and head merging."""

    def test_equal_full_and_swa_layers_share_buffers(self):
        pool = TpuUnifiedSWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=4,
            head_dim=128,
            swa_attention_layer_ids=[0, 2],
            full_attention_layer_ids=[1, 3],
            device="cpu",
        )
        self.assertEqual(pool.size, 64)
        self.assertEqual(pool.size_swa, 64)
        self.assertEqual(pool.num_swa_groups, 1)
        self.assertEqual(pool.num_shared_buffers, 2)
        self.assertFalse(pool.swa_kv_heads_merged)
        self.assertEqual(len(pool.get_state()), 2)

        # Full layer i and SWA layer i alias the exact same physical tensor.
        self.assertEqual(
            pool.get_fused_kv_buffer(1).data_ptr(),
            pool.get_fused_kv_buffer(0).data_ptr(),
        )
        self.assertEqual(
            pool.get_fused_kv_buffer(3).data_ptr(),
            pool.get_fused_kv_buffer(2).data_ptr(),
        )
        self.assertEqual(pool.swa_page_group(0), 0)
        self.assertEqual(pool.swa_page_group(2), 0)
        self.assertEqual(
            pool.get_kv_size_bytes(), pool._backing_pool.get_kv_size_bytes()
        )
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_gemma4_ratio_and_head_merging_bf16_and_fp8(self):
        # Gemma 4 31B ratio: 5 SWA layers per 1 Full layer (here 2 Full, 10 SWA -> M = 5),
        # with Full (1 head x 512) and SWA (2 heads x 256).
        full_ids = [5, 11]
        swa_ids = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = TpuUnifiedSWAKVPool(
                    size=80,
                    size_swa=40,
                    page_size=4,
                    dtype=dtype,
                    head_num=1,
                    head_dim=512,
                    swa_head_num=2,
                    swa_head_dim=256,
                    swa_attention_layer_ids=swa_ids,
                    full_attention_layer_ids=full_ids,
                    device="cpu",
                )
                self.assertTrue(pool.swa_kv_heads_merged)
                self.assertEqual(pool.num_swa_groups, 5)
                self.assertEqual(pool.num_shared_buffers, 2)
                # 80 // 4 = 20 usable pages + 1 reserved page-0 sink = 21 blocks.
                self.assertEqual(pool._backing_pool.num_blocks, 21)
                self.assertEqual(pool.swa_group_pages.shape, (4, 21))
                self.assertEqual(len(pool.get_state()), 2)

                for j, layer_id in enumerate(swa_ids):
                    self.assertEqual(pool.swa_page_group(layer_id), j // 2)
                    self.assertEqual(
                        pool.get_fused_kv_buffer(layer_id).data_ptr(),
                        pool.get_state()[j % 2].data_ptr(),
                    )

                new_state = [torch.zeros_like(buf) for buf in pool.get_state()]
                pool.set_state(new_state)
                for j, layer_id in enumerate(swa_ids):
                    self.assertEqual(
                        pool.get_fused_kv_buffer(layer_id).data_ptr(),
                        new_state[j % 2].data_ptr(),
                    )

    def test_incompatible_page_shapes_rejected(self):
        with self.assertRaises(ValueError):
            TpuUnifiedSWAKVPool(
                size=32,
                size_swa=32,
                page_size=4,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=512,
                swa_head_num=1,
                swa_head_dim=128,
                swa_attention_layer_ids=[0],
                full_attention_layer_ids=[1],
                device="cpu",
            )
        with self.assertRaises(ValueError):
            num_swa_groups(0, 4)

    def test_merge_swa_kv_heads_and_split_output_exact(self):
        torch.manual_seed(0)
        num_tokens, num_q_heads, num_kv_heads, head_dim = 6, 4, 2, 256
        q = torch.randn(num_tokens, num_q_heads, head_dim, dtype=torch.float32)
        k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float32)
        v = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float32)

        # Reference unmerged GQA attention per KV head group.
        q_grouped = q.view(
            num_tokens, num_kv_heads, num_q_heads // num_kv_heads, head_dim
        )
        scores_ref = torch.einsum("ighd,tgd->ghit", q_grouped, k)
        probs_ref = torch.softmax(scores_ref * (head_dim**-0.5), dim=-1)
        out_ref = torch.einsum("ghit,tgd->ighd", probs_ref, v).reshape(
            num_tokens, num_q_heads, head_dim
        )

        # Merged single-KV-head attention + split_swa_output.
        q_m, k_m, v_m = merge_swa_kv_heads(q, k, v)
        self.assertEqual(q_m.shape, (num_tokens, num_q_heads, num_kv_heads * head_dim))
        self.assertEqual(k_m.shape, (num_tokens, 1, num_kv_heads * head_dim))
        self.assertEqual(v_m.shape, (num_tokens, 1, num_kv_heads * head_dim))

        scores_m = torch.einsum("ihd,tsd->hit", q_m, k_m)
        probs_m = torch.softmax(scores_m * (head_dim**-0.5), dim=-1)
        out_m = torch.einsum("hit,tsd->ihd", probs_m, v_m)
        out_unpacked = split_swa_output(out_m, num_kv_heads)

        torch.testing.assert_close(out_unpacked, out_ref, rtol=1e-5, atol=1e-5)

    def test_swa_cpu_copy_and_restore_respects_group_pages(self):
        # 1 Full layer, 3 SWA layers -> M = 3 groups sharing 1 physical buffer.
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=48,
            size_swa=48,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=[0, 1, 2],
            full_attention_layer_ids=[3],
            device="cpu",
        )
        buf = pool.get_state()[0]
        # Map SWA page 3 to scattered physical pages [3, 9, 11] across groups 0, 1, 2.
        pool.swa_group_pages[:, 3] = torch.tensor([9, 11], dtype=torch.int32)
        for g, phys_page in enumerate((3, 9, 11)):
            buf[phys_page].fill_(float(10 + g))

        # SWA primary-row token indices for page 3 (tokens 12..15).
        src_swa_indices = torch.arange(3 * page_size, 4 * page_size, dtype=torch.long)
        cpu_copy = pool.swa_kv_pool.get_cpu_copy(src_swa_indices)

        # Restore into SWA page 6 mapped to scattered physical pages [6, 2, 10].
        pool.swa_group_pages[:, 6] = torch.tensor([2, 10], dtype=torch.int32)
        dst_swa_indices = torch.arange(6 * page_size, 7 * page_size, dtype=torch.long)
        pool.swa_kv_pool.load_cpu_copy(cpu_copy, dst_swa_indices)

        for g, phys_page in enumerate((6, 2, 10)):
            self.assertTrue(torch.all(buf[phys_page] == float(10 + g)).item())

    def test_gcd_shared_buffers_zero_internal_wastage(self):
        # 8 Full layers and 20 SWA layers: gcd(8, 20) = 4 shared buffers,
        # M_full = 2 groups of 4 layers, M_swa = 5 groups of 4 layers (0% internal waste).
        self.assertEqual(num_shared_kv_buffers(8, 20), 4)
        self.assertEqual(num_full_groups(8, 20), 2)
        self.assertEqual(num_swa_groups(8, 20), 5)

        page_size = 4
        full_ids = list(range(8))
        swa_ids = list(range(8, 28))
        pool = TpuUnifiedSWAKVPool(
            size=40,
            size_swa=40,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=swa_ids,
            full_attention_layer_ids=full_ids,
            device="cpu",
        )
        self.assertEqual(pool.num_shared_buffers, 4)
        self.assertEqual(pool.num_full_groups, 2)
        self.assertEqual(pool.num_swa_groups, 5)
        self.assertEqual(pool.full_group_pages.shape, (1, 11))
        self.assertEqual(pool.swa_group_pages.shape, (4, 11))
        self.assertEqual(len(pool.get_state()), 4)

        for i, layer_id in enumerate(full_ids):
            self.assertEqual(pool.full_page_group(layer_id), i // 4)
            self.assertEqual(
                pool.get_fused_kv_buffer(layer_id).data_ptr(),
                pool.get_state()[i % 4].data_ptr(),
            )
        for j, layer_id in enumerate(swa_ids):
            self.assertEqual(pool.swa_page_group(layer_id), j // 4)
            self.assertEqual(
                pool.get_fused_kv_buffer(layer_id).data_ptr(),
                pool.get_state()[j % 4].data_ptr(),
            )

        # Verify Full CPU copy/restore respects full_group_pages when M_full > 1.
        pool.full_group_pages[0, 2] = 7
        for i in range(8):
            phys_page = 2 if (i // 4) == 0 else 7
            pool.get_state()[i % 4][phys_page].fill_(float(100 + i))

        src_full_idx = torch.arange(2 * page_size, 3 * page_size, dtype=torch.long)
        cpu_copy = pool.full_kv_pool.get_cpu_copy(src_full_idx)

        pool.full_group_pages[0, 5] = 9
        dst_full_idx = torch.arange(5 * page_size, 6 * page_size, dtype=torch.long)
        pool.full_kv_pool.load_cpu_copy(cpu_copy, dst_full_idx)
        for i in range(8):
            phys_page = 5 if (i // 4) == 0 else 9
            self.assertTrue(
                torch.all(pool.get_state()[i % 4][phys_page] == float(100 + i)).item()
            )


class TestTpuUnifiedLogicalToPhysicalAllocator(unittest.TestCase):
    """Tests for PR 2: Zero-fragmentation logical-to-physical allocation, reclamation, and eviction."""

    def _make_gemma4_pool_and_allocator(self, size: int = 80, page_size: int = 4):
        pool = TpuUnifiedSWAKVPool(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=512,
            swa_head_num=2,
            swa_head_dim=256,
            swa_attention_layer_ids=[0, 1, 2, 3, 4],
            full_attention_layer_ids=[5],
            device="cpu",
        )
        allocator = TpuUnifiedSWATokenToKVPoolAllocator(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=True,
        )
        return pool, allocator

    def test_equal_ratio_alloc_extend_decode_and_free(self):
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=32,
            size_swa=32,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        allocator = TpuUnifiedSWATokenToKVPoolAllocator(
            size=32,
            size_swa=32,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=True,
        )
        self.assertEqual(allocator.full_available_size(), 32)
        self.assertEqual(allocator.swa_available_size(), 32)
        self.assertEqual(allocator.max_extend_tokens(), 16)

        prefix_lens = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([4], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64)
        full_idx = allocator.alloc_extend(
            prefix_lens, prefix_lens, seq_lens, seq_lens, last_loc, 4
        )
        self.assertIsNotNone(full_idx)
        swa_idx = allocator.translate_loc_from_full_to_swa(full_idx)
        self.assertFalse(
            torch.equal(full_idx // page_size, swa_idx // page_size),
            "Full and SWA pages must occupy distinct physical pages in the shared buffer.",
        )

        # Decode crossing page boundary (seq_len = 5 -> needs 1 new page each).
        dec_seq_lens = torch.tensor([5], dtype=torch.int64)
        dec_last_loc = full_idx[-1:]
        dec_idx = allocator.alloc_decode(dec_seq_lens, dec_seq_lens, dec_last_loc)
        self.assertIsNotNone(dec_idx)
        dec_swa_idx = allocator.translate_loc_from_full_to_swa(dec_idx)
        self.assertTrue(int(dec_swa_idx.item()) > 0)

        # Free all tokens and verify full page restoration.
        all_full = torch.cat([full_idx, dec_idx])
        allocator.free(all_full)
        self.assertEqual(allocator.full_available_size(), 32)
        self.assertEqual(allocator.swa_available_size(), 32)

    def test_zero_fragmentation_scattered_free_pages(self):
        # 10 usable physical pages (pages 1..10), page_size=4, M=5 SWA groups.
        pool, allocator = self._make_gemma4_pool_and_allocator(size=40, page_size=4)
        store = allocator._page_store
        self.assertEqual(store.num_free_pages(), 10)
        self.assertEqual(allocator.full_available_size(), 40)
        self.assertEqual(allocator.swa_available_size(), 8)

        # Allocate all 10 physical pages individually as Full Attention pages.
        fa_pages = [allocator.full_attn_allocator.alloc(4) for _ in range(10)]
        self.assertEqual(store.num_free_pages(), 0)

        # Free pages at indices 1, 2, 3, 4, 6, 7, 8, 9 (leaving pages 1 and 6 occupied by FA).
        # No 5 contiguous physical pages exist anywhere in HBM, yet 8 free pages exist!
        for idx in (1, 2, 3, 4, 6, 7, 8, 9):
            allocator.full_attn_allocator.free(fa_pages[idx])
        self.assertEqual(store.num_free_pages(), 8)
        self.assertEqual(allocator.full_available_size(), 32)
        self.assertEqual(allocator.swa_available_size(), 4)

        # SWA allocation of 1 logical page (4 tokens = 5 physical pages) succeeds with zero fragmentation!
        swa_tokens = allocator.swa_attn_allocator.alloc(4)
        self.assertIsNotNone(swa_tokens)
        primary_page = int(swa_tokens[0].item()) // 4
        companion_pages = pool.swa_group_pages[:, primary_page].tolist()
        all_swa_phys_pages = [primary_page] + companion_pages
        self.assertEqual(len(set(all_swa_phys_pages)), 5)
        for p in all_swa_phys_pages:
            self.assertNotIn(p, (1, 6))
            self.assertFalse(bool(store.page_is_free[p].item()))

        # Exactly 3 physical pages remain free for Full Attention.
        self.assertEqual(store.num_free_pages(), 3)
        self.assertEqual(allocator.full_available_size(), 12)
        self.assertEqual(allocator.swa_available_size(), 0)

        # Release the SWA logical page and remaining FA pages -> all 10 pages are free again.
        allocator.swa_attn_allocator.free(swa_tokens)
        allocator.full_attn_allocator.free(fa_pages[0])
        allocator.full_attn_allocator.free(fa_pages[5])
        self.assertEqual(store.num_free_pages(), 10)
        self.assertTrue(bool(store.page_is_free[1:11].all()))

    def test_can_reserve_eviction_and_leak_conservation(self):
        _, allocator = self._make_gemma4_pool_and_allocator(size=80, page_size=4)
        # Empty pool: 20 physical pages. 2 SWA pages (10 phys) + 10 FA pages (10 phys) fits.
        self.assertTrue(allocator.can_reserve(40, 8, empty_pool=True))
        self.assertFalse(allocator.can_reserve(44, 8, empty_pool=True))

        # Fill all 20 physical pages with 20 FA pages (80 tokens, swa_tail_len=0).
        prefix = torch.tensor([0], dtype=torch.int64)
        seq = torch.tensor([80], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64)
        full_all = allocator.alloc_extend_swa_tail(
            prefix, prefix, seq, seq, last_loc, extend_num_tokens=80, swa_tail_len=0
        )
        self.assertEqual(allocator.full_available_size(), 0)
        self.assertEqual(allocator.swa_available_size(), 0)

        # Check PoolStatsObserver conservation identity when all 80 FA tokens are in radix cache.
        (full_cap, full_avail), (swa_cap, swa_avail) = (
            allocator.swa_capacity_and_available(full_capacity=80, swa_capacity=80)
        )
        self.assertEqual(full_cap - (full_avail + 80), 0)
        self.assertEqual(swa_cap - (swa_avail + 0), 0)

        # Fake tree_cache that frees chunks from full_all when evict() is called.
        remaining = [full_all]
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = True
        tree_cache.full_evictable_size.side_effect = lambda: int(remaining[0].numel())
        tree_cache.swa_evictable_size.return_value = 0

        def _fake_evict(params):
            n = min(int(params.num_tokens), int(remaining[0].numel()))
            n = ((n + 3) // 4) * 4
            to_free, remaining[0] = remaining[0][:n], remaining[0][n:]
            allocator.free_full(to_free)

        tree_cache.evict.side_effect = _fake_evict
        tree_cache.evict_for_alloc.side_effect = _fake_evict

        # Reclaim room for 4 FA tokens (1 page) + 4 SWA tokens (5 pages) -> needs exactly 6 pages (24 FA tokens)!
        err = allocator.reclaim_for_prealloc(tree_cache, full_tokens=4, swa_tokens=4)
        self.assertIsNone(err)
        self.assertTrue(allocator.new_pages_available(1, 1))
        self.assertEqual(int(remaining[0].numel()), 80 - 24)
        allocator.free_full(remaining[0])

    def test_kv_free_fence_and_host_return_allocators(self):
        _, allocator = self._make_gemma4_pool_and_allocator(size=24, page_size=4)
        self.assertEqual(
            host_return_allocators(allocator),
            [allocator.full_attn_allocator, allocator.swa_attn_allocator],
        )

        fence = KVFreeFence(synchronize_device=lambda: None)
        fence.attach(allocator.full_attn_allocator)
        fence.attach(allocator.swa_attn_allocator)

        # Allocate 1 SWA logical page (5 phys pages) and 1 FA page (1 phys page) = all 6 pages.
        swa_tokens = allocator.swa_attn_allocator.alloc(4)
        fa_tokens = allocator.full_attn_allocator.alloc(4)
        self.assertEqual(allocator._page_store.unheld_free_pages(), 0)

        # Form a batch and launch a fake forward so freed pages are held.
        fence.on_batch_formed()
        result = MagicMock(copy_done=None)
        fence.on_forward_launched(result)

        allocator.swa_attn_allocator.free(swa_tokens)
        allocator.full_attn_allocator.free(fa_tokens)
        # SWA releases 5 physical pages (primary + 4 companions), FA releases 1 -> 6 held pages.
        self.assertEqual(allocator._page_store.held_pages.num_pages, 6)
        self.assertEqual(allocator._page_store.unheld_free_pages(), 0)

        # Complete forward -> all 6 physical pages are unheld and free again.
        fence.on_forward_done(result)
        self.assertEqual(allocator._page_store.unheld_free_pages(), 6)

    def test_gcd_multi_full_and_swa_groups_allocator(self):
        # 8 Full layers, 20 SWA layers -> gcd = 4 buffers, M_full = 2, M_swa = 5.
        # Pool has 14 physical pages (size=56, page_size=4).
        # 1 joint (Full + SWA) logical page takes M_full + M_swa = 2 + 5 = 7 physical pages.
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=56,
            size_swa=56,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=list(range(8, 28)),
            full_attention_layer_ids=list(range(8)),
            device="cpu",
        )
        allocator = TpuUnifiedSWATokenToKVPoolAllocator(
            size=56,
            size_swa=56,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=True,
        )
        self.assertEqual(allocator._full_groups, 2)
        self.assertEqual(allocator._swa_groups, 5)
        self.assertEqual(allocator.full_available_size(), 28)  # 14 // 2 * 4
        self.assertEqual(allocator.swa_available_size(), 8)  # 14 // 5 * 4
        self.assertEqual(allocator.max_extend_tokens(), 8)  # 14 // (2 + 5) * 4

        fa_tokens = allocator.full_attn_allocator.alloc(4)
        swa_tokens = allocator.swa_attn_allocator.alloc(4)
        self.assertIsNotNone(fa_tokens)
        self.assertIsNotNone(swa_tokens)
        fa_primary = int(fa_tokens[0].item()) // 4
        swa_primary = int(swa_tokens[0].item()) // 4
        fa_all = [fa_primary] + pool.full_group_pages[:, fa_primary].tolist()
        swa_all = [swa_primary] + pool.swa_group_pages[:, swa_primary].tolist()
        self.assertEqual(len(fa_all), 2)
        self.assertEqual(len(swa_all), 5)
        self.assertEqual(len(set(fa_all + swa_all)), 7)
        self.assertEqual(allocator._page_store.num_free_pages(), 7)

        allocator.full_attn_allocator.free(fa_tokens)
        allocator.swa_attn_allocator.free(swa_tokens)
        self.assertEqual(allocator._page_store.num_free_pages(), 14)

    def test_unified_kv_eviction_policies(self):
        from sglang.srt.environ import envs

        # 8 Full layers, 20 SWA layers -> G = 4, M_full = 2, M_swa = 5.
        # Pool size = 56 tokens (page_size = 4 -> 14 physical pages).
        # Static full share = 16 tokens (4 Full pages -> 4 * 2 = 8 physical pages),
        # so SWA share = 14 - 8 = 6 physical pages.
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=56,
            size_swa=56,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=list(range(8, 28)),
            full_attention_layer_ids=list(range(8)),
            device="cpu",
        )
        # With default SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1, policy is full_first:
        default_alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=56,
            size_swa=56,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=True,
            static_full_tokens=16,
        )
        self.assertEqual(default_alloc._eviction_policy, "full_first")

        # With SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=0, policy is compact:
        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(False):
            allocator = TpuUnifiedSWATokenToKVPoolAllocator(
                size=56,
                size_swa=56,
                page_size=page_size,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=pool,
                need_sort=True,
                static_full_tokens=16,
            )
        self.assertEqual(allocator._eviction_policy, "compact")
        self.assertEqual(allocator._full_share, 8)
        self.assertEqual(allocator._swa_share, 6)

        # Allocate all 14 physical pages as 7 Full logical pages (28 tokens):
        # Full holds 14 physical pages (> _full_share=8), SWA holds 0 (< _swa_share=6).
        fa_all = allocator.full_attn_allocator.alloc(28)
        self.assertEqual(allocator._free_page_count(), 0)

        calls = []
        remaining = [fa_all]
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = True

        def _on_evict(params):
            calls.append((params.num_tokens, params.swa_num_tokens))
            n = min(int(params.num_tokens), int(remaining[0].numel()))
            n = ((n + 3) // 4) * 4
            if n > 0:
                to_free, remaining[0] = remaining[0][:n], remaining[0][n:]
                allocator.full_attn_allocator.free(to_free)

        tree_cache.evict.side_effect = _on_evict
        tree_cache.evict_for_alloc.side_effect = _on_evict

        # Under static_shadow / compact, since Full overshoots its static share (14 > 8)
        # while SWA is under its share (0 + 5 <= 6), only Full is evicted!
        ok = allocator.evict_to_free_tokens(tree_cache, 4, swa_num_tokens=4)
        self.assertTrue(ok)
        # Need M_full + M_swa = 2 + 5 = 7 physical pages -> ceil(7 / 2) = 4 Full pages = 16 tokens:
        self.assertEqual(calls[0], (16, 0))
        allocator.full_attn_allocator.free(remaining[0])


if __name__ == "__main__":
    unittest.main()
