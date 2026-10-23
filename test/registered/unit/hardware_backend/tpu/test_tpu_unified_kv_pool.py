# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.allocator_tpu import host_return_allocators
from sglang_tpu.kv_free_fence import KVFreeFence
from sglang_tpu.tpu_platform import TpuSRTPlatform, _resolve_unified_kv_pool
from sglang_tpu.unified_kv_pool import (
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWAPoolConfigurator,
    TpuUnifiedSWATokenToKVPoolAllocator,
    merge_swa_kv_heads,
    num_full_groups,
    num_shared_kv_buffers,
    num_swa_groups,
    split_swa_output,
    unified_kv_pool_enabled,
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


class TestTpuUnifiedPoolWiring(unittest.TestCase):
    """Tests for PR 3: flag resolution, feature blocking, configurator sizing (bf16 & fp8), and RPA wiring."""

    def _make_kvc(
        self,
        dtype: torch.dtype = torch.bfloat16,
        dtype_str: str = "bfloat16",
        full_ids: list[int] | None = None,
        swa_ids: list[int] | None = None,
        page_size: int = 64,
    ):
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        if full_ids is None:
            full_ids = list(range(10))
        if swa_ids is None:
            swa_ids = list(range(10, 60))
        kvc = MagicMock()
        kvc.is_hybrid_swa = True
        kvc.use_mla_backend = False
        kvc.is_hybrid_gdn = False
        kvc.has_mamba_state = False
        kvc.is_draft_worker = False
        kvc.spec_algorithm.is_none.return_value = True
        kvc.spec_algorithm.is_eagle.return_value = False
        kvc.spec_algorithm.is_standalone.return_value = False
        kvc.spec_algorithm.is_dflash_family.return_value = False
        kvc.dcp_enabled = False
        kvc.pp_size = 1
        kvc.token_to_kv_pool_class = TpuFusedMHATokenToKVPool
        kvc.custom_mem_pool_head_num = None
        kvc.custom_mem_pool_head_dim = None
        kvc.kv_cache_dtype = dtype
        kvc.kv_cache_dtype_str = dtype_str
        kvc.pool_page_size = page_size
        kvc.page_size = page_size
        kvc.sliding_window_size = 1024
        kvc.attn_dp_size = 1
        kvc.layer_info.full_attention_layer_ids = full_ids
        kvc.layer_info.swa_attention_layer_ids = swa_ids
        kvc.model_config.attention_arch = None
        kvc.model_config.get_num_kv_heads.return_value = 1
        kvc.model_config.head_dim = 512
        kvc.model_config.v_head_dim = 512
        kvc.model_config.get_swa_num_kv_heads.return_value = 2
        kvc.model_config.swa_head_dim = 256
        kvc.model_config.swa_v_head_dim = 256
        return kvc

    def test_resolve_unified_kv_pool_flag_and_validation(self):
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy",
            enable_unified_memory=True,
            attention_backend="pallas_tpu_rpa_v3",
        )
        _resolve_unified_kv_pool(args)
        view = resolving_view(args)
        self.assertFalse(view.enable_unified_memory)
        self.assertTrue(view.tpu_unified_kv_pool)

        invalid_kwargs_list = [
            {"attention_backend": "pallas_tpu_flash"},
            {"pp_size": 2},
            {"dcp_size": 2},
            {"attn_cp_size": 2},
            {"enable_prefill_cp": True},
            {"disaggregation_mode": "prefill"},
            {"speculative_algorithm": "EAGLE3"},
            {"enable_hierarchical_cache": True, "hicache_io_backend": "direct"},
            {"enable_unified_cache_external_linker": True},
            {"page_size": 1},
        ]
        for bad_kwargs in invalid_kwargs_list:
            with self.subTest(bad_kwargs=bad_kwargs):
                bad_args = ServerArgs(
                    model_path="dummy",
                    enable_unified_memory=True,
                    attention_backend=bad_kwargs.get(
                        "attention_backend", "pallas_tpu_rpa_batched"
                    ),
                    **{
                        k: v
                        for k, v in bad_kwargs.items()
                        if k != "attention_backend"
                    },
                )
                with self.assertRaises(ValueError):
                    _resolve_unified_kv_pool(bad_args)

    def test_unified_kv_pool_enabled_feature_blocks(self):
        from unittest.mock import patch

        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool

        parallel_ok = MagicMock(
            attn_tp_size=8, attn_dcp_size=1, pp_size=1, attn_cp_size=1
        )
        disagg_ok = MagicMock(disaggregation_mode="null")
        mem_on = MagicMock(
            tpu_unified_kv_pool=True,
            enable_hierarchical_cache=False,
            hicache_io_backend="raiden",
            enable_unified_cache_external_linker=False,
            enable_unified_memory=False,
            disable_radix_cache=False,
        )
        mem_off = MagicMock(tpu_unified_kv_pool=False)

        with patch("sglang_tpu.unified_kv_pool.get_memory", return_value=mem_off):
            self.assertFalse(unified_kv_pool_enabled(self._make_kvc()))

        with (
            patch("sglang_tpu.unified_kv_pool.get_memory", return_value=mem_on),
            patch("sglang_tpu.unified_kv_pool.get_parallel", return_value=parallel_ok),
            patch("sglang_tpu.unified_kv_pool.get_disagg", return_value=disagg_ok),
        ):
            self.assertTrue(unified_kv_pool_enabled(self._make_kvc()))

            # Non-SWA, MLA, GDN, draft worker, SeqAlongLane, page_size=1, mxint8:
            for attr, val in [
                ("is_hybrid_swa", False),
                ("use_mla_backend", True),
                ("is_hybrid_gdn", True),
                ("is_draft_worker", True),
                ("dcp_enabled", True),
                ("pp_size", 2),
                ("token_to_kv_pool_class", TpuSeqAlongLaneKVPool),
                ("pool_page_size", 1),
                ("custom_mem_pool_head_num", 4),
            ]:
                kvc_bad = self._make_kvc()
                setattr(kvc_bad, attr, val)
                with self.assertRaises(ValueError):
                    unified_kv_pool_enabled(kvc_bad)

            kvc_mxint8 = self._make_kvc(dtype_str="mxint8")
            with self.assertRaises(NotImplementedError):
                unified_kv_pool_enabled(kvc_mxint8)

    def test_configurator_gemma4_bf16_and_fp8_sizing(self):
        from unittest.mock import patch

        page_size = 64
        parallel_mock = MagicMock(
            attn_tp_size=8, attn_dcp_size=1, pp_size=1, attn_cp_size=1
        )
        disagg_mock = MagicMock(disaggregation_mode="null")
        mem_mock = MagicMock(
            tpu_unified_kv_pool=True,
            enable_hierarchical_cache=False,
            hicache_io_backend="raiden",
            enable_unified_cache_external_linker=False,
            enable_unified_memory=False,
            disable_radix_cache=False,
        )
        sched_mock = MagicMock(
            swa_full_tokens_ratio=0.5,
            chunked_prefill_size=4096,
            disable_overlap_schedule=False,
            max_running_requests=None,
        )

        exec_mock = MagicMock()
        exec_mock.kernel.attention_backend = "pallas_tpu_rpa_v3"

        with (
            patch("sglang.srt.runtime_context.get_exec", return_value=exec_mock),
            patch("sglang_tpu.unified_kv_pool.get_parallel", return_value=parallel_mock),
            patch("sglang_tpu.unified_kv_pool.get_disagg", return_value=disagg_mock),
            patch("sglang_tpu.unified_kv_pool.get_memory", return_value=mem_mock),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_parallel",
                return_value=parallel_mock,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_disagg",
                return_value=disagg_mock,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_memory",
                return_value=mem_mock,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_schedule",
                return_value=sched_mock,
            ),
        ):
            cfg_bf16 = TpuUnifiedSWAPoolConfigurator(
                self._make_kvc(torch.bfloat16, "bfloat16")
            )
            cfg_fp8 = TpuUnifiedSWAPoolConfigurator(
                self._make_kvc(torch.float8_e4m3fn, "fp8_e4m3")
            )
            cfg_gcd = TpuUnifiedSWAPoolConfigurator(
                self._make_kvc(
                    torch.bfloat16,
                    "bfloat16",
                    full_ids=list(range(8)),
                    swa_ids=list(range(8, 28)),
                )
            )

            self.assertEqual(cfg_bf16._num_shared_buffers, 10)
            self.assertEqual(cfg_bf16._num_full_groups, 1)
            self.assertEqual(cfg_bf16._num_swa_groups, 5)
            self.assertEqual(cfg_gcd._num_shared_buffers, 4)
            self.assertEqual(cfg_gcd._num_full_groups, 2)
            self.assertEqual(cfg_gcd._num_swa_groups, 5)

            # In FP8, SWA tokens are priced at the shared buffer's row byte width
            # (_full_per_token) so merged SWA heads never overflow available_bytes!
            self.assertEqual(cfg_fp8._split._swa_per_token, cfg_fp8._split._full_per_token)

            budget_bytes = 512 * 1024 * 1024  # 512 MiB
            res_bf16 = cfg_bf16.calculate_pool_sizes(budget_bytes, page_size)
            res_fp8 = cfg_fp8.calculate_pool_sizes(budget_bytes, page_size)
            res_gcd = cfg_gcd.calculate_pool_sizes(budget_bytes, page_size)

            for cfg, res in (
                (cfg_bf16, res_bf16),
                (cfg_fp8, res_fp8),
                (cfg_gcd, res_gcd),
            ):
                self.assertEqual(
                    res.full_max_total_num_tokens, res.swa_max_total_num_tokens
                )
                self.assertLessEqual(
                    res.max_total_num_tokens, res.full_max_total_num_tokens
                )
                total_pages = res.full_max_total_num_tokens // page_size
                self.assertEqual(total_pages % 8, 0)
                self.assertGreater(total_pages, 0)
                actual_bytes = (total_pages + 1) * cfg._page_bytes
                self.assertLessEqual(actual_bytes, budget_bytes)

            # Reject a max_tokens cap too small for one sliding window + 1 page:
            with self.assertRaises(ValueError):
                cfg_bf16.calculate_pool_sizes_from_max_tokens(512, page_size)

    def test_platform_hooks_and_rpa_layer_page_indices(self):
        from sglang_tpu.attention_pallas_rpa_base import (
            BaseTpuPallasRpaAttnBackend,
            PallasForwardMetadata,
        )

        platform = TpuSRTPlatform()
        self.assertIs(platform.get_unified_kv_pool_cls(), TpuUnifiedSWAKVPool)
        self.assertIs(
            platform.get_unified_kv_pool_allocator_cls(),
            TpuUnifiedSWATokenToKVPoolAllocator,
        )
        self.assertIs(
            platform.get_unified_kv_pool_configurator_cls(),
            TpuUnifiedSWAPoolConfigurator,
        )

        # Verify BaseTpuPallasRpaAttnBackend._layer_page_indices for M_full=2, M_swa=5:
        backend = object.__new__(BaseTpuPallasRpaAttnBackend)
        backend.token_to_kv_pool = MagicMock(swa_raw_head_num=2)
        backend._swa_kv_heads_merged = True
        backend._full_layer_groups = {0: 0, 4: 1}
        backend._swa_layer_groups = {1: 0, 5: 1, 9: 2}
        backend.forward_metadata = PallasForwardMetadata(
            kv_lens=torch.tensor([4], dtype=torch.int32),
            cu_q_lens=torch.tensor([0, 4], dtype=torch.int32),
            cu_kv_lens=torch.tensor([0, 4], dtype=torch.int32),
            page_indices=torch.tensor([10], dtype=torch.int32),
            distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
            active_num_tokens=torch.tensor(4, dtype=torch.int32),
            swa_page_indices=torch.tensor([20], dtype=torch.int32),
            full_group_page_indices=torch.tensor([[11]], dtype=torch.int32),
            swa_group_page_indices=torch.tensor([[21], [22]], dtype=torch.int32),
        )
        self.assertEqual(
            backend._layer_page_indices(0, is_swa_layer=False),
            (backend.forward_metadata.page_indices, None),
        )
        self.assertTrue(
            torch.equal(
                backend._layer_page_indices(4, is_swa_layer=False)[0],
                torch.tensor([11], dtype=torch.int32),
            )
        )
        self.assertEqual(
            backend._layer_page_indices(1, is_swa_layer=True),
            (backend.forward_metadata.swa_page_indices, 2),
        )
        self.assertTrue(
            torch.equal(
                backend._layer_page_indices(9, is_swa_layer=True)[0],
                torch.tensor([22], dtype=torch.int32),
            )
        )

    def test_import_order_isolation(self):
        import subprocess
        import sys

        code = (
            "import sys, sglang_tpu.attention_pallas_rpa_base; "
            "assert 'sglang_tpu.kv_pool' not in sys.modules; "
            "assert 'sglang_tpu.unified_kv_pool' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
