# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.unified_kv_pool import (
    TpuUnifiedSWAKVPool,
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


if __name__ == "__main__":
    unittest.main()
