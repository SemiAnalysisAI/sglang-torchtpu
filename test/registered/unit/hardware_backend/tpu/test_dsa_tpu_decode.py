"""TPU DSA decode helpers: indexer top-k, sparse latent attention, tail-page metadata."""

import unittest
from unittest import mock

import numpy as np
import torch

from sglang.srt.hardware_backend.tpu import dsa_indexer_tpu
from sglang.srt.hardware_backend.tpu.attention_pallas_mla import (
    _dsa_page_level_metadata,
    _dsa_tail_page_metadata,
    dsa_sparse_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _torch_topk(values, k):
    return torch.topk(values, k, dim=-1)


class TestDecodeTopk(CustomTestCase):
    def _run(self, kv_lens, k):
        torch.manual_seed(0)
        b, h, d, s, rows = len(kv_lens), 4, 16, 12, 64
        query = torch.randn(b, h, d)
        weights = torch.rand(b, h)
        index_k = torch.randn(rows, d)
        slots = torch.stack([torch.randperm(rows)[:s] for _ in range(b)])
        kv = torch.tensor(kv_lens)
        with mock.patch("sglang.srt.hardware_backend.tpu.top_k.tpu_top_k", _torch_topk):
            got = dsa_indexer_tpu.decode_topk(query, weights, index_k, slots, kv, k)
        for i in range(b):
            keys = index_k[slots[i]].numpy()
            score = (
                np.maximum(query[i].numpy() @ keys.T, 0) * weights[i].numpy()[:, None]
            ).sum(0)
            valid = list(range(kv_lens[i]))
            want = set(sorted(valid, key=lambda p: -score[p])[:k])
            picked = [p for p in got[i].tolist() if p >= 0]
            self.assertEqual(set(picked), want)
            self.assertEqual(len(picked), min(k, kv_lens[i]))
        return got

    def test_selects_top_scoring_valid_positions(self):
        self._run([12, 7, 9], k=5)

    def test_short_context_selects_every_position(self):
        got = self._run([3, 5], k=8)
        self.assertEqual(sorted(p for p in got[0].tolist() if p >= 0), [0, 1, 2])
        self.assertEqual((got[1] < 0).sum().item(), 3)


class TestSparseAttention(CustomTestCase):
    def test_all_selected_equals_dense_attention(self):
        torch.manual_seed(1)
        b, h, k, rank, rope = 2, 3, 6, 8, 4
        q = torch.randn(b, h, rank + rope)
        rows = torch.randn(b, k, rank + rope)
        topk = torch.arange(k).repeat(b, 1)
        got = dsa_sparse_attention(q, rows, topk, 0.5, rank)
        probs = torch.softmax(torch.einsum("bhd,bkd->bhk", q, rows) * 0.5, dim=-1)
        want = torch.einsum("bhk,bkd->bhd", probs, rows[..., :rank])
        torch.testing.assert_close(got, want)

    def test_unselected_rows_are_ignored(self):
        torch.manual_seed(2)
        q = torch.randn(1, 2, 12)
        rows = torch.randn(1, 4, 12)
        topk = torch.tensor([[2, 0, -1, -1]])
        got = dsa_sparse_attention(q, rows, topk, 1.0, 8)
        sub = rows[:, :2]
        probs = torch.softmax(torch.einsum("bhd,bkd->bhk", q, sub), dim=-1)
        torch.testing.assert_close(
            got, torch.einsum("bhk,bkd->bhd", probs, sub[..., :8])
        )

    def test_nothing_selected_is_finite(self):
        out = dsa_sparse_attention(
            torch.randn(1, 2, 12), torch.randn(1, 3, 12), torch.full((1, 3), -1), 1.0, 8
        )
        self.assertTrue(torch.isfinite(out).all())


class TestTailPageMetadata(CustomTestCase):
    def test_points_each_sequence_at_its_newest_page(self):
        page_size, pages_per_seq = 4, 3
        page_indices = torch.tensor(
            [10, 11, 12, 20, 21, 22, 30, 31, 32], dtype=torch.int32
        )
        kv_lens = torch.tensor([9, 4, 0], dtype=torch.int32)
        lens, table = _dsa_tail_page_metadata(kv_lens, page_indices, page_size)
        table = table.view(3, pages_per_seq)
        # seq 0: token 8 is on page slot 2 -> page 12, offset 0 -> len 1.
        self.assertEqual(table[0, 0].item(), 12)
        self.assertEqual(lens[0].item(), 1)
        # seq 1: token 3 is on page slot 0 -> page 20, offset 3 -> len 4.
        self.assertEqual(table[1, 0].item(), 20)
        self.assertEqual(lens[1].item(), 4)
        # empty sequence stays empty.
        self.assertEqual(lens[2].item(), 0)


class TestPageLevelMetadata(CustomTestCase):
    def test_selected_pages_then_newest_page_on_its_true_slot(self):
        page_size, pages_per_seq = 4, 5
        page_indices = torch.tensor(
            [10, 11, 12, 13, 14, 20, 21, 22, 23, 24], dtype=torch.int32
        )
        # seq 0: newest token 17 (page 4, offset 1); picks pages 3, 0, 4, 0.
        # seq 1: newest token 5 (page 1, offset 1); picks only its own page.
        kv_lens = torch.tensor([18, 6], dtype=torch.int32)
        topk = torch.tensor([[13, 2, 17, 1], [5, 4, -1, -1]], dtype=torch.int32)
        lens, table = _dsa_page_level_metadata(topk, kv_lens, page_indices, page_size)
        table = table.view(2, pages_per_seq)
        self.assertEqual(table[0, :3].tolist(), [10, 13, 14])
        self.assertEqual(lens[0].item(), 2 * page_size + 2)
        self.assertEqual(table[1, 0].item(), 21)
        self.assertEqual(lens[1].item(), 2)


if __name__ == "__main__":
    unittest.main()
