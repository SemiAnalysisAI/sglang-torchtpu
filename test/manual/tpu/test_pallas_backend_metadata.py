"""Unit tests for ``PallasAttnBackend._build_metadata``.

These tests exercise the metadata builder on CPU and don't require TPU
hardware or torch_tpu. They cover the layout invariants the Pallas
kernel relies on, and the *batch-axis bucket padding* introduced to keep
``torch.compile`` from re-tracing per batch size:

* Every metadata tensor is padded to the next ``SEQ_BUCKETS`` size; real
  entries occupy ``[:real_bs]``, padded entries are zero.
* ``cu_q_lens`` cumulates query lengths (1 per real seq + 0 for padded;
  ``extend_seq_lens`` per real seq + 0 for padded in extend mode).
* ``cu_kv_lens[i] = i * pages_per_seq * page_size`` — the kernel reads
  ``page_indices[i * pages_per_seq + j]`` for token ``j`` of seq ``i``.
* ``page_indices`` for padded seqs points into the dummy padding block
  (req_pool_indices padded to 0 → req_to_token[0] is all-zero).
* ``distribution`` is ``[N, N, N]`` for decode and ``[0, N, N]`` for extend,
  where ``N`` is the number of active (non-padded) sequences.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.pallas_backend import (
    SEQ_BUCKETS,
    PallasAttnBackend,
    PallasForwardMetadata,
    _next_bucket,
)
from sglang.test.test_utils import CustomTestCase


class _DummyForwardMode:
    def __init__(self, mode: str):
        self._mode = mode

    def is_decode(self) -> bool:
        return self._mode == "decode"

    def is_extend(self) -> bool:
        return self._mode == "extend"

    def is_idle(self) -> bool:
        return self._mode == "idle"


def _make_backend(page_size: int, max_context_len: int) -> PallasAttnBackend:
    """Build a backend with a stubbed ModelRunner — CPU-only.

    Row 0 of ``req_to_token`` is left all-zero to mimic the real
    ``ReqToTokenPool`` convention (reserved padding row), so any
    bucket-padded ``req_pool_indices=0`` slot maps to block 0.
    """
    req_to_token = torch.arange(
        16 * max_context_len, dtype=torch.int32
    ).reshape(16, max_context_len)
    req_to_token[0] = 0  # reserved padding row
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    runner = SimpleNamespace(
        device="cpu",
        page_size=page_size,
        req_to_token_pool=req_to_token_pool,
    )
    return PallasAttnBackend(runner)


def _make_forward_batch(
    *,
    forward_mode: str,
    seq_lens: list[int],
    req_pool_indices: list[int],
    extend_seq_lens: list[int] | None = None,
):
    return SimpleNamespace(
        forward_mode=_DummyForwardMode(forward_mode),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        extend_seq_lens=(
            torch.tensor(extend_seq_lens, dtype=torch.int32)
            if extend_seq_lens is not None
            else None
        ),
    )


class TestPallasMetadataDecode(CustomTestCase):
    def setUp(self):
        self.page_size = 4
        self.max_context_len = 16
        self.pages_per_seq = self.max_context_len // self.page_size  # 4
        self.backend = _make_backend(self.page_size, self.max_context_len)

    def test_decode_q_lens_are_ones_real_zero_padded(self):
        """Real slots get ``extend_seq_lens=1`` (decode), padded slots get 0."""
        real_bs = 3
        seq_lens = [8, 12, 4]
        fb = _make_forward_batch(
            forward_mode="decode",
            seq_lens=seq_lens,
            req_pool_indices=[3, 7, 1],
        )
        md = self.backend._build_metadata(fb)
        padded_bs = _next_bucket(real_bs, SEQ_BUCKETS)  # 4

        # cu_q_lens is shape [padded_bs + 1]; first ``real_bs+1`` entries
        # are 0..real_bs, padded entries stay at real_bs.
        self.assertEqual(md.cu_q_lens.shape, (padded_bs + 1,))
        expected = torch.zeros(padded_bs + 1, dtype=torch.int32)
        expected[1 : real_bs + 1] = torch.arange(1, real_bs + 1, dtype=torch.int32)
        expected[real_bs + 1 :] = real_bs
        torch.testing.assert_close(md.cu_q_lens, expected)

    def test_decode_kv_lens_real_then_padded(self):
        """Real ``seq_lens`` first, zeros for padded slots."""
        real_bs = 3
        seq_lens = [8, 12, 4]
        fb = _make_forward_batch(
            forward_mode="decode",
            seq_lens=seq_lens,
            req_pool_indices=[3, 7, 1],
        )
        md = self.backend._build_metadata(fb)
        padded_bs = _next_bucket(real_bs, SEQ_BUCKETS)
        self.assertEqual(md.kv_lens.shape, (padded_bs,))

        expected = torch.zeros(padded_bs, dtype=torch.int32)
        expected[:real_bs] = torch.tensor(seq_lens, dtype=torch.int32)
        torch.testing.assert_close(md.kv_lens, expected)

    def test_decode_cu_kv_lens_are_rectangular_for_padded_bs(self):
        real_bs = 3
        fb = _make_forward_batch(
            forward_mode="decode",
            seq_lens=[8, 12, 4],
            req_pool_indices=[3, 7, 1],
        )
        md = self.backend._build_metadata(fb)
        padded_bs = _next_bucket(real_bs, SEQ_BUCKETS)  # 4
        stride = self.pages_per_seq * self.page_size  # 16

        expected = torch.tensor(
            [i * stride for i in range(padded_bs + 1)], dtype=torch.int32
        )
        torch.testing.assert_close(md.cu_kv_lens, expected)

    def test_decode_page_indices_layout(self):
        """``page_indices[i * pages_per_seq + j]`` must match the j-th page of seq i.

        For real sequences, the table is computed from ``req_to_token[req_idx]``.
        For padded sequences (req_pool_indices padded to 0), the table points
        into block 0 — the fused KV pool's dummy padding block.
        """
        real_bs = 3
        req_pool_indices = [3, 7, 1]
        fb = _make_forward_batch(
            forward_mode="decode",
            seq_lens=[8, 12, 4],
            req_pool_indices=req_pool_indices,
        )
        md = self.backend._build_metadata(fb)
        padded_bs = _next_bucket(real_bs, SEQ_BUCKETS)
        self.assertEqual(md.page_indices.shape, (padded_bs * self.pages_per_seq,))

        req_to_token = self.backend.req_to_token_pool.req_to_token
        # Real entries
        for i, req_idx in enumerate(req_pool_indices):
            for j in range(self.pages_per_seq):
                expected = req_to_token[req_idx, j * self.page_size].item() // self.page_size
                got = md.page_indices[i * self.pages_per_seq + j].item()
                self.assertEqual(got, expected, msg=f"seq {i} page {j}")
        # Padded entries: req_to_token[0] is all-zero → page_indices = 0
        for i in range(real_bs, padded_bs):
            for j in range(self.pages_per_seq):
                got = md.page_indices[i * self.pages_per_seq + j].item()
                self.assertEqual(got, 0, msg=f"padded seq {i} page {j} not in block 0")

    def test_decode_distribution_counts_active_seqs(self):
        fb = _make_forward_batch(
            forward_mode="decode",
            seq_lens=[8, 0, 4],  # middle seq is padded
            req_pool_indices=[3, 7, 1],
        )
        md = self.backend._build_metadata(fb)
        # decode: [N, N, N] where N = number of nonzero seq_lens (independent
        # of bucket padding because we count only real-input entries).
        torch.testing.assert_close(
            md.distribution, torch.tensor([2, 2, 2], dtype=torch.int32)
        )

    def test_decode_metadata_shape_is_constant_across_batch_sizes(self):
        """All batches in the same SEQ_BUCKETS bin produce the same shapes."""
        shapes_seen = set()
        for batch_size in (1, 2, 3, 4):  # all map to bucket 1, 2, 4
            fb = _make_forward_batch(
                forward_mode="decode",
                seq_lens=[5] * batch_size,
                req_pool_indices=list(range(1, batch_size + 1)),
            )
            md = self.backend._build_metadata(fb)
            shapes_seen.add(
                (
                    tuple(md.kv_lens.shape),
                    tuple(md.cu_q_lens.shape),
                    tuple(md.cu_kv_lens.shape),
                    tuple(md.page_indices.shape),
                )
            )
        # We expect *at most* one shape per bucket (1, 2, 4) — so 3 unique shapes.
        self.assertLessEqual(len(shapes_seen), 3)
        # And the batch=3 case must be padded to bucket=4, sharing shapes with batch=4.
        bs3_fb = _make_forward_batch(
            forward_mode="decode", seq_lens=[5, 5, 5], req_pool_indices=[1, 2, 3]
        )
        bs4_fb = _make_forward_batch(
            forward_mode="decode", seq_lens=[5, 5, 5, 5], req_pool_indices=[1, 2, 3, 4]
        )
        md3 = self.backend._build_metadata(bs3_fb)
        md4 = self.backend._build_metadata(bs4_fb)
        self.assertEqual(md3.kv_lens.shape, md4.kv_lens.shape)
        self.assertEqual(md3.cu_kv_lens.shape, md4.cu_kv_lens.shape)
        self.assertEqual(md3.page_indices.shape, md4.page_indices.shape)


class TestPallasMetadataExtend(CustomTestCase):
    def setUp(self):
        self.page_size = 4
        self.max_context_len = 16
        self.pages_per_seq = self.max_context_len // self.page_size  # 4
        self.backend = _make_backend(self.page_size, self.max_context_len)

    def test_extend_cu_q_lens_use_extend_seq_lens(self):
        # batch_size=2 already matches the bucket size 2, so no padding here.
        fb = _make_forward_batch(
            forward_mode="extend",
            seq_lens=[8, 12],
            extend_seq_lens=[8, 4],  # seq 1 already has 8 cached prefix tokens
            req_pool_indices=[3, 7],
        )
        md = self.backend._build_metadata(fb)
        torch.testing.assert_close(
            md.cu_q_lens, torch.tensor([0, 8, 12], dtype=torch.int32)
        )

    def test_extend_with_padded_batch(self):
        """A 3-seq extend should pad to the bucket=4 batch shape."""
        real_bs = 3
        fb = _make_forward_batch(
            forward_mode="extend",
            seq_lens=[8, 12, 6],
            extend_seq_lens=[8, 4, 6],
            req_pool_indices=[3, 7, 1],
        )
        md = self.backend._build_metadata(fb)
        padded_bs = _next_bucket(real_bs, SEQ_BUCKETS)  # 4
        # cu_q_lens: real cumsum, then constant at sum(extend_seq_lens)
        total = 8 + 4 + 6
        expected = torch.tensor([0, 8, 12, 18, 18], dtype=torch.int32)
        self.assertEqual(md.cu_q_lens.shape, (padded_bs + 1,))
        torch.testing.assert_close(md.cu_q_lens, expected)
        # distribution counts real (non-zero seq_lens) sequences.
        torch.testing.assert_close(
            md.distribution, torch.tensor([0, 3, 3], dtype=torch.int32)
        )

    def test_extend_distribution_is_mixed_only(self):
        fb = _make_forward_batch(
            forward_mode="extend",
            seq_lens=[8, 12],
            extend_seq_lens=[8, 4],
            req_pool_indices=[3, 7],
        )
        md = self.backend._build_metadata(fb)
        # extend: [0, N, N] — kernel internally remaps to [0, 0, N] (MIXED only)
        # when chunk_prefill_size is None.
        torch.testing.assert_close(
            md.distribution, torch.tensor([0, 2, 2], dtype=torch.int32)
        )

    def test_extend_metadata_dtype_is_int32(self):
        fb = _make_forward_batch(
            forward_mode="extend",
            seq_lens=[8, 12],
            extend_seq_lens=[8, 4],
            req_pool_indices=[3, 7],
        )
        md = self.backend._build_metadata(fb)
        for name in ("kv_lens", "cu_q_lens", "cu_kv_lens", "page_indices", "distribution"):
            tensor = getattr(md, name)
            self.assertEqual(tensor.dtype, torch.int32, msg=f"{name} dtype")


class TestPallasMetadataIsPallasForwardMetadata(CustomTestCase):
    def test_returns_metadata_instance(self):
        backend = _make_backend(page_size=4, max_context_len=16)
        fb = _make_forward_batch(
            forward_mode="decode",
            seq_lens=[8, 12],
            req_pool_indices=[3, 7],
        )
        md = backend._build_metadata(fb)
        self.assertIsInstance(md, PallasForwardMetadata)


if __name__ == "__main__":
    unittest.main()
