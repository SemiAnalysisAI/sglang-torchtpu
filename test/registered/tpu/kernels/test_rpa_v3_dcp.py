"""Pure-JAX TPU tests for the decode-context-parallel additions to the Pallas
ragged-paged-attention v3 kernel.

The geometry is the gpt-oss-120b DCP2 decode shape: queries all-gathered across
the group to ``[T, 16, 64]``, two local KV heads, bf16, page size 128, attention
sinks, sliding window 128.

Cases:

* ``cache_only`` + ``return_lse`` on each rank's position-interleaved shard,
  merged across the two shards by LSE, equals a dense reference over the whole
  sequence -- with and without the window, with and without the sink fold --
  including a sequence whose shard is empty on one rank.
* The same on an extend prefix pass, where ``q_len > 1`` and every cached key
  precedes every query.
* ``attention_sink`` with ``return_lse``: the sink is inside the returned
  ``m + log(l)``, which is what makes the extend two-pass merge exact.
* ``skip_kv_write`` leaves the KV cache untouched while the output is unchanged.
* With every new flag left at its default the kernel still reproduces the
  in-repo ``ref_ragged_paged_attention`` over decode, extend, windowed and
  soft-capped shapes -- the arm TP8 and CI run.

The cache is laid out with the kernel's own ``get_kv_cache_shape`` /
``merge_kv`` helpers, so a change to the packed layout moves the test with it.
Requires JAX on a TPU device; skipped otherwise. No torch_tpu, no server.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=300, suite="stage-a-test-tpu-unit")

# Standalone raw-JAX kernel test, not a TorchTPU runtime process. SGLang
# platform discovery selects CPU-only JAX for export, so select the physical
# backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU device"

NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
HEAD_DIM = 64
PAGE_SIZE = 128
CP = 2
SLIDING_WINDOW = 128
SM_SCALE = 1.0 / float(np.sqrt(HEAD_DIM))
TOLERANCE = 0.05


def _round_to_bf16(array: np.ndarray) -> np.ndarray:
    """bf16-round a float32 array and return it as float32.

    Kernel and reference then consume identical numbers, so the comparison
    measures the kernel, not the cast.
    """
    import jax.numpy as jnp

    return np.asarray(jnp.asarray(array, jnp.bfloat16).astype(jnp.float32))


def _to_bf16(array: np.ndarray):
    import jax.numpy as jnp

    return jnp.asarray(array, jnp.bfloat16)


def _fused_cache(
    local_k: np.ndarray,
    local_v: np.ndarray,
    total_pages: int,
    num_kv_heads: int = NUM_KV_HEADS,
):
    """Pack per-slot K/V into the kernel's paged fused-cache layout."""
    import jax.numpy as jnp

    from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
        get_kv_cache_shape,
        merge_kv,
    )

    kv = merge_kv(_to_bf16(local_k), _to_bf16(local_v))
    cache = kv.reshape(total_pages, PAGE_SIZE, *kv.shape[1:])
    expected = get_kv_cache_shape(
        total_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM, jnp.bfloat16
    )
    assert cache.shape == tuple(expected), f"{cache.shape} != {tuple(expected)}"
    return cache


def _owned_positions(length: int, rank: int) -> list:
    return [p for p in range(length) if p % CP == rank]


def _dense_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    query_positions: np.ndarray,
    window: int | None,
    causal: bool = False,
):
    """Reference softmax attention of ``queries`` over ``keys``/``values``.

    queries: [q_len, NUM_Q_HEADS, HEAD_DIM]; keys/values: [kv_len, NUM_KV_HEADS,
    HEAD_DIM] indexed by absolute position. ``window`` keeps key ``p`` for query
    at position ``Q`` when ``p > Q - window``, and ``causal`` keeps it when
    ``p <= Q``. Returns (out, lse) in float32.
    """
    q_len = queries.shape[0]
    kv_len = keys.shape[0]
    out = np.zeros((q_len, NUM_Q_HEADS, HEAD_DIM), np.float32)
    lse = np.zeros((q_len, NUM_Q_HEADS), np.float32)
    key_positions = np.arange(kv_len)
    for i in range(q_len):
        keep = np.ones(kv_len, dtype=bool)
        if causal:
            keep &= key_positions <= int(query_positions[i])
        if window is not None:
            keep &= key_positions > int(query_positions[i]) - window
        for h in range(NUM_Q_HEADS):
            kv_head = h // Q_PER_KV
            scores = (keys[:, kv_head] @ queries[i, h]) * SM_SCALE
            scores = np.where(keep, scores, -np.inf)
            if not keep.any():
                lse[i, h] = -np.inf
                continue
            top = scores.max()
            weights = np.exp(scores - top)
            total = weights.sum()
            lse[i, h] = float(top + np.log(total))
            out[i, h] = (weights / total) @ values[:, kv_head]
    return out, lse


def _merge_by_lse(partials):
    """Combine per-rank (out, lse) pairs into the full-sequence (out, lse)."""
    lses = np.stack([lse for _, lse in partials])
    top = lses.max(axis=0)
    finite = np.isfinite(top)
    shifted = np.where(np.isfinite(lses), lses - np.where(finite, top, 0.0), -np.inf)
    weights = np.exp(shifted)
    total = weights.sum(axis=0)
    merged_lse = np.where(
        finite, top + np.log(np.where(total > 0, total, 1.0)), -np.inf
    )
    out = np.zeros_like(partials[0][0])
    for rank, (partial_out, _) in enumerate(partials):
        scale = np.where(total > 0, weights[rank] / total, 0.0)
        out += partial_out * scale[..., None]
    return out, merged_lse


def _fold_sink(out: np.ndarray, lse: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """The sink is a virtual key with logit ``sink`` and an all-zero value."""
    scale = 1.0 / (1.0 + np.exp(np.where(np.isfinite(lse), sinks[None, :] - lse, 0.0)))
    scale = np.where(np.isfinite(lse), scale, 0.0)
    return out * scale[..., None]


def _dense_attention_with_sink(queries, keys, values, query_positions, window, sinks):
    out, lse = _dense_attention(queries, keys, values, query_positions, window)
    return _fold_sink(out, lse, sinks)


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRpaV3DcpDecode(unittest.TestCase):
    """Two interleaved shards of one sequence merge back into the dense result."""

    SEQ_LENS = (300, 1, 173)  # global KV lengths; the middle one empties rank 1
    PAGES_PER_SEQ = 2
    TOTAL_PAGES = 8
    MAX_NUM_TOKENS = 8

    def setUp(self):
        rng = np.random.default_rng(1234)
        self.full_k = [
            _round_to_bf16(
                rng.standard_normal((length, NUM_KV_HEADS, HEAD_DIM), np.float32)
            )
            for length in self.SEQ_LENS
        ]
        self.full_v = [
            _round_to_bf16(
                rng.standard_normal((length, NUM_KV_HEADS, HEAD_DIM), np.float32)
            )
            for length in self.SEQ_LENS
        ]
        self.queries = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), np.float32
            )
        )
        self.sinks = _round_to_bf16(rng.standard_normal(NUM_Q_HEADS, np.float32))
        self.page_base = [s * self.PAGES_PER_SEQ for s in range(len(self.SEQ_LENS))]

    def _shard_arrays(self, rank):
        slots = self.TOTAL_PAGES * PAGE_SIZE
        local_k = np.zeros((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        local_v = np.zeros((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        local_lens = []
        for seq, length in enumerate(self.SEQ_LENS):
            owned = _owned_positions(length, rank)
            base = self.page_base[seq] * PAGE_SIZE
            for j, position in enumerate(owned):
                local_k[base + j] = self.full_k[seq][position]
                local_v[base + j] = self.full_v[seq][position]
            local_lens.append(len(owned))
        return local_k, local_v, local_lens

    def _run_shard(self, rank, window):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )

        local_k, local_v, local_lens = self._shard_arrays(rank)
        num_seqs = len(self.SEQ_LENS)
        page_indices = np.concatenate(
            [self.page_base[s] + np.arange(self.PAGES_PER_SEQ) for s in range(num_seqs)]
        ).astype(np.int32)
        empty_kv = np.zeros((self.MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), np.float32)

        out, _, lse = ragged_paged_attention(
            _to_bf16(self.queries),
            _to_bf16(empty_kv),
            _to_bf16(empty_kv),
            _fused_cache(local_k, local_v, self.TOTAL_PAGES),
            jnp.asarray(local_lens, jnp.int32),
            jnp.asarray(page_indices, jnp.int32),
            jnp.arange(num_seqs + 1, dtype=jnp.int32),
            jnp.asarray(
                [i * self.PAGES_PER_SEQ * PAGE_SIZE for i in range(num_seqs + 1)],
                jnp.int32,
            ),
            jnp.asarray([num_seqs, num_seqs, num_seqs], jnp.int32),
            None,  # custom_mask
            None,  # attention_sink: folded after the cross-rank merge
            sm_scale=SM_SCALE,
            sliding_window=window,
            cache_only=True,
            skip_kv_write=True,
            return_lse=True,
            cp_group_size=CP,
            cp_rank=jnp.asarray([rank], jnp.int32),
            kv_lens_global=jnp.asarray(self.SEQ_LENS, jnp.int32),
        )
        return (
            np.asarray(out.astype(jnp.float32))[:num_seqs],
            np.asarray(lse.astype(jnp.float32))[:num_seqs],
        )

    def _reference(self, window):
        out = np.zeros((len(self.SEQ_LENS), NUM_Q_HEADS, HEAD_DIM), np.float32)
        lse = np.zeros((len(self.SEQ_LENS), NUM_Q_HEADS), np.float32)
        for seq, length in enumerate(self.SEQ_LENS):
            seq_out, seq_lse = _dense_attention(
                self.queries[seq : seq + 1],
                self.full_k[seq],
                self.full_v[seq],
                np.asarray([length - 1]),
                window,
            )
            out[seq] = seq_out[0]
            lse[seq] = seq_lse[0]
        return out, lse

    def _assert_close(self, got, want, label):
        diff = float(np.abs(got - want).max())
        self.assertLess(diff, TOLERANCE, f"{label}: max abs diff {diff}")

    def test_merged_shards_match_the_dense_reference(self):
        for window in (None, SLIDING_WINDOW):
            with self.subTest(window=window):
                partials = [self._run_shard(rank, window) for rank in range(CP)]
                out, lse = _merge_by_lse(partials)
                ref_out, ref_lse = self._reference(window)
                self._assert_close(out, ref_out, "output")
                self._assert_close(lse, ref_lse, "lse")

    def test_sink_fold_after_the_merge_matches_the_dense_reference(self):
        for window in (None, SLIDING_WINDOW):
            with self.subTest(window=window):
                partials = [self._run_shard(rank, window) for rank in range(CP)]
                out, lse = _merge_by_lse(partials)
                folded = _fold_sink(out, lse, self.sinks)
                ref = np.zeros_like(folded)
                for seq, length in enumerate(self.SEQ_LENS):
                    ref[seq] = _dense_attention_with_sink(
                        self.queries[seq : seq + 1],
                        self.full_k[seq],
                        self.full_v[seq],
                        np.asarray([length - 1]),
                        window,
                        self.sinks,
                    )[0]
                self._assert_close(folded, ref, "sink-folded output")

    def test_empty_shard_row_returns_zero_output_and_minus_inf_lse(self):
        # Sequence 1 has one global token, which rank 0 owns; rank 1's shard is
        # empty and its q_len (1) exceeds its kv_len (0).
        out, lse = self._run_shard(1, SLIDING_WINDOW)
        self.assertTrue(np.all(out[1] == 0.0), out[1])
        self.assertTrue(np.all(np.isneginf(lse[1])), lse[1])
        # The sequence after the empty one must still be attended.
        self.assertTrue(np.all(np.isfinite(lse[2])))
        self.assertGreater(float(np.abs(out[2]).max()), 0.0)


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRpaV3DcpExtendPrefixPass(unittest.TestCase):
    """cache_only with q_len > 1: every cached key precedes every query."""

    PREFIX_LEN = 192
    CHUNK_LEN = 8
    PAGES_PER_SEQ = 1
    TOTAL_PAGES = 4
    MAX_NUM_TOKENS = 8

    def setUp(self):
        rng = np.random.default_rng(99)
        self.prefix_k = _round_to_bf16(
            rng.standard_normal((self.PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM), np.float32)
        )
        self.prefix_v = _round_to_bf16(
            rng.standard_normal((self.PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM), np.float32)
        )
        self.queries = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), np.float32
            )
        )

    def _run_shard(self, rank, window):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )

        owned = _owned_positions(self.PREFIX_LEN, rank)
        slots = self.TOTAL_PAGES * PAGE_SIZE
        local_k = np.zeros((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        local_v = np.zeros((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        for j, position in enumerate(owned):
            local_k[j] = self.prefix_k[position]
            local_v[j] = self.prefix_v[position]
        empty_kv = np.zeros((self.MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), np.float32)
        global_len = self.PREFIX_LEN + self.CHUNK_LEN

        out, _, lse = ragged_paged_attention(
            _to_bf16(self.queries),
            _to_bf16(empty_kv),
            _to_bf16(empty_kv),
            _fused_cache(local_k, local_v, self.TOTAL_PAGES),
            jnp.asarray([len(owned)], jnp.int32),
            jnp.asarray([0], jnp.int32),
            jnp.asarray([0, self.CHUNK_LEN], jnp.int32),
            jnp.asarray([0, self.PAGES_PER_SEQ * PAGE_SIZE], jnp.int32),
            jnp.asarray([0, 1, 1], jnp.int32),
            None,  # custom_mask
            None,  # attention_sink
            sm_scale=SM_SCALE,
            sliding_window=window,
            cache_only=True,
            skip_kv_write=True,
            return_lse=True,
            cp_group_size=CP,
            cp_rank=jnp.asarray([rank], jnp.int32),
            kv_lens_global=jnp.asarray([global_len], jnp.int32),
        )
        return (
            np.asarray(out.astype(jnp.float32))[: self.CHUNK_LEN],
            np.asarray(lse.astype(jnp.float32))[: self.CHUNK_LEN],
        )

    def test_merged_prefix_pass_matches_the_dense_prefix_reference(self):
        query_positions = self.PREFIX_LEN + np.arange(self.CHUNK_LEN)
        for window in (None, SLIDING_WINDOW):
            with self.subTest(window=window):
                partials = [self._run_shard(rank, window) for rank in range(CP)]
                out, lse = _merge_by_lse(partials)
                ref_out, ref_lse = _dense_attention(
                    self.queries[: self.CHUNK_LEN],
                    self.prefix_k,
                    self.prefix_v,
                    query_positions,
                    window,
                )
                self.assertLess(float(np.abs(out - ref_out).max()), TOLERANCE)
                self.assertLess(float(np.abs(lse - ref_lse).max()), TOLERANCE)


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRpaV3DcpFlagsOffPath(unittest.TestCase):
    """skip_kv_write and the cp_group_size=None default on the plain path."""

    KV_LEN = 200
    Q_LEN = 8
    PAGES_PER_SEQ = 2
    TOTAL_PAGES = 4
    MAX_NUM_TOKENS = 8

    def setUp(self):
        rng = np.random.default_rng(7)
        slots = self.TOTAL_PAGES * PAGE_SIZE
        self.cache_k = _round_to_bf16(
            rng.standard_normal((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        )
        self.cache_v = _round_to_bf16(
            rng.standard_normal((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        )
        self.queries = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), np.float32
            )
        )
        self.new_k = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), np.float32
            )
        )
        self.new_v = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), np.float32
            )
        )

    def _call(self, **kernel_kwargs):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )

        out, cache = ragged_paged_attention(
            _to_bf16(self.queries),
            _to_bf16(self.new_k),
            _to_bf16(self.new_v),
            _fused_cache(self.cache_k, self.cache_v, self.TOTAL_PAGES),
            jnp.asarray([self.KV_LEN], jnp.int32),
            jnp.asarray([0, 1], jnp.int32),
            jnp.asarray([0, self.Q_LEN], jnp.int32),
            jnp.asarray([0, self.PAGES_PER_SEQ * PAGE_SIZE], jnp.int32),
            jnp.asarray([0, 1, 1], jnp.int32),
            None,  # custom_mask
            None,  # attention_sink
            sm_scale=SM_SCALE,
            sliding_window=SLIDING_WINDOW,
            **kernel_kwargs,
        )
        return (
            np.asarray(out.astype(jnp.float32)),
            np.asarray(cache.astype(jnp.float32)),
        )

    def _input_cache(self):
        import jax.numpy as jnp

        return np.asarray(
            _fused_cache(self.cache_k, self.cache_v, self.TOTAL_PAGES).astype(
                jnp.float32
            )
        )

    def test_skip_kv_write_leaves_the_cache_unchanged(self):
        written_out, written_cache = self._call()
        skipped_out, skipped_cache = self._call(skip_kv_write=True)
        original = self._input_cache()

        self.assertTrue(np.array_equal(skipped_cache, original))
        self.assertFalse(np.array_equal(written_cache, original))
        # Fetch semantics are unchanged: the tail still comes from keys/values.
        self.assertTrue(np.array_equal(written_out, skipped_out))


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRpaV3SinkIsInsideTheLse(unittest.TestCase):
    """``return_lse`` reports ``log(e^sink + sum e^s)``, and the extend merge
    depends on it.

    The backend's extend path runs the cached prefix without a sink and the
    chunk with one, then joins them by log-sum-exp. That is only exact when the
    chunk pass folds the sink into both its output and its LSE, so both halves
    are asserted here, followed by the merge itself.
    """

    PREFIX_LEN = 192
    CHUNK_LEN = 8
    PAGES_PER_SEQ = 2
    TOTAL_PAGES = 4
    MAX_NUM_TOKENS = 8

    def setUp(self):
        rng = np.random.default_rng(2026)
        slots = self.TOTAL_PAGES * PAGE_SIZE
        self.cache_k = np.zeros((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        self.cache_v = np.zeros((slots, NUM_KV_HEADS, HEAD_DIM), np.float32)
        self.prefix_k = _round_to_bf16(
            rng.standard_normal((self.PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM), np.float32)
        )
        self.prefix_v = _round_to_bf16(
            rng.standard_normal((self.PREFIX_LEN, NUM_KV_HEADS, HEAD_DIM), np.float32)
        )
        self.cache_k[: self.PREFIX_LEN] = self.prefix_k
        self.cache_v[: self.PREFIX_LEN] = self.prefix_v
        self.chunk_k = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), np.float32
            )
        )
        self.chunk_v = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), np.float32
            )
        )
        self.queries = _round_to_bf16(
            rng.standard_normal(
                (self.MAX_NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), np.float32
            )
        )
        self.sinks = _round_to_bf16(rng.standard_normal(NUM_Q_HEADS, np.float32))

    def _call(self, *, kv_len, keys, values, sinks, cache_only):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )

        out, _, lse = ragged_paged_attention(
            _to_bf16(self.queries),
            _to_bf16(keys),
            _to_bf16(values),
            _fused_cache(self.cache_k, self.cache_v, self.TOTAL_PAGES),
            jnp.asarray([kv_len], jnp.int32),
            jnp.asarray(np.arange(self.PAGES_PER_SEQ), jnp.int32),
            jnp.asarray([0, self.CHUNK_LEN], jnp.int32),
            jnp.asarray([0, self.PAGES_PER_SEQ * PAGE_SIZE], jnp.int32),
            jnp.asarray([0, 1, 1], jnp.int32),
            None,  # custom_mask
            None if sinks is None else jnp.asarray(sinks, jnp.float32),
            sm_scale=SM_SCALE,
            cache_only=cache_only,
            skip_kv_write=True,
            return_lse=True,
        )
        return (
            np.asarray(out.astype(jnp.float32))[: self.CHUNK_LEN],
            np.asarray(lse.astype(jnp.float32))[: self.CHUNK_LEN],
        )

    def _prefix_pass(self):
        # Every cached key precedes every query, so the whole prefix is read.
        empty = np.zeros((self.MAX_NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), np.float32)
        return self._call(
            kv_len=self.PREFIX_LEN,
            keys=empty,
            values=empty,
            sinks=None,
            cache_only=True,
        )

    def _chunk_pass(self):
        # kv_len == q_len: nothing is read from the cache, the sink is counted
        # here, and the mask is the chunk-local causal one.
        return self._call(
            kv_len=self.CHUNK_LEN,
            keys=self.chunk_k,
            values=self.chunk_v,
            sinks=self.sinks,
            cache_only=False,
        )

    def _chunk_reference(self):
        return _dense_attention(
            self.queries[: self.CHUNK_LEN],
            self.chunk_k[: self.CHUNK_LEN],
            self.chunk_v[: self.CHUNK_LEN],
            np.arange(self.CHUNK_LEN),
            None,
            causal=True,
        )

    def _full_reference(self):
        keys = np.concatenate([self.prefix_k, self.chunk_k[: self.CHUNK_LEN]])
        values = np.concatenate([self.prefix_v, self.chunk_v[: self.CHUNK_LEN]])
        return _dense_attention(
            self.queries[: self.CHUNK_LEN],
            keys,
            values,
            self.PREFIX_LEN + np.arange(self.CHUNK_LEN),
            None,
            causal=True,
        )

    def test_chunk_pass_output_and_lse_both_carry_the_sink(self):
        out, lse = self._chunk_pass()
        ref_out, ref_lse = self._chunk_reference()
        self.assertLess(
            float(np.abs(out - _fold_sink(ref_out, ref_lse, self.sinks)).max()),
            TOLERANCE,
        )
        # log(e^sink + sum e^s): the sink enters m/l, so it is inside the LSE.
        want_lse = np.logaddexp(self.sinks[None, :], ref_lse)
        self.assertLess(float(np.abs(lse - want_lse).max()), TOLERANCE)
        # A sink-free LSE would be the reference one; the two must differ or the
        # assertion above proves nothing.
        self.assertGreater(float(np.abs(want_lse - ref_lse).max()), TOLERANCE)

    def test_prefix_pass_without_a_sink_reports_the_bare_lse(self):
        out, lse = self._prefix_pass()
        ref_out, ref_lse = _dense_attention(
            self.queries[: self.CHUNK_LEN],
            self.prefix_k,
            self.prefix_v,
            np.full(self.CHUNK_LEN, self.PREFIX_LEN),
            None,
        )
        self.assertLess(float(np.abs(out - ref_out).max()), TOLERANCE)
        self.assertLess(float(np.abs(lse - ref_lse).max()), TOLERANCE)

    def test_two_pass_merge_reproduces_the_full_causal_sink_reference(self):
        merged_out, merged_lse = _merge_by_lse(
            [self._prefix_pass(), self._chunk_pass()]
        )
        ref_out, ref_lse = self._full_reference()
        self.assertLess(
            float(np.abs(merged_out - _fold_sink(ref_out, ref_lse, self.sinks)).max()),
            TOLERANCE,
        )
        self.assertLess(
            float(
                np.abs(merged_lse - np.logaddexp(self.sinks[None, :], ref_lse)).max()
            ),
            TOLERANCE,
        )


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRpaV3FlagsOffMatchesReference(unittest.TestCase):
    """With every DCP flag at its default the kernel still matches the in-repo
    reference -- the arm TP8 and CI run through shared code the DCP work
    touched (the LSE gate on the output normaliser, ``kv_q_gap``, the
    block-skip start)."""

    PAGES_PER_SEQ = 2
    TOTAL_PAGES = 8

    def _inputs(self, seqs, seed, num_kv_heads=NUM_KV_HEADS):
        """Kernel and reference inputs for ``seqs`` = [(kv_len, q_len), ...].

        ``kv_len`` counts the ``q_len`` new tokens the kernel appends, so the
        reference pages carry them at the same absolute slots while the fused
        cache handed to the kernel is the pre-write one.
        """
        capacity = self.PAGES_PER_SEQ * PAGE_SIZE
        for kv_len, q_len in seqs:
            assert q_len <= kv_len <= capacity, (kv_len, q_len, capacity)
        assert len(seqs) * self.PAGES_PER_SEQ <= self.TOTAL_PAGES

        rng = np.random.default_rng(seed)
        slots = self.TOTAL_PAGES * PAGE_SIZE
        total_q = sum(q for _, q in seqs)
        max_num_tokens = max(8, ((total_q + 7) // 8) * 8)
        cache_k = _round_to_bf16(
            rng.standard_normal((slots, num_kv_heads, HEAD_DIM), np.float32)
        )
        cache_v = _round_to_bf16(
            rng.standard_normal((slots, num_kv_heads, HEAD_DIM), np.float32)
        )
        new_k = _round_to_bf16(
            rng.standard_normal((max_num_tokens, num_kv_heads, HEAD_DIM), np.float32)
        )
        new_v = _round_to_bf16(
            rng.standard_normal((max_num_tokens, num_kv_heads, HEAD_DIM), np.float32)
        )
        queries = _round_to_bf16(
            rng.standard_normal((max_num_tokens, NUM_Q_HEADS, HEAD_DIM), np.float32)
        )

        ref_k = cache_k.copy()
        ref_v = cache_v.copy()
        q_offset = 0
        for seq, (kv_len, q_len) in enumerate(seqs):
            base = seq * self.PAGES_PER_SEQ * PAGE_SIZE + kv_len - q_len
            ref_k[base : base + q_len] = new_k[q_offset : q_offset + q_len]
            ref_v[base : base + q_len] = new_v[q_offset : q_offset + q_len]
            q_offset += q_len

        return {
            "queries": queries,
            "new_k": new_k,
            "new_v": new_v,
            "cache": _fused_cache(cache_k, cache_v, self.TOTAL_PAGES, num_kv_heads),
            "ref_k": ref_k.reshape(self.TOTAL_PAGES, PAGE_SIZE, num_kv_heads, HEAD_DIM),
            "ref_v": ref_v.reshape(self.TOTAL_PAGES, PAGE_SIZE, num_kv_heads, HEAD_DIM),
            "kv_lens": [kv_len for kv_len, _ in seqs],
            "q_lens": [q_len for _, q_len in seqs],
            "total_q": total_q,
        }

    def _compare(self, seqs, seed, num_kv_heads=NUM_KV_HEADS, **kernel_kwargs):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
            ref_ragged_paged_attention,
        )

        inputs = self._inputs(seqs, seed, num_kv_heads)
        num_seqs = len(seqs)
        page_table = np.arange(num_seqs * self.PAGES_PER_SEQ, dtype=np.int32).reshape(
            num_seqs, self.PAGES_PER_SEQ
        )
        cu_q_lens = np.concatenate([[0], np.cumsum(inputs["q_lens"])]).astype(np.int32)
        is_decode = all(q == 1 for q in inputs["q_lens"])
        distribution = [num_seqs] * 3 if is_decode else [0, num_seqs, num_seqs]

        out, _ = ragged_paged_attention(
            _to_bf16(inputs["queries"]),
            _to_bf16(inputs["new_k"]),
            _to_bf16(inputs["new_v"]),
            inputs["cache"],
            jnp.asarray(inputs["kv_lens"], jnp.int32),
            jnp.asarray(page_table.reshape(-1), jnp.int32),
            jnp.asarray(cu_q_lens, jnp.int32),
            jnp.asarray(
                [i * self.PAGES_PER_SEQ * PAGE_SIZE for i in range(num_seqs + 1)],
                jnp.int32,
            ),
            jnp.asarray(distribution, jnp.int32),
            None,  # custom_mask
            None,  # attention_sink
            sm_scale=SM_SCALE,
            **kernel_kwargs,
        )
        ref_out = ref_ragged_paged_attention(
            _to_bf16(inputs["queries"])[: inputs["total_q"]],
            _to_bf16(inputs["ref_k"]),
            _to_bf16(inputs["ref_v"]),
            jnp.asarray(inputs["kv_lens"], jnp.int32),
            jnp.asarray(page_table, jnp.int32),
            jnp.asarray(cu_q_lens, jnp.int32),
            jnp.asarray([num_seqs], jnp.int32),
            sm_scale=SM_SCALE,
            **kernel_kwargs,
        )
        got = np.asarray(out.astype(jnp.float32))[: inputs["total_q"]]
        want = np.asarray(ref_out.astype(jnp.float32))
        diff = float(np.abs(got - want).max())
        self.assertLess(diff, TOLERANCE, f"max abs diff {diff}")

    def test_decode(self):
        self._compare([(37, 1), (256, 1), (1, 1)], seed=1)

    def test_extend(self):
        self._compare([(200, 8), (64, 5)], seed=2)

    def test_multi_head_grouping(self):
        # One KV head for all 16 query heads, and the 8:1 grouping of the
        # benchmark shape.
        self._compare([(129, 4)], seed=3, num_kv_heads=1)
        self._compare([(129, 4)], seed=3, num_kv_heads=NUM_KV_HEADS)

    def test_sliding_window(self):
        self._compare([(256, 8), (130, 1)], seed=4, sliding_window=SLIDING_WINDOW)

    def test_soft_cap(self):
        self._compare([(200, 8)], seed=5, soft_cap=30.0)


if __name__ == "__main__":
    unittest.main()
