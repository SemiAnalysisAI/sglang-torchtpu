# SPDX-License-Identifier: Apache-2.0
"""TPU numerics coverage for rank-4 RPA-seq prefill and stacked decode."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestRpaSeqStackedEquivalence(unittest.TestCase):
    """Compare stacked decode with the reference after rank-4 prefill."""

    TOLERANCE = 2e-2

    def test_fp8_kv_across_page_boundary(self):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_seq import (
            kernel as seq_kernel,
        )
        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )
        from sglang.srt.layers.attention.pallas.stacked_rpa import wrapper as stacked

        rng = np.random.default_rng(0)
        num_seqs = 2
        prefill_len = 300
        page_size = 256
        q_heads = 8
        kv_heads = 1
        head_dim = 128
        scale = 0.01
        pages_per_seq = math.ceil((prefill_len + 1) / page_size)
        total_pages = pages_per_seq * num_seqs
        cache_shape = seq_kernel.get_kv_cache_shape(
            total_pages,
            page_size,
            kv_heads,
            head_dim,
            jnp.float8_e4m3fn,
        )
        self.assertEqual(len(cache_shape), 4)
        cache = jnp.zeros(cache_shape, jnp.float8_e4m3fn)
        page_indices = rng.permutation(total_pages).astype(np.int32)
        sm_scale = 1.0 / math.sqrt(head_dim)

        def bf16(value):
            return jnp.asarray(value, jnp.bfloat16)

        def quantize(value):
            return (jnp.asarray(value, jnp.float32) / scale).astype(jnp.float8_e4m3fn)

        total_q = prefill_len * num_seqs
        prefill_q = (
            rng.standard_normal((total_q, q_heads, head_dim), dtype=np.float32) * 0.5
        )
        prefill_k = quantize(
            rng.standard_normal((total_q, kv_heads, head_dim), dtype=np.float32) * 0.5
        )
        prefill_v = quantize(
            rng.standard_normal((total_q, kv_heads, head_dim), dtype=np.float32) * 0.5
        )
        _, cache = seq_kernel.ragged_paged_attention_prefill(
            bf16(prefill_q),
            prefill_k,
            prefill_v,
            cache,
            jnp.full((num_seqs,), prefill_len, jnp.int32),
            jnp.asarray(page_indices),
            jnp.arange(num_seqs + 1, dtype=jnp.int32) * prefill_len,
            jnp.asarray([0, 0, num_seqs], jnp.int32),
            sm_scale=sm_scale,
            k_scale=scale,
            v_scale=scale,
        )

        decode_q_raw = (
            rng.standard_normal((num_seqs, q_heads, head_dim), dtype=np.float32) * 0.5
        )
        decode_k_raw = (
            rng.standard_normal((num_seqs, kv_heads, head_dim), dtype=np.float32) * 0.5
        )
        decode_v_raw = (
            rng.standard_normal((num_seqs, kv_heads, head_dim), dtype=np.float32) * 0.5
        )
        decode_q = bf16(decode_q_raw)
        decode_k = quantize(decode_k_raw)
        decode_v = quantize(decode_v_raw)
        kv_lens = jnp.full((num_seqs,), prefill_len + 1, jnp.int32)
        cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
        distribution = jnp.full((3,), num_seqs, jnp.int32)
        actual, _ = stacked.ragged_paged_attention(
            bf16(np.copy(decode_q_raw)),
            quantize(np.copy(decode_k_raw)),
            quantize(np.copy(decode_v_raw)),
            cache,
            kv_lens,
            jnp.asarray(page_indices),
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=scale,
            v_scale=scale,
        )

        k_pages = jnp.zeros(
            (total_pages, page_size, kv_heads, head_dim), jnp.float8_e4m3fn
        )
        v_pages = jnp.zeros_like(k_pages)
        for seq_idx in range(num_seqs):
            start = seq_idx * prefill_len
            end = start + prefill_len
            sequence_k = jnp.concatenate(
                [prefill_k[start:end], decode_k[seq_idx : seq_idx + 1]], axis=0
            )
            sequence_v = jnp.concatenate(
                [prefill_v[start:end], decode_v[seq_idx : seq_idx + 1]], axis=0
            )
            for logical_page in range(pages_per_seq):
                physical_page = int(
                    page_indices[seq_idx * pages_per_seq + logical_page]
                )
                page_start = logical_page * page_size
                page_end = min(page_start + page_size, prefill_len + 1)
                page_len = page_end - page_start
                k_pages = k_pages.at[physical_page, :page_len].set(
                    sequence_k[page_start:page_end]
                )
                v_pages = v_pages.at[physical_page, :page_len].set(
                    sequence_v[page_start:page_end]
                )

        expected = ref_ragged_paged_attention(
            decode_q,
            k_pages,
            v_pages,
            kv_lens,
            jnp.asarray(page_indices).reshape(num_seqs, pages_per_seq),
            cu_q_lens,
            jnp.asarray([num_seqs], jnp.int32),
            sm_scale=sm_scale,
            k_scale=scale,
            v_scale=scale,
        )
        expected = np.asarray(expected.astype(jnp.float32))
        actual = np.asarray(actual.astype(jnp.float32))
        max_abs = float(np.abs(expected - actual).max())
        relative_error = max_abs / max(float(np.abs(expected).max()), 1e-6)
        self.assertLess(relative_error, self.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
