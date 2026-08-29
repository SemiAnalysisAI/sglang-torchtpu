# SPDX-License-Identifier: Apache-2.0
"""Correctness coverage for the native rank-4 RPA-seq prefill kernel."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except (ImportError, RuntimeError):
        return False


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestPallasRpaSeqPrefill(unittest.TestCase):
    PAGE_SIZE = 128
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 1
    HEAD_DIM = 64

    def _run_and_compare(self, seq_lens, kv_dtype):
        import jax
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_seq import (
            ragged_paged_attention_prefill,
        )
        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        q_lens = [q_len for q_len, _ in seq_lens]
        kv_lens_host = [kv_len for _, kv_len in seq_lens]
        total_q_tokens = sum(q_lens)
        padded_q_tokens = max(128, math.ceil(total_q_tokens / 128) * 128)
        max_num_seqs = 8
        pages_per_seq = max(
            math.ceil(kv_len / self.PAGE_SIZE) for kv_len in kv_lens_host
        )
        total_pages = (
            sum(math.ceil(kv_len / self.PAGE_SIZE) for kv_len in kv_lens_host) + 1
        )

        key = jax.random.PRNGKey(17)
        q_key, k_key, v_key, cache_k_key, cache_v_key = jax.random.split(key, 5)

        def random_tensor(key, shape, dtype):
            return jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)

        queries = random_tensor(
            q_key,
            (padded_q_tokens, self.NUM_Q_HEADS, self.HEAD_DIM),
            jnp.bfloat16,
        )
        keys = random_tensor(
            k_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            kv_dtype,
        )
        values = random_tensor(
            v_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            kv_dtype,
        )
        cache_k = random_tensor(
            cache_k_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            kv_dtype,
        )
        cache_v = random_tensor(
            cache_v_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            kv_dtype,
        )

        page_rows = []
        next_page = 0
        for kv_len in kv_lens_host:
            num_pages = math.ceil(kv_len / self.PAGE_SIZE)
            page_rows.append(
                list(range(next_page, next_page + num_pages))
                + [0] * (pages_per_seq - num_pages)
            )
            next_page += num_pages
        page_rows.extend([[0] * pages_per_seq] * (max_num_seqs - len(page_rows)))
        page_indices_host = np.asarray(page_rows, dtype=np.int32)

        ref_k = cache_k
        ref_v = cache_v
        q_offset = 0
        for seq_idx, (q_len, kv_len) in enumerate(seq_lens):
            first_new_pos = kv_len - q_len
            for token_offset in range(q_len):
                logical_pos = first_new_pos + token_offset
                logical_page, slot = divmod(logical_pos, self.PAGE_SIZE)
                physical_page = int(page_indices_host[seq_idx, logical_page])
                ref_k = ref_k.at[physical_page, slot].set(keys[q_offset + token_offset])
                ref_v = ref_v.at[physical_page, slot].set(
                    values[q_offset + token_offset]
                )
            q_offset += q_len

        kv_cache = (
            jnp.stack([cache_k, cache_v], axis=3)
            .reshape(
                total_pages,
                self.PAGE_SIZE,
                2 * self.NUM_KV_HEADS,
                self.HEAD_DIM,
            )
            .transpose(0, 2, 3, 1)
        )
        kv_lens = jnp.asarray(
            kv_lens_host + [0] * (max_num_seqs - len(seq_lens)), dtype=jnp.int32
        )
        cu_q_lens = jnp.asarray(
            [0]
            + list(np.cumsum(q_lens, dtype=np.int32))
            + [total_q_tokens] * (max_num_seqs - len(seq_lens)),
            dtype=jnp.int32,
        )
        page_indices = jnp.asarray(page_indices_host.reshape(-1))
        distribution = jnp.asarray([0, 0, len(seq_lens)], dtype=jnp.int32)
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)
        scale_kwargs = (
            {"k_scale": 1.0, "v_scale": 1.0} if kv_dtype == jnp.float8_e4m3fn else {}
        )

        expected = ref_ragged_paged_attention(
            queries[:total_q_tokens],
            ref_k,
            ref_v,
            jnp.asarray(kv_lens_host, dtype=jnp.int32),
            jnp.asarray(page_indices_host[: len(seq_lens)]),
            cu_q_lens[: len(seq_lens) + 1],
            jnp.asarray([len(seq_lens)], dtype=jnp.int32),
            sm_scale=sm_scale,
            **scale_kwargs,
        )
        actual, updated_cache = ragged_paged_attention_prefill(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            num_kv_pages_per_block=1,
            num_queries_per_block=32,
            **scale_kwargs,
        )

        np.testing.assert_allclose(
            np.asarray(actual[:total_q_tokens], dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=0.02,
            atol=0.02,
        )
        logical_cache = updated_cache.transpose(0, 3, 1, 2).reshape(
            total_pages,
            self.PAGE_SIZE,
            self.NUM_KV_HEADS,
            2,
            self.HEAD_DIM,
        )
        actual_k = logical_cache[:, :, :, 0]
        actual_v = logical_cache[:, :, :, 1]
        next_page = 0
        for _, kv_len in seq_lens:
            for logical_page in range(math.ceil(kv_len / self.PAGE_SIZE)):
                used = min(kv_len - logical_page * self.PAGE_SIZE, self.PAGE_SIZE)
                for actual_cache, expected_cache in (
                    (actual_k, ref_k),
                    (actual_v, ref_v),
                ):
                    np.testing.assert_array_equal(
                        np.asarray(actual_cache[next_page, :used]),
                        np.asarray(expected_cache[next_page, :used]),
                    )
                next_page += 1
        np.testing.assert_array_equal(
            np.asarray(actual_k[next_page:]), np.asarray(ref_k[next_page:])
        )
        np.testing.assert_array_equal(
            np.asarray(actual_v[next_page:]), np.asarray(ref_v[next_page:])
        )

    def test_prefill_and_cached_prefix_match_reference(self):
        import jax.numpy as jnp

        for kv_dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
            with self.subTest(kv_dtype=kv_dtype):
                self._run_and_compare([(128, 128), (64, 192)], kv_dtype)


class TestPallasRpaSeqPrefillWideHeads(TestPallasRpaSeqPrefill):
    """Cover a high-GQA, 128-wide-head shape with larger cache pages."""

    PAGE_SIZE = 512
    NUM_Q_HEADS = 12
    NUM_KV_HEADS = 1
    HEAD_DIM = 128


if __name__ == "__main__":
    unittest.main()
