# SPDX-License-Identifier: Apache-2.0
"""Direct TPU correctness coverage for the batched-RPA kernel."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# SGLang platform discovery selects CPU-only JAX for export, so select the
# physical backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaKernel(unittest.TestCase):
    """Compare the kernel against pure-JAX attention on the token-major layout.

    16 q heads over 8 kv heads with kv_len 2049 -- a partial last page and an
    idle lane -- plus a check that the new K/V landed in the right cache slot.
    """

    PAGE_SIZE = 256
    PAGES_PER_SEQ = 9
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    TOLERANCE = 0.05

    def test_idle_lane_multihead_decode_and_writeback_match_reference(self):
        self._check_decode(query_size=1, prefix_len=2048)

    def test_verify_cross_page_writeback_matches_reference(self):
        # Distinct physical pages expose writes that incorrectly continue into
        # the next physical page instead of following the request page table.
        for seq_layout in (False, True):
            for prefix_len in (2046, 2048):
                with self.subTest(seq_layout=seq_layout, prefix_len=prefix_len):
                    self._check_decode(
                        query_size=4, prefix_len=prefix_len, seq_layout=seq_layout
                    )

    def _check_decode(self, *, query_size, prefix_len, seq_layout=False):
        import jax
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.batched_rpa.utils import (
            get_dtype_packing,
        )
        from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
            ragged_paged_attention,
        )
        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        kv_dtype = jnp.bfloat16
        kv_len = prefix_len + query_size
        total_pages = self.PAGES_PER_SEQ
        padded_q_tokens = 8

        key = jax.random.PRNGKey(17)
        q_key, new_k_key, new_v_key, cache_k_key, cache_v_key = jax.random.split(key, 5)
        q = jax.random.normal(
            q_key,
            (padded_q_tokens, self.NUM_Q_HEADS, self.HEAD_DIM),
            dtype=jnp.bfloat16,
        )
        new_k = jax.random.normal(
            new_k_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        new_v = jax.random.normal(
            new_v_key,
            (padded_q_tokens, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        cache_k = jax.random.normal(
            cache_k_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )
        cache_v = jax.random.normal(
            cache_v_key,
            (total_pages, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=kv_dtype,
        )

        page_indices_host = (
            np.random.default_rng(42)
            .permutation(total_pages)
            .astype(np.int32)
            .reshape(1, -1)
        )
        ref_k, ref_v = cache_k, cache_v
        for token in range(query_size):
            rel_page, slot = divmod(prefix_len + token, self.PAGE_SIZE)
            abs_page = int(page_indices_host[0, rel_page])
            ref_k = ref_k.at[abs_page, slot].set(new_k[token])
            ref_v = ref_v.at[abs_page, slot].set(new_v[token])

        packing = get_dtype_packing(kv_dtype)
        unpacked_kv_heads = self.NUM_KV_HEADS * 2
        aligned_kv_heads = ((unpacked_kv_heads + packing - 1) // packing) * packing
        kv_cache_unpacked = jnp.stack([cache_k, cache_v], axis=3).reshape(
            total_pages,
            self.PAGE_SIZE,
            unpacked_kv_heads,
            self.HEAD_DIM,
        )
        kv_cache_unpacked = jnp.pad(
            kv_cache_unpacked,
            ((0, 0), (0, 0), (0, aligned_kv_heads - unpacked_kv_heads), (0, 0)),
        )
        kv_cache = kv_cache_unpacked.reshape(
            total_pages,
            self.PAGE_SIZE,
            aligned_kv_heads // packing,
            packing,
            self.HEAD_DIM,
        )

        from sglang.srt.layers.attention.pallas.batched_rpa.configs import KVLayout

        if seq_layout:
            kv_cache = kv_cache_unpacked.reshape(
                total_pages,
                self.PAGE_SIZE,
                unpacked_kv_heads,
                self.HEAD_DIM // packing,
                packing,
            ).transpose(0, 2, 3, 4, 1)

        kv_lens = jnp.asarray([kv_len], dtype=jnp.int32)
        cu_q_lens = jnp.asarray([0, query_size], dtype=jnp.int32)
        page_indices = jnp.asarray(page_indices_host.reshape(-1))
        distribution = jnp.ones((3,), dtype=jnp.int32)
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)

        expected = ref_ragged_paged_attention(
            q[:query_size],
            ref_k,
            ref_v,
            kv_lens,
            jnp.asarray(page_indices_host),
            cu_q_lens,
            jnp.asarray([1], dtype=jnp.int32),
            sm_scale=sm_scale,
        )
        actual, updated_cache = ragged_paged_attention(
            q,
            new_k,
            new_v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            decode_query_size=query_size,
            kv_layout=(
                KVLayout.SEQ_ALONG_LANE if seq_layout else KVLayout.HEAD_ALONG_SUBLANE
            ),
        )

        max_diff = jnp.max(
            jnp.abs(
                actual[:query_size].astype(jnp.float32) - expected.astype(jnp.float32)
            )
        )
        self.assertLess(float(max_diff), self.TOLERANCE)

        if seq_layout:
            updated_cache = updated_cache.transpose(0, 4, 1, 2, 3)
        unpacked_cache = updated_cache.reshape(
            total_pages, self.PAGE_SIZE, aligned_kv_heads, self.HEAD_DIM
        )
        checked_slots = np.ones((total_pages, self.PAGE_SIZE), dtype=bool)
        if seq_layout:
            # This layout writes whole owned pages. Unused slots after kv_len
            # are unspecified; live prefixes and all other pages must survive.
            last_page, end_slot = divmod(kv_len, self.PAGE_SIZE)
            if end_slot:
                checked_slots[page_indices_host[0, last_page], end_slot:] = False
        np.testing.assert_array_equal(
            np.asarray(unpacked_cache[:, :, 0:unpacked_kv_heads:2])[checked_slots],
            np.asarray(ref_k)[checked_slots],
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked_cache[:, :, 1:unpacked_kv_heads:2])[checked_slots],
            np.asarray(ref_v)[checked_slots],
        )


if __name__ == "__main__":
    unittest.main()
