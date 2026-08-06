# SPDX-License-Identifier: Apache-2.0
"""TPU correctness coverage for adaptive split-K batched RPA decode.

This is a direct-JAX TPU test, so it intentionally avoids ``CustomTestCase``.
Importing the SGLang runtime test helpers activates the CPU-only JAX export
policy before this test can acquire the TPU client.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# Platform discovery above enforces CPU-only JAX for SGLang, but does not import
# JAX. Select the physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaAdaptiveSplitKernel(unittest.TestCase):
    """Compare the real split/merge kernel against pure-JAX attention."""

    PAGE_SIZE = 256
    PAGES_PER_SEQ = 257
    NUM_Q_HEADS = 4
    NUM_KV_HEADS = 1
    HEAD_DIM = 128
    TOLERANCE = 0.05

    def test_skewed_decode_split_and_writeback_match_reference(self):
        import jax
        import jax.numpy as jnp

        for kv_dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
            with self.subTest(kv_dtype=str(kv_dtype)):
                self._run_decode_case(
                    jax,
                    jnp,
                    kv_dtype,
                    kv_lens_host=np.asarray([65537, 257, 257, 257], dtype=np.int32),
                    kernel_batch_size=4,
                    stacked=False,
                    split_k_run=True,
                )

    def test_single_request_stacked_with_idle_lane_matches_reference(self):
        import jax
        import jax.numpy as jnp

        # Stacked mode forces bkv=1024 here. Three KV blocks therefore leave
        # one of the four lanes idle, exercising the finalize source mask.
        self._run_decode_case(
            jax,
            jnp,
            jnp.bfloat16,
            kv_lens_host=np.asarray([2049], dtype=np.int32),
            # Exercise production block-size selection with Qwen3's local
            # attention shape. The stacked override must cap the normal b8
            # decode choice so its 1K KV tiles fit in v7 VMEM.
            kernel_batch_size=None,
            num_q_heads=16,
            num_kv_heads=8,
            stacked=True,
            split_k_run=False,
        )

    def _run_decode_case(
        self,
        jax,
        jnp,
        kv_dtype,
        *,
        kv_lens_host,
        kernel_batch_size: int | None,
        stacked,
        split_k_run,
        num_q_heads: int | None = None,
        num_kv_heads: int | None = None,
    ):
        from sglang.srt.layers.attention.pallas.batched_rpa.configs import (
            BlockSizes,
        )
        from sglang.srt.layers.attention.pallas.batched_rpa.utils import (
            get_dtype_packing,
        )
        from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
            ragged_paged_attention,
        )
        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ref_ragged_paged_attention,
        )

        # The first request has 64K cached tokens. At bkv=256 the adaptive
        # planner splits its 256-block cached prefix across all four lanes,
        # merges the online-softmax partials, then processes the new-token
        # writeback block on the root lane. The other requests stay short.
        num_seqs = len(kv_lens_host)
        total_q_tokens = num_seqs
        padded_q_tokens = 8
        total_pages = num_seqs * self.PAGES_PER_SEQ
        num_q_heads = num_q_heads or self.NUM_Q_HEADS
        num_kv_heads = num_kv_heads or self.NUM_KV_HEADS
        q_dtype = jnp.bfloat16

        def random_kv(random_key, shape):
            return jax.random.normal(random_key, shape, dtype=q_dtype).astype(kv_dtype)

        key = jax.random.PRNGKey(17)
        q_key, new_k_key, new_v_key, cache_k_key, cache_v_key = jax.random.split(key, 5)
        q = jax.random.normal(
            q_key,
            (padded_q_tokens, num_q_heads, self.HEAD_DIM),
            dtype=q_dtype,
        )
        new_k = random_kv(
            new_k_key,
            (padded_q_tokens, num_kv_heads, self.HEAD_DIM),
        )
        new_v = random_kv(
            new_v_key,
            (padded_q_tokens, num_kv_heads, self.HEAD_DIM),
        )
        cache_k = random_kv(
            cache_k_key,
            (total_pages, self.PAGE_SIZE, num_kv_heads, self.HEAD_DIM),
        )
        cache_v = random_kv(
            cache_v_key,
            (total_pages, self.PAGE_SIZE, num_kv_heads, self.HEAD_DIM),
        )

        page_indices_host = np.arange(total_pages, dtype=np.int32).reshape(
            num_seqs, self.PAGES_PER_SEQ
        )
        ref_k = cache_k
        ref_v = cache_v
        for seq, kv_len in enumerate(kv_lens_host):
            new_token_pos = int(kv_len) - 1
            rel_page, slot = divmod(new_token_pos, self.PAGE_SIZE)
            abs_page = int(page_indices_host[seq, rel_page])
            ref_k = ref_k.at[abs_page, slot].set(new_k[seq])
            ref_v = ref_v.at[abs_page, slot].set(new_v[seq])

        # Batched RPA stores interleaved K/V heads in uint32-width packed
        # elements. Pad the head axis when FP8 packs four values per element.
        packing = get_dtype_packing(kv_dtype)
        unpacked_kv_heads = num_kv_heads * 2
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
        kv_lens = jnp.asarray(kv_lens_host)
        cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
        page_indices = jnp.asarray(page_indices_host.reshape(-1))
        distribution = jnp.full((3,), num_seqs, dtype=jnp.int32)
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)

        expected = ref_ragged_paged_attention(
            q[:total_q_tokens],
            ref_k,
            ref_v,
            kv_lens,
            jnp.asarray(page_indices_host),
            cu_q_lens,
            jnp.asarray([num_seqs], dtype=jnp.int32),
            sm_scale=sm_scale,
        )
        decode_block_sizes = None
        if kernel_batch_size is not None:
            decode_block_sizes = BlockSizes(
                bq_sz=1,
                bq_c_sz=1,
                bkv_sz=self.PAGE_SIZE,
                batch_size=kernel_batch_size,
                n_buffer=2,
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
            decode_block_sizes=decode_block_sizes,
            stacked=stacked,
            decode_only=True,
            split_k_run=split_k_run,
        )

        max_diff = jnp.max(
            jnp.abs(
                actual[:total_q_tokens].astype(jnp.float32)
                - expected.astype(jnp.float32)
            )
        )
        tolerance = 0.08 if kv_dtype == jnp.float8_e4m3fn else self.TOLERANCE
        self.assertLess(float(max_diff), tolerance)

        unpacked_cache = updated_cache.reshape(
            total_pages, self.PAGE_SIZE, aligned_kv_heads, self.HEAD_DIM
        )
        for seq, kv_len in enumerate(kv_lens_host):
            new_token_pos = int(kv_len) - 1
            rel_page, slot = divmod(new_token_pos, self.PAGE_SIZE)
            abs_page = int(page_indices_host[seq, rel_page])
            np.testing.assert_array_equal(
                np.asarray(unpacked_cache[abs_page, slot, 0:unpacked_kv_heads:2]),
                np.asarray(new_k[seq]),
            )
            np.testing.assert_array_equal(
                np.asarray(unpacked_cache[abs_page, slot, 1:unpacked_kv_heads:2]),
                np.asarray(new_v[seq]),
            )


if __name__ == "__main__":
    unittest.main()
