"""Correctness tests for the vendored Pallas ragged-paged-attention v3 kernel.

The kernel and its pure-JAX ``ref_ragged_paged_attention`` reference both
live in ``sglang.srt.layers.attention.pallas.ragged_paged_attention_v3``.
These tests build small inputs in both formats (fused/packed for the
kernel, unpacked pages for the reference) and assert the outputs match
within bfloat16 tolerance for the two cases the kernel splits on:

* decode-only (q_len == 1)
* extend (q_len > 1)

The tests require JAX on a TPU device and are skipped otherwise. They
do **not** depend on torch_tpu or the SGLang server runtime.
"""

from __future__ import annotations

import unittest

import numpy as np


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


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestRaggedPagedAttentionV3(unittest.TestCase):
    """Compare the Pallas kernel against the in-file reference impl."""

    NUM_Q_HEADS = 4
    NUM_KV_HEADS = 1
    HEAD_DIM = 128
    PAGE_SIZE = 16
    TOTAL_NUM_PAGES = 8
    PAGES_PER_SEQ = 4

    def _build_inputs(self, kv_len: int, q_len: int, *, seed: int = 0):
        """Build matching kernel/reference inputs for a single sequence.

        Returns a dict carrying both the kernel-format and reference-format
        tensors, plus the shared metadata.
        """
        import jax
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.util import get_dtype_packing

        dtype = jnp.bfloat16
        assert kv_len >= q_len, "kv_len must include the new q_len tokens"
        prefix_len = kv_len - q_len

        # Pad max_num_tokens to next multiple of 8 so the kernel's tile shapes
        # are happy; the q region used is q_len.
        max_num_tokens = max(8, ((q_len + 7) // 8) * 8)

        key = jax.random.PRNGKey(seed)
        k_q, k_new_k, k_new_v, k_kpages, k_vpages = jax.random.split(key, 5)

        q = jax.random.normal(
            k_q, (max_num_tokens, self.NUM_Q_HEADS, self.HEAD_DIM), dtype=dtype
        )
        new_k = jax.random.normal(
            k_new_k, (max_num_tokens, self.NUM_KV_HEADS, self.HEAD_DIM), dtype=dtype
        )
        new_v = jax.random.normal(
            k_new_v, (max_num_tokens, self.NUM_KV_HEADS, self.HEAD_DIM), dtype=dtype
        )

        # Unpacked KV pages for the reference. The kernel will write new_k/new_v
        # into the cache itself, so we pre-populate the reference's pages with
        # the new tokens at the slots the kernel writes to.
        k_pages = jax.random.normal(
            k_kpages,
            (self.TOTAL_NUM_PAGES, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=dtype,
        )
        v_pages = jax.random.normal(
            k_vpages,
            (self.TOTAL_NUM_PAGES, self.PAGE_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=dtype,
        )
        ref_kpages = k_pages
        ref_vpages = v_pages
        for tok in range(q_len):
            abs_pos = prefix_len + tok
            page = abs_pos // self.PAGE_SIZE
            slot = abs_pos % self.PAGE_SIZE
            ref_kpages = ref_kpages.at[page, slot].set(new_k[tok])
            ref_vpages = ref_vpages.at[page, slot].set(new_v[tok])

        # Fused/packed KV cache for the kernel, built from the *pre-write*
        # pages so the kernel must perform the writes itself.
        kv_packing = get_dtype_packing(dtype)
        kv_fused = jnp.concatenate([k_pages, v_pages], axis=2).reshape(
            self.TOTAL_NUM_PAGES,
            self.PAGE_SIZE,
            (self.NUM_KV_HEADS * 2) // kv_packing,
            kv_packing,
            self.HEAD_DIM,
        )

        kv_lens = jnp.array([kv_len], dtype=jnp.int32)
        page_indices = jnp.arange(self.PAGES_PER_SEQ, dtype=jnp.int32)
        cu_q_lens = jnp.array([0, q_len], dtype=jnp.int32)
        cu_kv_lens = jnp.array([0, kv_len], dtype=jnp.int32)

        return {
            "q": q,
            "new_k": new_k,
            "new_v": new_v,
            "kv_fused": kv_fused,
            "ref_kpages": ref_kpages,
            "ref_vpages": ref_vpages,
            "kv_lens": kv_lens,
            "page_indices": page_indices,
            "cu_q_lens": cu_q_lens,
            "cu_kv_lens": cu_kv_lens,
            "max_num_tokens": max_num_tokens,
            "q_len": q_len,
        }

    def _run_and_compare(self, inputs, *, distribution):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
            ref_ragged_paged_attention,
        )

        sm_scale = 1.0 / float(np.sqrt(self.HEAD_DIM))

        ref_out = ref_ragged_paged_attention(
            inputs["q"][: inputs["q_len"]],
            inputs["ref_kpages"],
            inputs["ref_vpages"],
            inputs["kv_lens"],
            inputs["page_indices"].reshape(1, self.PAGES_PER_SEQ),
            inputs["cu_q_lens"],
            jnp.array([1], dtype=jnp.int32),
            sm_scale=sm_scale,
        )

        kernel_out, _ = ragged_paged_attention(
            inputs["q"],
            inputs["new_k"],
            inputs["new_v"],
            inputs["kv_fused"],
            inputs["kv_lens"],
            inputs["page_indices"],
            inputs["cu_q_lens"],
            inputs["cu_kv_lens"],
            jnp.asarray(distribution, dtype=jnp.int32),
            None,  # custom_mask
            None,  # attention_sink
            sm_scale=sm_scale,
        )
        kernel_out = kernel_out[: inputs["q_len"]]

        diff = jnp.abs(
            ref_out.astype(jnp.float32) - kernel_out.astype(jnp.float32)
        ).max()
        # bf16 has ~3 decimal digits of precision; outputs are O(1).
        self.assertLess(float(diff), 0.05)

    def test_decode_single_seq(self):
        """q_len = 1, distribution = (1, 1, 1) -> DECODE-only path."""
        inputs = self._build_inputs(kv_len=17, q_len=1)
        self._run_and_compare(inputs, distribution=(1, 1, 1))

    def test_extend_single_seq(self):
        """q_len > 1, distribution = (0, 0, 1) -> MIXED path."""
        inputs = self._build_inputs(kv_len=33, q_len=8)
        self._run_and_compare(inputs, distribution=(0, 0, 1))


if __name__ == "__main__":
    unittest.main()
