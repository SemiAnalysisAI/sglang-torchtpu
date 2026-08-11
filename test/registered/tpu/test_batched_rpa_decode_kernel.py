# SPDX-License-Identifier: Apache-2.0
"""Decode-path correctness for the batched-RPA pallas kernel.

Covers the attention output, the KV writeback (which the existing stacked test
only checks for a single sequence), and the long-context mask regression that
``ServingConfigs.int_ty`` guards against.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=300, suite="stage-a-test-tpu-unit")

# Standalone raw-JAX kernel test, not a TorchTPU runtime process. SGLang
# platform discovery selects CPU-only JAX for export, so pick the physical
# backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"


def _has_jax_tpu() -> bool:
    try:
        import jax

        return any("TPU" in device.device_kind for device in jax.devices())
    except Exception:
        return False


PAGE_SIZE = 256
NUM_Q_HEADS = 12
NUM_KV_HEADS = 1
HEAD_DIM = 128


def _build(num_seqs: int, ctx_len: int, *, seed: int = 0):
    """Build one decode-shaped call: ``num_seqs`` sequences of ``ctx_len``."""
    import jax
    import jax.numpy as jnp

    from sglang.srt.layers.attention.pallas.batched_rpa.utils import get_dtype_packing

    pages_per_seq = (ctx_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = num_seqs * pages_per_seq
    kv_dtype = jnp.bfloat16

    kq, kk, kv, kck, kcv = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(kq, (num_seqs, NUM_Q_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
    new_k = jax.random.normal(kk, (num_seqs, NUM_KV_HEADS, HEAD_DIM), dtype=kv_dtype)
    new_v = jax.random.normal(kv, (num_seqs, NUM_KV_HEADS, HEAD_DIM), dtype=kv_dtype)
    cache_k = jax.random.normal(
        kck, (total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=kv_dtype
    )
    cache_v = jax.random.normal(
        kcv, (total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=kv_dtype
    )

    packing = get_dtype_packing(kv_dtype)
    kv_x2 = NUM_KV_HEADS * 2
    aligned = ((kv_x2 + packing - 1) // packing) * packing
    fused = jnp.stack([cache_k, cache_v], axis=3).reshape(
        total_pages, PAGE_SIZE, kv_x2, HEAD_DIM
    )
    fused = jnp.pad(fused, ((0, 0), (0, 0), (0, aligned - kv_x2), (0, 0)))
    kv_cache = fused.reshape(
        total_pages, PAGE_SIZE, aligned // packing, packing, HEAD_DIM
    )

    # Shuffle the page table so a writeback to the slot index cannot pass for a
    # writeback to the mapped page.
    perm = np.random.default_rng(seed + 1).permutation(total_pages).astype(np.int32)
    page_table = perm.reshape(num_seqs, pages_per_seq)

    return dict(
        q=q,
        new_k=new_k,
        new_v=new_v,
        cache_k=np.asarray(cache_k.astype(jnp.float32)),
        cache_v=np.asarray(cache_v.astype(jnp.float32)),
        new_k_host=np.asarray(new_k.astype(jnp.float32)),
        new_v_host=np.asarray(new_v.astype(jnp.float32)),
        q_host=np.asarray(q.astype(jnp.float32)),
        kv_cache=kv_cache,
        kv_lens=jnp.full((num_seqs,), ctx_len, dtype=jnp.int32),
        cu_q_lens=jnp.asarray(np.arange(num_seqs + 1, dtype=np.int32)),
        page_indices=jnp.asarray(page_table.reshape(-1)),
        page_table=page_table,
        distribution=jnp.full((3,), num_seqs, dtype=jnp.int32),
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        aligned=aligned,
        pages_per_seq=pages_per_seq,
    )


def _reference(b, num_seqs: int, ctx_len: int) -> np.ndarray:
    """Naive float32 attention over the cache with this step's token appended."""
    g = NUM_Q_HEADS // NUM_KV_HEADS
    out = np.zeros((num_seqs, NUM_Q_HEADS, HEAD_DIM), np.float32)
    for s in range(num_seqs):
        pages = b["page_table"][s]
        k_seq = np.concatenate([b["cache_k"][p] for p in pages], axis=0)[:ctx_len]
        v_seq = np.concatenate([b["cache_v"][p] for p in pages], axis=0)[:ctx_len]
        k_seq[ctx_len - 1] = b["new_k_host"][s]
        v_seq[ctx_len - 1] = b["new_v_host"][s]
        for h in range(NUM_Q_HEADS):
            logits = (b["q_host"][s, h] @ k_seq[:, h // g].T) * b["sm_scale"]
            p = np.exp(logits - logits.max())
            out[s, h] = (p / p.sum()) @ v_seq[:, h // g]
    return out


def _run(b):
    from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention(
        b["q"],
        b["new_k"],
        b["new_v"],
        b["kv_cache"],
        b["kv_lens"],
        b["page_indices"],
        b["cu_q_lens"],
        b["distribution"],
        sm_scale=b["sm_scale"],
    )


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaDecodeKernel(unittest.TestCase):
    TOLERANCE = 0.03

    def _assert_matches_reference(self, num_seqs: int, ctx_len: int):
        import jax.numpy as jnp

        b = _build(num_seqs, ctx_len)
        expected = _reference(b, num_seqs, ctx_len)
        actual, _ = _run(b)
        got = np.asarray(actual.astype(jnp.float32))
        max_diff = float(np.abs(got - expected).max())
        self.assertLess(
            max_diff,
            self.TOLERANCE,
            f"num_seqs={num_seqs} ctx_len={ctx_len}: max abs diff {max_diff}",
        )

    def test_multi_sequence_decode_matches_reference(self):
        self._assert_matches_reference(num_seqs=8, ctx_len=2049)

    def test_ragged_short_context_matches_reference(self):
        self._assert_matches_reference(num_seqs=3, ctx_len=257)

    def test_context_beyond_int16_matches_reference(self):
        """Regression: the mask iotas carry absolute token positions.

        With ``int_ty=int16`` (the v6e/v7 default for packed Q dtypes) those
        wrap once the addressable context passes INT16_MAX, which flips the
        causal predicate and destroys the output. ``ServingConfigs.int_ty``
        promotes to int32 above that bound.
        """
        self._assert_matches_reference(num_seqs=2, ctx_len=33024)

    def test_int_ty_promotes_past_int16(self):
        """The type must cover the whole last KV block, not just the context.

        The mask iota spans a full ``bkv_sz`` block, so the largest position it
        materialises is ``align_to(max_model_len, bkv_sz) - 1``. A window that
        fits int16 can still have a last block that does not.
        """
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.batched_rpa import configs

        def cfgs(pages_per_seq: int, bkv_sz: int) -> configs.RpaConfigs:
            serve = configs.ServingConfigs(
                num_seqs=4,
                page_size=PAGE_SIZE,
                total_q_tokens=4,
                num_page_indices=4 * pages_per_seq,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.bfloat16,
                dtype_out=jnp.bfloat16,
            )
            model = configs.ModelConfigs(
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                mask_value=float(jnp.finfo(jnp.bfloat16).min),
            )
            return configs.RpaConfigs(
                block=configs.BlockSizes(
                    bq_sz=1, bq_c_sz=1, bkv_sz=bkv_sz, batch_size=1, n_buffer=2
                ),
                model=model,
                serve=serve,
                mode=configs.RpaCase.DECODE,
                vmem_limit_bytes=1 << 26,
            )

        # Window past INT16_MAX: int32 regardless of the block size.
        self.assertEqual(cfgs(128, 32768).serve.max_model_len, 32768)
        self.assertEqual(cfgs(128, 32768).int_ty, jnp.int32)
        # Window inside INT16_MAX and a block size that divides it: int16.
        self.assertEqual(cfgs(127, 32512).serve.max_model_len, 32512)
        self.assertEqual(cfgs(127, 32512).int_ty, jnp.int16)
        # Same window, but a block size that does not divide it: the last
        # block reaches 33023, so the narrow type would wrap.
        self.assertEqual(cfgs(127, 768).int_ty, jnp.int32)

    def test_writeback_lands_on_the_mapped_page_and_spares_the_rest(self):
        """The appended token must hit exactly one (page, slot, head) cell.

        Everything else in the touched page must survive: a partially filled
        page can be co-owned by another writer under prefix caching / chunked
        prefill, so a page-granular writeback would corrupt live tokens.
        """
        import jax.numpy as jnp

        num_seqs, ctx_len = 4, 1025
        b = _build(num_seqs, ctx_len, seed=7)
        before_k = b["cache_k"].copy()
        before_v = b["cache_v"].copy()
        _, new_cache = _run(b)

        total_pages = num_seqs * b["pages_per_seq"]
        unpacked = np.asarray(
            new_cache.reshape(total_pages, PAGE_SIZE, b["aligned"], HEAD_DIM).astype(
                jnp.float32
            )
        )
        after_k = unpacked[:, :, 0 : NUM_KV_HEADS * 2 : 2]
        after_v = unpacked[:, :, 1 : NUM_KV_HEADS * 2 : 2]

        touched = set()
        for s in range(num_seqs):
            page, slot = divmod(ctx_len - 1, PAGE_SIZE)
            abs_page = int(b["page_table"][s, page])
            touched.add((abs_page, slot))
            np.testing.assert_allclose(
                after_k[abs_page, slot], b["new_k_host"][s], rtol=0, atol=0
            )
            np.testing.assert_allclose(
                after_v[abs_page, slot], b["new_v_host"][s], rtol=0, atol=0
            )

        changed = np.argwhere(
            (np.abs(after_k - before_k) > 0).any(axis=(2, 3))
            | (np.abs(after_v - before_v) > 0).any(axis=(2, 3))
        )
        self.assertEqual(
            {(int(p), int(t)) for p, t in changed},
            touched,
            "writeback touched cells outside the appended token",
        )


@unittest.skipIf(not _has_jax_tpu(), "Requires JAX with a TPU device")
class TestBatchedRpaBlockSizes(unittest.TestCase):
    """The decode block-size picker and its tuned-table override."""

    def _cfgs(self, *, max_model_len: int, num_seqs: int = 64, kv_dtype=None):
        import jax.numpy as jnp

        from sglang.srt.layers.attention.pallas.batched_rpa import configs

        pages_per_seq = max_model_len // PAGE_SIZE
        model = configs.ModelConfigs(
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            mask_value=float(jnp.finfo(jnp.bfloat16).min),
        )
        serve = configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=PAGE_SIZE,
            total_q_tokens=num_seqs,
            num_page_indices=num_seqs * pages_per_seq,
            dtype_q=jnp.bfloat16,
            dtype_kv=kv_dtype or jnp.bfloat16,
            dtype_out=jnp.bfloat16,
        )
        return model, serve

    def test_decode_bkv_never_exceeds_max_model_len(self):
        from jax.experimental.pallas import tpu as pltpu

        from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
            calculate_block_sizes,
        )

        vmem = pltpu.get_tpu_info().vmem_capacity_bytes
        for max_model_len in (1024, 2048, 8192):
            model, serve = self._cfgs(max_model_len=max_model_len)
            decode, _ = calculate_block_sizes(model, serve, vmem)
            self.assertLessEqual(
                decode.bkv_sz,
                max_model_len,
                f"bkv_sz {decode.bkv_sz} > max_model_len {max_model_len}: the "
                "flash does not clip MXU work to kv_len, so the tail is paid for",
            )
            self.assertEqual(decode.bkv_sz % PAGE_SIZE, 0)
            self.assertGreaterEqual(decode.bkv_sz, PAGE_SIZE)

    def test_decode_block_sizes_fit_vmem(self):
        from jax.experimental.pallas import tpu as pltpu

        from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
            _calculate_vmem_usage,
            calculate_block_sizes,
        )

        vmem = pltpu.get_tpu_info().vmem_capacity_bytes
        for max_model_len in (1024, 32768, 262144):
            model, serve = self._cfgs(max_model_len=max_model_len)
            decode, prefill = calculate_block_sizes(model, serve, vmem)
            for blocks in (decode, prefill):
                used = _calculate_vmem_usage(
                    model,
                    serve,
                    batch_size=blocks.batch_size,
                    n_buffer=blocks.n_buffer,
                    bq_sz=blocks.bq_sz,
                    bkv_sz=blocks.bkv_sz,
                    num_lanes=pltpu.get_tpu_info().num_lanes,
                )
                self.assertLessEqual(used, vmem, f"{blocks} needs {used} > {vmem}")

    def test_tuned_lookup_returns_none_for_an_unknown_key(self):
        from sglang.srt.layers.attention.pallas.batched_rpa import (
            configs,
            tuned_block_sizes,
        )

        model, serve = self._cfgs(max_model_len=3 * PAGE_SIZE, num_seqs=7)
        self.assertIsNone(
            tuned_block_sizes.get_tuned_block_sizes(
                model, serve, configs.RpaCase.DECODE, 1 << 26
            )
        )

    def test_tuned_entry_overrides_the_heuristic(self):
        from sglang.srt.layers.attention.pallas.batched_rpa import (
            configs,
            tuned_block_sizes,
        )

        vmem = 1 << 26
        model, serve = self._cfgs(max_model_len=4096, num_seqs=8)
        key = tuned_block_sizes.make_lookup_key(
            model, serve, configs.RpaCase.DECODE, vmem
        )
        self.assertNotIn(key, tuned_block_sizes.TUNED_BLOCK_SIZES)
        tuned_block_sizes.TUNED_BLOCK_SIZES[key] = (1, 1, 2048, 3, 2)
        try:
            self.assertEqual(
                tuned_block_sizes.get_tuned_block_sizes(
                    model, serve, configs.RpaCase.DECODE, vmem
                ),
                configs.BlockSizes(
                    bq_sz=1, bq_c_sz=1, bkv_sz=2048, batch_size=3, n_buffer=2
                ),
            )
            # A different VMEM budget must not reuse the entry.
            self.assertIsNone(
                tuned_block_sizes.get_tuned_block_sizes(
                    model, serve, configs.RpaCase.DECODE, vmem // 2
                )
            )
        finally:
            del tuned_block_sizes.TUNED_BLOCK_SIZES[key]

    def test_every_tuned_entry_is_self_consistent(self):
        """Each shipped entry must fit its keyed VMEM budget and be page-aligned."""
        from jax.experimental.pallas import tpu as pltpu

        from sglang.srt.layers.attention.pallas.batched_rpa import (
            tuned_block_sizes,
        )
        from sglang.srt.layers.attention.pallas.batched_rpa.wrapper import (
            _calculate_vmem_usage,
        )

        num_lanes = pltpu.get_tpu_info().num_lanes
        for key, entry in tuned_block_sizes.TUNED_BLOCK_SIZES.items():
            bq_sz, bq_c_sz, bkv_sz, batch_size, n_buffer = entry
            page_size, max_model_len, vmem_budget = key[1], key[4], key[8]
            self.assertEqual(bkv_sz % page_size, 0, key)
            self.assertLessEqual(bkv_sz, max_model_len, key)
            self.assertGreaterEqual(bq_sz, bq_c_sz, key)
            self.assertGreaterEqual(n_buffer, 2, key)
            self.assertGreaterEqual(batch_size, 1, key)

            heads = key[3]  # "q{n}_kv{n}_d{n}"
            q_per_kv, kv_heads, head_dim = (
                int(part.lstrip("qkvd")) for part in heads.split("_")
            )
            model, serve = self._cfgs(max_model_len=max_model_len)
            model = dataclasses.replace(
                model,
                num_q_heads=q_per_kv * kv_heads,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
            )
            used = _calculate_vmem_usage(
                model,
                serve,
                batch_size=batch_size,
                n_buffer=n_buffer,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                num_lanes=num_lanes,
            )
            self.assertLessEqual(used, vmem_budget, f"{key} -> {entry} needs {used}")


if __name__ == "__main__":
    unittest.main()
