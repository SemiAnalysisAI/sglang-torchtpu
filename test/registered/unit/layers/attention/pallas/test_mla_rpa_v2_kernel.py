# SPDX-License-Identifier: Apache-2.0
"""Numerical-accuracy tests for the vendored MLA v2 ragged-paged-attention kernel.

Ported from sglang-jax ``python/sgl_jax/test/test_mla_attention.py``: the same
shape/flag matrix (page sizes, mixed-extend and decode length lists, sliding
window, logit soft-cap), re-pointed at ``mla_ragged_paged_attention`` itself
instead of sglang-jax's ``MLAAttentionBackend``, since only the kernel is
vendored here. The reference is ported from sglang-jax's v1
``ref_mla_ragged_paged_attention`` (same scale / soft-cap / mask / bf16-probs
ordering) but evaluated in NumPy over the *logical* per-sequence K/V, so it
shares no paging or cache-addressing code with the kernel under test.

What each case checks
---------------------
1. ``o_latent`` matches the reference MQA attention, and
2. the kernel's own KV-cache write (the cache is donated and the kernel owns the
   write — the pool rejects out-of-band writes) lands in exactly the expected
   slots, byte-for-byte, with no collateral damage to prefix tokens, to other
   sequences' pages, or to unallocated pages. The one documented exception is
   the sub-word tail past ``kv_len``; see ``assert_cache_write`` in
   ``sglang_tpu.testing.mla_rpa_v2``.

Metadata mirrors what ``TpuPallasRpaV3AttnBackend._build_metadata`` feeds the
kernel in production: real (unaligned) ``kv_lens``, a *rectangular*
``page_indices`` table of ``bs * pages_per_seq`` entries and
``cu_kv_lens[i] = i * pages_per_seq * page_size`` as the page-indices offset.
Page ids are assigned from a shuffled pool rather than sequentially, so an
addressing bug that a contiguous layout would hide still shows up.

``distribution`` follows the MLA backend's override, not RPA v3's: decode is
``[N, N, N]`` and extend is ``[0, 0, N]`` (the kernel's PREFILL slot is dead
code, so extend must route through MIXED). Block-size tuples use that same slot
order, ``(decode, prefill, mixed)``, and carry the kernel's hardcoded fallbacks
so a growing tuned table cannot quietly change what these tests measure.
``_DECODE_BATCH_SIZE`` is what ``TpuPallasMlaAttnBackend`` passes in production.

Geometry is DeepSeek V2/V3 MLA — 512 latent + 64 rope, both padded to 128 lanes
by the kernel — at 8 q heads per shard, which is DeepSeek-V3 at attention_tp=16
and Ling-2.6 at attention_tp=8, the shape the v7 tuned table targets.

Runtime is ~50 s for the 21 cases on one v7x host, dominated by pallas compiles.

Usage:
    python test/registered/unit/layers/attention/pallas/test_mla_rpa_v2_kernel.py
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.mla_ragged_paged_attention_v2 import (
    mla_ragged_paged_attention,
)
from sglang_tpu.testing.mla_rpa_v2 import (
    MASK_VALUE,
    NUM_Q_HEADS,
    SM_SCALE,
    assert_cache_write,
    build_case,
    ref_attention,
)

_NUM_KV_PAGES_PER_BLOCK = (3, 1, 1)
_NUM_QUERIES_PER_BLOCK = (1, 1, 16)
_DECODE_BATCH_SIZE = 4

# atol must absorb a few bf16 ulps at the output scale (1 ulp in [2, 4) is
# 0.015625); observed kernel-vs-reference deviations are sparse (<0.005% of
# elements) and hardware-dependent (up to 0.031 on v7x). Upstream tpu-inference
# tests this kernel at atol=0.1/rtol=0.2.
_RTOL, _ATOL = 2e-2, 4e-2


@unittest.skipUnless(
    any(d.platform == "tpu" for d in jax.devices()), "needs a TPU device"
)
class TestMlaRaggedPagedAttentionV2(CustomTestCase):
    """Kernel-level accuracy for ``mla_ragged_paged_attention``.

    ``EXTEND_LENS`` mixes pure prefill (``q_len == kv_len``), extend with a
    prefix and single-token extend, at ``kv_len`` values that both do and do not
    fill their last page. ``DECODE_LENS`` is single-token decodes straddling page
    boundaries (127/128/129, 1023/1024/1025); 9 sequences at
    ``decode_batch_size=4`` exercises the batched pass (8 seqs) and the residual
    DECODE-tail pass (1 seq) in the same call.

    The ``SWA_*`` lists straddle a 128-token window, so both the in-window and
    the dropped ranges are covered: ``sliding_window`` bounds the causal span to
    ``(p - sliding_window, p]`` for a query at absolute position ``p``.
    ``soft_cap`` applies ``s <- soft_cap * tanh(s / soft_cap)`` to the logits
    before masking; 30.0 is the Gemma-2 / DeepSeek-style magnitude.
    """

    EXTEND_LENS = [(1, 128), (125, 125), (123, 522), (1, 511), (256, 512)]

    DECODE_LENS = [
        (1, 119),
        (1, 127),
        (1, 128),
        (1, 129),
        (1, 133),
        (1, 1001),
        (1, 1023),
        (1, 1024),
        (1, 1025),
    ]

    SWA_EXTEND_LENS = [(1, 128), (64, 64), (128, 256), (100, 300), (1, 400)]
    SWA_DECODE_LENS = [(1, 256), (1, 400), (1, 512), (1, 1024)]

    def _run(
        self,
        mode,
        lens,
        *,
        page_size,
        num_heads=NUM_Q_HEADS,
        sliding_window=None,
        soft_cap=None,
        tuned_blocks=False,
        decode_batch_size=_DECODE_BATCH_SIZE,
        dtype=None,
        s_dtype=None,
        seed=0,
    ):
        assert mode in ("extend", "decode")
        dtype = jnp.bfloat16 if dtype is None else dtype
        case = build_case(
            lens, page_size=page_size, num_heads=num_heads, dtype=dtype, seed=seed
        )
        bs = len(lens)

        if mode == "decode":
            assert all(q == 1 for q, _ in lens), "decode expects q_len == 1"
            distribution = np.array([bs, bs, bs], dtype=np.int32)
        else:
            distribution = np.array([0, 0, bs], dtype=np.int32)

        block_kwargs = {}
        if not tuned_blocks:
            block_kwargs = dict(
                num_kv_pages_per_block=_NUM_KV_PAGES_PER_BLOCK,
                num_queries_per_block=_NUM_QUERIES_PER_BLOCK,
            )

        out, updated_cache = mla_ragged_paged_attention(
            jnp.asarray(case.ql_nope),
            jnp.asarray(case.q_pe),
            jnp.asarray(case.new_kv_c),
            jnp.asarray(case.new_k_pe),
            jnp.asarray(case.cache_before),
            jnp.asarray(case.kv_lens),
            jnp.asarray(case.page_indices),
            jnp.asarray(case.cu_q_lens),
            jnp.asarray(case.cu_kv_lens),
            jnp.asarray(distribution),
            sm_scale=SM_SCALE,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=MASK_VALUE,
            decode_batch_size=decode_batch_size,
            **({} if s_dtype is None else {"s_dtype": s_dtype}),
            **block_kwargs,
        )
        jax.block_until_ready((out, updated_cache))

        expected = ref_attention(
            case,
            sm_scale=SM_SCALE,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=MASK_VALUE,
        )

        got = np.asarray(out).astype(np.float32)
        exp = expected.astype(np.float32)
        self.assertEqual(got.shape, exp.shape)
        max_diff = float(np.max(np.abs(got - exp))) if got.size else 0.0
        print(
            f"mode={mode} page_size={page_size} heads={num_heads} "
            f"seqs={len(lens)} shape={got.shape} max_diff={max_diff:.6f}"
        )
        np.testing.assert_allclose(
            got,
            exp,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=(
                f"o_latent differs from the reference (mode={mode}, "
                f"page_size={page_size}, max_diff={max_diff})"
            ),
        )

        assert_cache_write(case, updated_cache)

    def test_extend_page_size_8(self):
        """page_size 8 at bf16 packing 2 -> 4 rows per page, the tightest layout."""
        self._run("extend", self.EXTEND_LENS, page_size=8)

    def test_extend_page_size_16(self):
        self._run("extend", self.EXTEND_LENS, page_size=16)

    def test_extend_page_size_32(self):
        self._run("extend", self.EXTEND_LENS, page_size=32)

    def test_extend_page_size_64(self):
        """64 is the TPU backend's production page size."""
        self._run("extend", self.EXTEND_LENS, page_size=64)

    def test_extend_single_long_seq(self):
        """One pure-prefill sequence spanning many pages and many query blocks."""
        self._run("extend", [(1024, 1024)], page_size=64)

    def test_extend_odd_num_q_heads(self):
        """3 heads is not a multiple of the bf16 q packing (2).

        The kernel pads the head axis and slices the padding back off.
        """
        self._run("extend", self.EXTEND_LENS, page_size=64, num_heads=3)

    def test_extend_six_q_heads(self):
        """6 bf16 heads (Kimi-K3 at TP16) must compile in Mosaic.

        3 packed head rows do not fit the sublane tile; without q/o scratch
        padding the MIXED kernel fails Mosaic lowering.
        """
        self._run("extend", self.EXTEND_LENS, page_size=128, num_heads=6)

    def test_decode_six_q_heads(self):
        """6 bf16 heads through both the batched and the tail decode passes."""
        self._run(
            "decode",
            [(1, 64), (1, 130), (1, 257), (1, 300), (1, 511)],
            page_size=128,
            num_heads=6,
            decode_batch_size=4,
        )

    def test_decode_page_size_8(self):
        self._run("decode", self.DECODE_LENS, page_size=8)

    def test_decode_page_size_16(self):
        self._run("decode", self.DECODE_LENS, page_size=16)

    def test_decode_page_size_32(self):
        self._run("decode", self.DECODE_LENS, page_size=32)

    def test_decode_page_size_64(self):
        self._run("decode", self.DECODE_LENS, page_size=64)

    def test_decode_batched_pass_only(self):
        """bs == decode_batch_size: the DECODE-tail pass gets an empty grid."""
        self._run(
            "decode",
            [(1, 64), (1, 128), (1, 192), (1, 256)],
            page_size=64,
            decode_batch_size=4,
        )

    def test_decode_tail_pass_only(self):
        """bs < decode_batch_size: the BATCHED_DECODE grid is empty instead."""
        self._run(
            "decode",
            [(1, 64), (1, 130), (1, 257)],
            page_size=64,
            decode_batch_size=4,
        )

    def test_decode_batch_size_one(self):
        """decode_batch_size=1 (the kernel default): tail pass for all."""
        self._run(
            "decode",
            [(1, 129), (1, 512)],
            page_size=64,
            decode_batch_size=1,
        )

    def test_extend_fp32_scores(self):
        """s_dtype=fp32 anchors correctness independent of the bf16-scores
        default the other tests exercise (upstream #2198 made bf16 the
        default; fp32 removes that rounding term from the comparison)."""
        self._run("extend", self.EXTEND_LENS, page_size=64, s_dtype=jnp.float32)

    def test_decode_fp32_scores(self):
        self._run("decode", self.DECODE_LENS, page_size=64, s_dtype=jnp.float32)

    def test_extend_sliding_window(self):
        self._run("extend", self.SWA_EXTEND_LENS, page_size=16, sliding_window=128)

    def test_decode_sliding_window(self):
        self._run("decode", self.SWA_DECODE_LENS, page_size=16, sliding_window=128)

    def test_extend_soft_cap(self):
        self._run("extend", self.SWA_EXTEND_LENS, page_size=16, soft_cap=30.0)

    def test_decode_soft_cap(self):
        self._run("decode", self.SWA_DECODE_LENS, page_size=16, soft_cap=30.0)

    def test_extend_sliding_window_and_soft_cap(self):
        self._run(
            "extend",
            self.SWA_EXTEND_LENS,
            page_size=16,
            sliding_window=128,
            soft_cap=30.0,
        )

    def test_decode_sliding_window_and_soft_cap(self):
        self._run(
            "decode",
            self.SWA_DECODE_LENS,
            page_size=16,
            sliding_window=128,
            soft_cap=30.0,
        )

    def test_extend_with_tuned_block_sizes(self):
        """Block sizes left unset -> the kernel consults the tuned table.

        ``get_tuned_block_sizes_mla`` falls back to the hardcoded defaults on a
        miss, so this guards the lookup wiring: a wrong slot order or a bad
        tuple unpack fails here, not in numerics.
        """
        self._run("extend", self.EXTEND_LENS, page_size=64, tuned_blocks=True)

    def test_decode_with_tuned_block_sizes(self):
        self._run("decode", self.DECODE_LENS, page_size=64, tuned_blocks=True)


if __name__ == "__main__":
    unittest.main()
