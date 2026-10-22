# SPDX-License-Identifier: Apache-2.0
"""Whole-decoder Pallas megakernel for Muse Spark decode and DFlash verify on TPU.

One grid-less ``pallas_call`` per device per step takes the embedded rows and runs all
decoder layers (paged attention, router, experts with the post-expert norm,
pooler/unpooler) and the final norm. Every tensor-parallel reduction is an in-kernel
one-hop remote DMA. The LM head, logits and sampling stay in sglang.

It reads the weights sglang already loaded, in their serving layouts:

* experts: the fused-RS expert-TP layout (``w13 [E, H_e, 2*I_local]`` fp4 or fp8,
  ``w2 [E, I_local, H_e]``); per-layer refs, addressed by a branch tree on the layer
  index so the layer loop can stay rolled;
* KV: the paged fused pools ``[pages, page_size, kv_heads_x2/2, 2, 128]`` bf16 of the
  hybrid SWA allocator, one ref per layer, read in place;
* attention / pooler / unpooler / router: stacked K-major copies made at load time
  (``ops.MuseMegakernel.from_model``), because the matvec streams ``[K, N]`` tiles faster
  than the checkpoint's ``[N, K]``; fp8 with per-output-channel scales, except the router.

Rows are ``num_requests * rows_per_request`` tokens padded to a multiple of the TP size:
one per request at decode, a DFlash verify block per request at target-verify. Each row
carries its own position. The step's own K/V rows join attention from VMEM, causal within
a verify block, and come back as ``newkv`` for ``ops`` to scatter into the pools.

The expert phase follows the fused-RS expert-TP contract: each rank evaluates its
intermediate slice of every selected expert for all rows, the partial outputs are
reduce-scattered by token, summed in f32, rounded to bf16, normalised by the post-expert
norm, weighted and summed. Collective rounds per layer: o_proj reduce-scatter and
all-gather, pooler all-gather, expert reduce-scatter, combined-output all-gather, unpooler
all-gather.
"""

import contextlib
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128
NEG = -1e30
# Barrier semaphore id; distinct from the fused-RS MoE (0) and two-shot all-reduce (7) kernels.
_COLLECTIVE_ID = 6
FP4_PACK = 8
FP4_GROUP = 16

# Profiling only: set True to name in-kernel regions, visible in an xprof trace
# with libtpu's --xla_enable_custom_call_region_trace=true.
_TRACE = False


def _scope(name):
    return jax.named_scope(name) if _TRACE else contextlib.nullcontext()


# small-slab rows (replicated per layer, f32, width = hidden)
SMALL = dict(
    pre_attn_w=0,
    attn_alpha=1,
    attn_beta=2,
    pre_ffn_w=3,
    post_ffn_w=4,
    ffn_alpha=5,
    ffn_beta=6,
    pre_expert_w=7,
    expert_bias=8,
    post_expert_w=9,
    # per-output-channel scales of the fp8 dense weights
    wqkvg_s=10,
    wo_s=11,
    wpool_s=12,
    wunpool_s=13,
    # NVFP4 per-expert global scales (scale_fp8): gate in lanes [0, Ep), up in [Ep, 2 Ep)
    expert_gs=14,
    # and down in lanes [0, Ep)
    expert_gs_down=15,
)
SMALL_ROWS = 16


class MKConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Static description of one compiled megakernel (model geometry + step shape)."""

    hidden: int
    head_dim: int
    q_heads: int  # per device
    kv_heads: int  # per device
    n_layers: int
    n_experts: int
    top_k: int
    expert_hidden: int
    inter_local: int
    tp: int
    rms_eps: float
    post_eps: float
    scale_query_by: float
    attn_scale: float
    router_mult: float
    route_scale: float
    router_eps: float
    window: int
    global_layers: tuple  # NoPE + full attention
    expert_kinds: tuple  # per layer: "fp4" | "fp8"
    page_size: int
    pages_per_seq: int
    n_pages_entries: int  # length of the flattened page tables
    max_position: int  # rope table rows
    rows: int  # padded rows R (multiple of tp)
    rows_per_req: int  # 1 at decode, the verify block at target-verify
    n_req: int  # requests in the bucket
    capture_layers: tuple  # aux hidden-state layers for the draft model
    pool_padded_head_dim: int = 128
    # streaming
    tile_k: int = 1024
    tile_n: int = 1152
    # dense weight stream: ring slots (must divide the per-layer tile count) and issue depth
    ring_slots: int = 11
    stream_depth: int = 10
    # expert slots = expert_depth + 1, sized for fp4 slabs; fp8 layers use adjacent slot pairs
    expert_depth: int = 3
    # fp4 experts used by at most this many live (row, slot) pairs (0, 1, 2, 4 or 8) run the
    # exact fp8 block-diagonal dot on just those rows, padded up to the next power of two;
    # busier experts dequantize to bf16 for all rows
    compact_rows: int = 8
    # fp4 experts carry e4m3 block scales plus per-expert globals (smalls rows expert_gs*)
    # instead of bf16 scales with the globals folded in
    scale_fp8: bool = False
    # K chunk of the block-diagonal dot; 128-256 beat 512 at 2-8 rows
    blockdiag_kc: int = 256
    # in a step of at most few_rows_max rows (n_req * rows_per_req), the o-proj all-reduce is one
    # gather of those rows' partials (one latency phase instead of two) and the ffn-output
    # gather ships only those rows
    single_row_coll: bool = True
    # one gather ships every live row to every peer: it trades bytes for a latency phase
    few_rows_max: int = 2
    fp4_chunk_words: int = 128
    chunk_pages: int = 2
    # KV chunks are fetched in groups of kv_group per layer branch-tree walk (each walk has a
    # fixed cost); the ring holds two groups: one being consumed, the next in flight
    kv_group: int = 4
    # independent online-softmax streams over consecutive chunks, merged at the end
    attn_streams: int = 2
    # > 0: every score is exponentiated against this constant instead of a running row max (no
    # rescale). Exact because q and k are RMS-normalized per head: |score| <= scale_query_by *
    # head_dim * attn_scale, which validate() requires this to cover with fp8-rounding margin.
    attn_fixed_max: float = 0.0
    # a stream with no chunk left in its loop iteration skips the chunk (lax.cond) instead of
    # scoring a re-read chunk fully masked
    attn_skip_idle_streams: bool = False
    # SEQ_ALONG_LANE only. > 0: SWA layers fetch chunks of this many pages (a divisor of
    # chunk_pages) into sub-slots of the same ring, and the attention body is specialized per
    # layer type. A window spans a few short chunks; only full-attention layers gain from long ones.
    swa_chunk_pages: int = 0
    # With swa_chunk_pages: a full-attention layer's last chunk masks by key column only instead
    # of by [rows, keys] (same values; a row mask over a long chunk is costly).
    attn_col_mask: bool = False
    # Pool rows hold every local KV head side by side in one pool_padded_head_dim-lane row
    # (kv_heads * head_dim == pool_padded_head_dim) instead of one zero-padded row per head:
    # half the KV bytes and attention work at head_dim 64. The serving pool must match.
    kv_packed: bool = False
    # SEQ_ALONG_LANE pools (TpuSeqAlongLaneKVPool, batched RPA, page >= 256): a page block is
    # [2 * kv_heads, head_dim // pk, pk, page_size] with slots [K_h0, V_h0, K_h1, ...], tokens on
    # lanes and pk consecutive head dims per 32-bit word. Attention runs the packed path on K/V
    # transposed to [kv_dim, tokens]; kv_write_call merges new token columns into their pages.
    kv_sal: bool = False
    kv_sal_fp8: bool = False
    # fp8 SEQ_ALONG_LANE chunks reach the MXU as fp8 (exact bf16 x fp8 products): 1 = K only,
    # 2 = K and V; 0 widens the whole page block to bf16 on the VPU first
    sal_fp8_dot: int = 0
    # each row's expert outputs leave for their reduce-scatter owner as soon as the row's last
    # expert is done, overlapping y_rs with the rest of the expert loop
    y_early: bool = False
    # The expert loop runs owner block by owner block (peers from me+1 on, own rows last), so
    # each block of reduce-scatter rows is final part-way through the loop and leaves for its
    # owner as one DMA while the remaining experts run; y_rs then only waits. Same arithmetic
    # (every (row, slot) is written by exactly one expert).
    y_block_send: bool = False
    # xfull holds the bf16 expert inputs as packed bf16 pairs instead of f32 (exact: the
    # values are bf16): half its VMEM, for the larger steps.
    xfull_packed: bool = False
    # The busy-expert output merge runs one top-k slot at a time and the post-expert combine
    # one owned row at a time, so their temporaries stay [R, He] / [K, He] instead of growing
    # with every row of the step (Mosaic places those values in VMEM). Same arithmetic.
    row_bounded_temps: bool = False
    # page tables arrive as int32 words holding two 16-bit page ids (even entry low): half the
    # SMEM. Past ~600 KB of tables every page lookup stops being an SMEM read (24 requests at
    # 1M context: the 96-row step took 670 ms, the KV write 12 ms).
    packed_pages: bool = False
    # attention multiplies the bf16 queries by attn_scale once instead of every score; exact
    # because ops only sets it for a power-of-two scale
    fold_attn_scale: bool = False
    vmem_limit_bytes: int = 60 * 2**20

    @property
    def q_dim(self):
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    @property
    def qkvg_dim(self):
        return 2 * self.q_dim + 2 * self.kv_dim

    @property
    def g(self):
        return self.q_heads // self.kv_heads

    @property
    def hidden_local(self):
        return self.hidden // self.tp

    @property
    def expert_hidden_local(self):
        return self.expert_hidden // self.tp

    @property
    def n_experts_padded(self):
        return -(-self.n_experts // LANE) * LANE

    @property
    def pool_kv_heads(self):
        return 1 if (self.kv_packed or self.kv_sal) else self.kv_heads

    @property
    def packed_attn(self):
        # attention sees one row per token, every local head side by side
        return self.kv_packed or self.kv_sal

    @property
    def kv_dtype(self):
        return jnp.float8_e4m3fn if self.kv_sal_fp8 else jnp.bfloat16

    @property
    def sal_pack(self):
        # head dims per 32-bit word of a SEQ_ALONG_LANE page block
        return 4 if self.kv_sal_fp8 else 2

    @property
    def sal_words(self):
        # 32-bit word rows of one SEQ_ALONG_LANE page block: (slot, word) pairs
        return 2 * self.kv_heads * self.head_dim // self.sal_pack

    @property
    def kv_block(self):
        # one page block (SEQ_ALONG_LANE) or one token's row (fused) of a layer's pool
        if self.kv_sal:
            return (
                2 * self.kv_heads,
                self.head_dim // self.sal_pack,
                self.sal_pack,
                self.page_size,
            )
        return (self.pool_kv_heads, 2, self.pool_padded_head_dim)

    @property
    def kv_ring(self):
        return 2 * self.kv_group

    @property
    def swa_kv_group(self):
        """SWA chunks per prefetch group: half the ring counted in SWA sub-slots."""
        return self.kv_group * (self.chunk_pages // (self.swa_chunk_pages or self.chunk_pages))

    @property
    def rows_per_dev(self):
        return self.rows // self.tp

    def use_rope(self, l):
        return l not in self.global_layers

    def validate(self):
        assert self.rows % self.tp == 0 and self.rows >= self.tp
        assert self.n_req * self.rows_per_req <= self.rows
        assert 1 <= self.stream_depth <= self.ring_slots - 1
        assert "fp8" not in self.expert_kinds or (self.expert_depth + 1) % 2 == 0, (
            "fp8 layers pair the expert slots"
        )
        assert self.compact_rows in (0, 1, 2, 4, 8)
        # bf16-packed collective buffers are half the per-device width and must stay lane tiles
        assert (
            self.hidden_local % (2 * LANE) == 0
            and self.expert_hidden_local % (2 * LANE) == 0
        )
        assert (
            not self.kv_packed
            or self.kv_heads * self.head_dim == self.pool_padded_head_dim
        )
        assert not self.kv_sal or (
            not self.kv_packed
            and self.page_size % LANE == 0
            and self.head_dim % (8 * self.sal_pack) == 0
        )
        assert self.attn_streams >= 1 and self.kv_group % self.attn_streams == 0
        assert not self.swa_chunk_pages or (
            self.kv_sal
            and self.chunk_pages % self.swa_chunk_pages == 0
            and self.swa_kv_group % self.attn_streams == 0
        )
        assert not self.attn_col_mask or self.swa_chunk_pages
        # 1.1: fp8 rounding of the cached K can stretch |k| a few percent past sqrt(head_dim)
        assert not self.attn_fixed_max or self.attn_fixed_max >= 1.1 * (
            self.scale_query_by * self.head_dim * self.attn_scale
        )
        return self


# ----------------------------------------------------------------------------- small helpers
def _rms(x, eps, w=None):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def _heads_rms(x, n_heads, d, eps, gain=1.0):
    parts = [_rms(x[:, h * d : (h + 1) * d], eps) * gain for h in range(n_heads)]
    return parts[0] if n_heads == 1 else jnp.concatenate(parts, axis=1)


def _rope_roll(x, n_heads, d, cos_x, sin_x):
    """Interleaved rotary with lane-expanded tables: the pair partner is one lane roll away."""
    w = n_heads * d
    xv = x.astype(jnp.float32)
    c = jnp.concatenate([cos_x] * n_heads, axis=1) if n_heads > 1 else cos_x
    s = jnp.concatenate([sin_x] * n_heads, axis=1) if n_heads > 1 else sin_x
    lane = jax.lax.broadcasted_iota(jnp.int32, xv.shape, 1)
    even = (lane % 2) == 0
    partner = jnp.where(
        even, pltpu.roll(xv, shift=w - 1, axis=1), pltpu.roll(xv, shift=1, axis=1)
    )
    return xv * c + jnp.where(even, -1.0, 1.0) * partner * s


def _topk_by_rank(scores, k, n_live, rank_ref, score_ref=None):
    """Top-k by rank: ``rank[r, e]`` = number of experts beating ``e`` (higher score, ties to
    the lower id), so pick ``j`` is the expert of rank ``j`` -- the same picks as ``k`` argmax
    rounds, from one ``[E, E]`` comparison per live row instead of a dependent chain. Rows past
    ``n_live`` keep stale ranks."""
    R, E = scores.shape
    lower = jax.lax.broadcasted_iota(jnp.int32, (E, E), 0) < jax.lax.broadcasted_iota(
        jnp.int32, (E, E), 1
    )
    if score_ref is None:
        s_t = scores.T  # [E, R]: e' on sublanes
        for r in range(R):

            @pl.when(r < n_live)
            def _(r=r):
                col, row = s_t[:, r : r + 1], scores[r : r + 1, :]
                beats = (col > row) | ((col == row) & lower)
                rank_ref[r : r + 1, :] = jnp.sum(
                    beats.astype(jnp.int32), axis=0, keepdims=True
                )
    else:
        # The same comparisons over 8-row blocks in a loop: one block's [E, E] temporaries live
        # at a time instead of one set per unrolled row (they grew VMEM ~0.3 MiB per row).
        score_ref[...] = scores

        def block(b, carry):
            r0 = pl.multiple_of(b * 8, 8)
            rows8 = score_ref[pl.ds(r0, 8), :]  # [8, E]
            cols8 = rows8.T  # [E, 8]
            for i in range(8):

                @pl.when(r0 + i < n_live)
                def _(i=i):
                    col, row = cols8[:, i : i + 1], rows8[i : i + 1, :]
                    beats = (col > row) | ((col == row) & lower)
                    rank_ref[pl.ds(r0 + i, 1), :] = jnp.sum(
                        beats.astype(jnp.int32), axis=0, keepdims=True
                    )

            return carry

        jax.lax.fori_loop(0, (n_live + 7) // 8, block, 0)

    rank = rank_ref[...]
    idx = jax.lax.broadcasted_iota(jnp.int32, (R, E), 1)
    picks = [rank == j for j in range(k)]
    return [jnp.sum(jnp.where(p, idx, 0), axis=1, keepdims=True) for p in picks], picks


def _bf16_bits(x):
    """f32 -> its round-to-nearest-even bf16 bit pattern in the low 16 bits of a u32."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint32)
    return (b + jnp.uint32(0x7FFF) + ((b >> 16) & jnp.uint32(1))) >> 16


def _pack_bf16x2(x):
    """f32 ``[..., 2n]`` -> u32 ``[..., n]``: bf16 of the first half low, of the second half high.
    Halves a collective payload without a 16-row bf16 tile on 8-row buffers."""
    n = x.shape[-1] // 2
    return _bf16_bits(x[..., :n]) | (_bf16_bits(x[..., n:]) << 16)


def _unpack_bf16x2(w):
    lo = jax.lax.bitcast_convert_type(w << 16, jnp.float32)
    hi = jax.lax.bitcast_convert_type(w & jnp.uint32(0xFFFF0000), jnp.float32)
    return jnp.concatenate([lo, hi], axis=-1)


def _branch(idx, fns, lo=0):
    """Run ``fns[idx - lo]`` for a traced ``idx`` through a balanced tree of two-way branches.

    ``lax.switch`` over ~60 layers costs time linear in the branch count; a tree costs log2(n)
    compares.
    """
    if len(fns) == 1:
        fns[0]()
        return
    mid = len(fns) // 2
    jax.lax.cond(
        idx < lo + mid,
        lambda: _branch(idx, fns[:mid], lo),
        lambda: _branch(idx, fns[mid:], lo + mid),
    )


def _dispatch(idx, refs, fn, only=None):
    """Run ``fn(ref)`` on ``refs[idx]``; a traced index goes through ``_branch``. ``only``: the
    sorted indices ``idx`` can take, so the tree has one leaf per index rather than per ref."""
    if isinstance(idx, int):
        fn(refs[idx])
        return
    if only is None:
        _branch(idx, [functools.partial(fn, r) for r in refs])
        return

    def tree(keys):
        if len(keys) == 1:
            fn(refs[keys[0]])
            return
        mid = len(keys) // 2
        jax.lax.cond(idx < keys[mid], lambda: tree(keys[:mid]), lambda: tree(keys[mid:]))

    tree(tuple(only))


class _AttnGeom(NamedTuple):
    """Chunk geometry of one attention body: ``cp`` pages per chunk, ``kr`` ring slots fetched in
    groups of ``kg``. ``is_global`` is a Python bool when the body is specialized per layer
    type, else the layer's traced flag."""

    cp: int
    kg: int
    kr: int
    is_global: Any


def _sel(flag, a, b):
    """``where(flag, a, b)``, resolved at trace time for a Python-bool flag."""
    if isinstance(flag, bool):
        return a if flag else b
    return jnp.where(flag, a, b)


def _in_window(is_global, cond):
    """``is_global or cond``; None when every key is in range."""
    if isinstance(is_global, bool):
        return None if is_global else cond
    return jnp.logical_or(is_global, cond)


def _dequant_fp4(words, scale):
    """``[cw, N]`` u32 + ``[cw*8/16, N]`` scales -> ``[cw*8, N]`` bf16."""
    cw, n = words.shape
    v = pltpu.bitcast(words, jnp.float4_e2m1fn).astype(jnp.bfloat16)
    gq = cw * FP4_PACK // FP4_GROUP
    s = scale.astype(jnp.bfloat16)
    s = jnp.broadcast_to(s[:, None, :], (gq, FP4_GROUP, n)).reshape(cw * FP4_PACK, n)
    return v * s


def _fp4_dot(x, wslot, sslot, K, N, chunk_words):
    kw = K // FP4_PACK
    gq = chunk_words * FP4_PACK // FP4_GROUP
    # e4m3 scales tile 32 rows per sublane group: read whole tiles, slice them as f32
    srows = 32 if sslot.dtype == jnp.float8_e4m3fn and gq < 32 else 0
    acc = sblk = None
    for c in range(0, kw, chunk_words):
        cw = min(chunk_words, kw - c)
        g0, gn = c * FP4_PACK // FP4_GROUP, cw * FP4_PACK // FP4_GROUP
        if srows:
            if g0 % srows == 0:
                sblk = sslot[pl.ds(g0, srows), pl.ds(0, N)].astype(jnp.float32)
            sc = sblk[g0 % srows : g0 % srows + gn]
        else:
            sc = sslot[pl.ds(g0, gn), pl.ds(0, N)]
        w = _dequant_fp4(wslot[pl.ds(c, cw), pl.ds(0, N)], sc)
        part = jnp.dot(
            x[:, c * FP4_PACK : (c + cw) * FP4_PACK].astype(jnp.bfloat16),
            w,
            preferred_element_type=jnp.float32,
        )
        acc = part if acc is None else acc + part
    return acc


def _fp4_blockdiag_dot(xc, wslot, sslot, K, N, kc):
    """``xc [A, K]`` bf16 @ an NVFP4 slab (``wslot`` u32 words ``[K/8, N]``, block-16 scales
    ``sslot [K/16, N]``) -> f32 ``[A, N]``, exact in f32.

    The e2m1 codes go to the MXU as fp8 (exact) at the fp8 weight-push rate. The input is
    expanded block-diagonally per ``kc``-row chunk, row ``a*G + g`` = row ``a`` masked to scale
    group ``g``, so each group's partial product lands in its own result row and its scale
    multiplies the ``[A*G, N]`` result instead of the ``[K, N]`` weights.
    """
    A = xc.shape[0]
    kc = min(K, kc)
    gpc = kc // FP4_GROUP
    rows = A * gpc
    diag = (
        jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 0) % gpc
        == jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 1) // FP4_GROUP
    )
    # e4m3 scales tile 32 rows per sublane group: read whole tiles, slice them as f32
    srows = 32 if sslot.dtype == jnp.float8_e4m3fn and gpc < 32 else gpc
    acc = sblk = None
    for c0 in range(0, K, kc):
        xs = xc[:, c0 : c0 + kc]
        lhs = jnp.where(
            diag, jnp.broadcast_to(xs[:, None, :], (A, gpc, kc)).reshape(rows, kc), 0
        ).astype(jnp.bfloat16)
        w = pltpu.bitcast(
            wslot[pl.ds(c0 // FP4_PACK, kc // FP4_PACK), pl.ds(0, N)], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        res = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
        g0 = c0 // FP4_GROUP
        if g0 % srows == 0:
            sblk = sslot[pl.ds(g0, srows), pl.ds(0, N)].astype(jnp.float32)
        s = sblk[g0 % srows : g0 % srows + gpc]
        part = res * (jnp.concatenate([s] * A, axis=0) if A > 1 else s)
        acc = part if acc is None else acc + part
    return jnp.concatenate(
        [
            jnp.sum(acc[a * gpc : (a + 1) * gpc], axis=0, keepdims=True)
            for a in range(A)
        ],
        axis=0,
    )


def _fp8_dot(x, wslot, scale_row, K, N, chunk_words):
    """fp8 weights held as ``[K/4, N]`` u32 words; channelwise ``scale_row [1, N]``."""
    kw = K // 4
    acc = None
    for c in range(0, kw, chunk_words):
        cw = min(chunk_words, kw - c)
        w = pltpu.bitcast(wslot[pl.ds(c, cw), pl.ds(0, N)], jnp.float8_e4m3fn)
        part = jnp.dot(
            x[:, c * 4 : (c + cw) * 4].astype(jnp.bfloat16),
            w,
            preferred_element_type=jnp.float32,
        )
        acc = part if acc is None else acc + part
    return acc * scale_row


@functools.lru_cache(maxsize=None)
def _choose_tile(K, N, tk_max, tn_max):
    tns = [d for d in range(LANE, min(N, tn_max) + 1, LANE) if N % d == 0] or [N]
    tks = [d for d in range(16, min(K, tk_max) + 1, 16) if K % d == 0] or [K]
    return max(tks), max(tns)


# ----------------------------------------------------------------------------- weight stream
_PACK = {jnp.dtype(jnp.bfloat16): 2, jnp.dtype(jnp.float8_e4m3fn): 4}


def dense_plan(c):
    """(name, K, N, dtype) of every dense per-layer weight, in consumption order."""
    shapes = (
        ("wqkvg", c.hidden, c.qkvg_dim),
        ("wo", c.q_dim, c.hidden),
        ("wpool", c.hidden, c.expert_hidden_local),
        ("wrouter", c.hidden, c.n_experts_padded),
        ("wunpool", c.expert_hidden, c.hidden_local),
    )
    # qkvg / o / pool / unpool stream as fp8 e4m3 with per-output-channel scales
    fp8 = ("wqkvg", "wo", "wpool", "wunpool")
    return tuple(
        (n, K, N, jnp.float8_e4m3fn if n in fp8 else jnp.bfloat16) for n, K, N in shapes
    )


# The ring sizes worth streaming through: below the minimum the issue window runs too few
# tiles ahead to hide DMA latency; the maximum is the largest ring ops.py asks for.
MIN_RING_SLOTS = 8
MAX_RING_SLOTS = 11


def ring_slot_shape(c):
    """u32 words per ring slot: one [tile_k, tile_n] bf16 tile."""
    return (c.tile_k // 2, c.tile_n)


def dense_tiles(c):
    """(name, ki, ni, tk, tn, dtype) of every ring tile of one layer, in consumption order.

    The ring size must divide the tile count, so a count with no divisor in
    ``[MIN_RING_SLOTS, MAX_RING_SLOTS]`` gets its largest tiles split along K until it has
    one (tp16's 26 tiles would otherwise leave a 2-slot ring that runs 1 tile ahead).
    """
    rk, rn = ring_slot_shape(c)
    plan = dense_plan(c)
    shapes = [_choose_tile(K, N, rk * _PACK[jnp.dtype(dt)], rn) for _, K, N, dt in plan]
    for _ in plan:
        T = sum((K // tk) * (N // tn) for (_, K, N, _), (tk, tn) in zip(plan, shapes))
        if any(T % d == 0 for d in range(MIN_RING_SLOTS, MAX_RING_SLOTS + 1)):
            break
        splittable = [
            i
            for i, ((_, K, _, _), (tk, _)) in enumerate(zip(plan, shapes))
            if tk % 32 == 0 and K % (tk // 2) == 0
        ]
        if not splittable:
            break
        i = max(
            splittable,
            key=lambda i: shapes[i][0] * shapes[i][1] * jnp.dtype(plan[i][3]).itemsize,
        )
        shapes[i] = (shapes[i][0] // 2, shapes[i][1])
    tiles = []
    for (name, K, N, dt), (tk, tn) in zip(plan, shapes):
        tiles += [
            (name, ki, ni, tk, tn, dt) for ni in range(N // tn) for ki in range(K // tk)
        ]
    return tuple(tiles)


def fit_ring_slots(c, cap):
    """Largest ring size <= cap that divides the dense tiles per layer (the slot schedule repeats per layer)."""
    T = len(dense_tiles(c))
    return max(d for d in range(2, cap + 1) if T % d == 0)


class _WeightStream:
    """Every dense weight of a layer as one static tile sequence through one VMEM ring.

    The sequence is identical for every layer, so tile t of any layer lives in slot
    ``t % slots`` and the issue window runs ``depth`` tiles ahead across matvec and layer
    boundaries: attention, collectives and the expert phase do not leave the DMA engine idle
    while there is dense weight left to fetch. ``depth <= slots - 1``: the slot a new tile lands
    in held the tile consumed just before it. Slots are u32 words, viewed as the tile's dtype.
    """

    def __init__(self, W, buf, sems, tiles, depth):
        self.W, self.buf, self.sems = W, buf, sems
        self.slots = buf.shape[0]
        self.tiles, self.ranges = tiles, {}
        for t, (name, _, ni, tk, tn, dt) in enumerate(tiles):
            t0 = self.ranges[name][0] if name in self.ranges else t
            self.ranges[name] = (t0, t + 1, tk, tn, ni + 1, dt)
        self.T = len(self.tiles)
        if self.T % self.slots:
            raise ValueError(
                f"{self.T} dense tiles per layer do not divide into {self.slots} ring slots"
            )
        self.depth = min(depth, self.slots - 1, self.T)

    def _copy(self, l, t):
        name, ki, ni, tk, tn, dt = self.tiles[t]
        pack = _PACK[jnp.dtype(dt)]
        s_ = t % self.slots
        src = (
            self.W[name]
            .bitcast(jnp.uint32)
            .at[l, pl.ds(ki * tk // pack, tk // pack), pl.ds(ni * tn, tn)]
        )
        return pltpu.make_async_copy(
            src, self.buf.at[s_, pl.ds(0, tk // pack), pl.ds(0, tn)], self.sems.at[s_]
        )

    def prologue(self, l):
        for t in range(self.depth):
            self._copy(l, t).start()

    def mv(self, x, l, name, next_l, sm=None):
        """``x [R, K]`` @ ``W_l[name]`` -> f32 ``[R, N]``; fp8 weights take their scale row from ``sm``."""
        t0, t1, tk, tn, nn, dt = self.ranges[name]
        xb = x.astype(jnp.bfloat16)
        cols = [None] * nn
        for t in range(t0, t1):
            with _scope("ring_wait"):
                self._copy(l, t).wait()
            _, ki, ni, _, _, _ = self.tiles[t]
            w = self.buf.at[t % self.slots].bitcast(dt)[pl.ds(0, tk), pl.ds(0, tn)]
            part = jnp.dot(
                xb[:, ki * tk : (ki + 1) * tk], w, preferred_element_type=jnp.float32
            )
            cols[ni] = part if cols[ni] is None else cols[ni] + part
            u = t + self.depth
            if u < self.T:
                self._copy(l, u).start()
            elif next_l is not None:
                self._copy(next_l, u - self.T).start()
        out = cols[0] if nn == 1 else jnp.concatenate(cols, axis=1)
        if jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn):
            r = SMALL[name + "_s"]
            out = out * sm[r : r + 1, : out.shape[1]]
        return out


# ----------------------------------------------------------------------------- collectives
class _Coll:
    """One-hop collectives over the mesh axis as remote DMAs into peers' VMEM pads.

    Every collective kind has its own pad and its own send/receive semaphores: a peer
    one round ahead then cannot satisfy this round's receive wait with its next round's
    bytes. A kind's pad is reused only after at least one other rendezvous round, so a
    single buffer per kind suffices. ``slot`` is the sender's position in the gathered
    result (its TP rank for column-parallel outputs, its mesh index otherwise).
    """

    def __init__(self, tp, me, axis, pads, ssems, rsems):
        self.tp, self.me, self.axis = tp, me, axis
        self.pads, self.ssems, self.rsems = pads, ssems, rsems
        self.kinds = {k: i for i, k in enumerate(pads)}

    def _peer(self, d):
        return jax.lax.rem(self.me + d, self.tp)

    def _copy(self, kind, src, dst, peer):
        i = self.kinds[kind]
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=self.ssems.at[i],
            recv_sem=self.rsems.at[i],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def all_gather(self, kind, src_ref, slot, live=None):
        """``src_ref`` from every device into ``pad[slot_of_sender]``; returns the pad ref.

        ``live(m)``: whether device m's part is needed at all; parts of dead devices are
        neither sent nor waited for, and their pad slots hold stale data.
        """
        pad = self.pads[kind]
        if live is None:
            return self.all_gather_start(kind, src_ref, slot)()
        pad[slot] = src_ref[...]
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]

        @pl.when(live(self.me))
        def _():
            for c in cps:
                c.start()
            for c in cps:
                c.wait_send()

        for d, c in zip(range(1, self.tp), cps):

            @pl.when(live(self._peer(d)))
            def _(c=c):
                c.wait_recv()

        return pad

    def gather_few(self, kind, src_ref, slot):
        """``src_ref [F, W]`` from every device into ``pad[slot_of_sender]`` of the pad ``[tp, F, W]``."""
        pad = self.pads[kind]
        pad[slot] = src_ref[...]
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        return pad

    def all_gather_start(self, kind, src_ref, slot):
        """Issue ``all_gather``; returns the function that waits for it and returns the pad."""
        pad = self.pads[kind]
        pad[slot] = src_ref[...]
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()

        def wait():
            for c in cps:
                c.wait()
            return pad

        return wait

    def block_copy(self, kind, send_ref, owner, rows_per):
        """The ``reduce_scatter_rows`` DMA of rows ``[owner*rows_per, ...)`` to ``owner``."""
        return self._copy(
            kind,
            send_ref.at[pl.ds(pl.multiple_of(owner * rows_per, rows_per), rows_per)],
            self.pads[kind].at[self.me],
            owner,
        )

    def reduce_scatter_rows(self, kind, send_ref, rows_per, n_live, started=False):
        """bf16-packed ``send_ref [tp*rows_per, ...]``: device m receives everyone's rows
        ``[m*rows_per, ...)``; returns the pad ``[tp, rows_per, ...]`` (own slot filled locally).
        Devices whose rows are all past ``n_live`` neither receive nor get sent to.
        ``started``: every live owner's ``block_copy`` was already started; only wait."""
        pad = self.pads[kind]

        def rows(ref, m):
            return ref.at[pl.ds(pl.multiple_of(m * rows_per, rows_per), rows_per)]

        my_live = self.me * rows_per < n_live

        @pl.when(my_live)
        def _():
            pad[self.me] = rows(send_ref, self.me)[...]

        cps = []
        for d in range(1, self.tp):
            peer = self._peer(d)
            cp = self.block_copy(kind, send_ref, peer, rows_per)
            cps.append((peer, cp))
            if started:
                continue

            @pl.when(peer * rows_per < n_live)
            def _(cp=cp):
                cp.start()

        for peer, cp in cps:

            @pl.when(peer * rows_per < n_live)
            def _(cp=cp):
                cp.wait_send()

        @pl.when(my_live)
        def _():
            for _, cp in cps:
                cp.wait_recv()

        return pad

    def row_copy(self, kind, send_ref, r, rows_per):
        """Row ``r`` of ``send_ref`` to its reduce-scatter owner ``r // rows_per``."""
        owner = r // rows_per
        return owner, self._copy(
            kind,
            send_ref.at[pl.ds(r, 1)],
            self.pads[kind].at[self.me, pl.ds(r - owner * rows_per, 1)],
            owner,
        )

    def reduce_scatter_rows_finish(self, kind, send_ref, rows_per, n_live):
        """``reduce_scatter_rows`` whose remote rows were already started one by one
        (``row_copy``, every row of every live owner exactly once): wait for them."""
        pad = self.pads[kind]
        my_live = self.me * rows_per < n_live

        @pl.when(my_live)
        def _():
            pad[self.me] = send_ref[
                pl.ds(pl.multiple_of(self.me * rows_per, rows_per), rows_per)
            ]

        for r in range(self.tp * rows_per):
            owner, cp = self.row_copy(kind, send_ref, r, rows_per)

            @pl.when((owner != self.me) & (owner * rows_per < n_live))
            def _(cp=cp):
                cp.wait_send()

        @pl.when(my_live)
        def _():
            _, cp = self.row_copy(kind, send_ref, 0, rows_per)
            for _ in range((self.tp - 1) * rows_per):
                cp.wait_recv()

        return pad

    def reduce_scatter_cols(self, kind, src_ref, width):
        """``src_ref [R, tp*width]`` f32: device m receives everyone's columns ``[m*width, ...)``."""
        pad = self.pads[kind]
        pad[self.me] = src_ref[:, pl.ds(pl.multiple_of(self.me * width, LANE), width)]
        cps = []
        for d in range(1, self.tp):
            peer = self._peer(d)
            cps.append(
                self._copy(
                    kind,
                    src_ref.at[:, pl.ds(pl.multiple_of(peer * width, LANE), width)],
                    pad.at[self.me],
                    peer,
                )
            )
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        return pad


def _page(tab_ref, ent, packed):
    """Page id ``ent`` of an SMEM page table, unpacked or two ids per int32 word."""
    if not packed:
        return tab_ref[ent]
    word = tab_ref[ent // 2]
    return jax.lax.shift_right_logical(word, (ent % 2) * 16) & 0xFFFF


def pack_page_table(tab):
    """int32 ``[2n]`` page ids (< 65536) -> int32 ``[n]`` words, even entry in the low half."""
    assert tab.shape[0] % 2 == 0
    return (tab[0::2] & 0xFFFF) | (tab[1::2] << 16)


def few_rows(c):
    """Rows a step can have, if few enough for the gather-only collectives (else 1, unused)."""
    n = c.n_req * c.rows_per_req
    return n if n <= c.few_rows_max else 1


# ----------------------------------------------------------------------------- scratch
def scratch_vmem_bytes(cfg: MKConfig) -> int:
    """VMEM bytes the scratch plan declares (Mosaic's own temporaries come on top)."""
    total = 0
    for _, ref in scratch_plan(cfg):
        if getattr(ref, "memory_space", None) == pltpu.VMEM:
            total += math.prod(ref.shape) * jnp.dtype(ref.dtype).itemsize
    return total


def scratch_plan(cfg: MKConfig):
    c = cfg
    R, K, H = c.rows, c.top_k, c.hidden
    He, Il = c.expert_hidden, c.inter_local
    nslot = c.expert_depth + 1
    # expert rings in u32 words sized for fp4 (8 K-rows per word); an fp8 slab (4 per word)
    # takes two adjacent slots
    w13_words = He // FP4_PACK
    w2_words = Il // FP4_PACK
    cpt = c.chunk_pages * c.page_size
    sdt = jnp.float8_e4m3fn if c.scale_fp8 else jnp.bfloat16
    items = [
        (
            "ring",
            pltpu.VMEM((c.ring_slots, *ring_slot_shape(c)), jnp.uint32),
        ),  # dense weight stream
        ("ring_sems", pltpu.SemaphoreType.DMA((c.ring_slots,))),
        ("w13", pltpu.VMEM((nslot, w13_words, 2 * Il), jnp.uint32)),
        ("w13s", pltpu.VMEM((nslot, He // FP4_GROUP, 2 * Il), sdt)),
        ("w2", pltpu.VMEM((nslot, w2_words, He), jnp.uint32)),
        ("w2s", pltpu.VMEM((nslot, max(Il // FP4_GROUP, 16), He), sdt)),
        ("w13_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w13s_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w2_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w2s_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("smalls", pltpu.VMEM((2, SMALL_ROWS, H), jnp.float32)),
        ("small_sems", pltpu.SemaphoreType.DMA((2,))),
        ("final_norm", pltpu.VMEM((8, H), jnp.float32)),
        ("rope_blk", pltpu.VMEM((R, 8, 2 * c.head_dim), jnp.float32)),
        ("misc_sems", pltpu.SemaphoreType.DMA((R + 4,))),
        ("xpack", pltpu.VMEM((R, H // 2), jnp.uint32)),
        ("o_send", pltpu.VMEM((R, c.hidden_local // 2), jnp.uint32)),
        ("ysend", pltpu.VMEM((R, K, He // 2), jnp.uint32)),
        ("attn_buf", pltpu.VMEM((R, c.q_dim), jnp.float32)),
        ("qn_s", pltpu.VMEM((R, c.q_dim), jnp.float32)),
        ("kb_s", pltpu.VMEM((R, c.kv_dim), jnp.float32)),
        ("vb_s", pltpu.VMEM((R, c.kv_dim), jnp.float32)),
        ("wk_s", pltpu.VMEM((R, K, LANE), jnp.float32)),
        ("u_send", pltpu.VMEM((c.rows_per_dev, He // 2), jnp.uint32)),
        ("pool_send", pltpu.VMEM((R, c.expert_hidden_local // 2), jnp.uint32)),
        ("f_send", pltpu.VMEM((R, c.hidden_local // 2), jnp.uint32)),
        (
            "kv_chunk",
            pltpu.VMEM((c.kv_ring, c.chunk_pages, *c.kv_block), c.kv_dtype)
            if c.kv_sal
            else pltpu.VMEM((c.kv_ring, cpt, *c.kv_block), jnp.bfloat16),
        ),
        ("kv_chunk_sems", pltpu.SemaphoreType.DMA((c.kv_ring, c.chunk_pages))),
        (
            "kv_row",
            pltpu.VMEM((c.sal_words, R), jnp.uint32)
            if c.kv_sal
            else pltpu.VMEM((R, *c.kv_block), jnp.bfloat16),
        ),
        ("kv_row_sem", pltpu.SemaphoreType.DMA((1,))),
        ("uniq", pltpu.SMEM((R * K + 8,), jnp.int32)),
        # per distinct expert u: live (row, slot) count; its pairs, encoded row * K + slot, are
        # perm[ex_off[u] : ex_off[u] + ex_cnt[u]] (ex_pos is the fill cursor)
        ("ex_cnt", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_off", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_pos", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("perm", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_of", pltpu.SMEM((c.n_experts_padded,), jnp.int32)),
        (
            "xfull",
            pltpu.VMEM((R, He // 2), jnp.uint32)
            if c.xfull_packed
            else pltpu.VMEM((R, He), jnp.float32),
        ),
        ("rank", pltpu.VMEM((R, c.n_experts_padded), jnp.int32)),
        # row_bounded_temps: router scores, read 8 rows at a time by the top-k
        (
            "topk_s",
            pltpu.VMEM(
                (R if c.row_bounded_temps else 8, c.n_experts_padded), jnp.float32
            ),
        ),
        ("ids_v", pltpu.VMEM((R, LANE), jnp.int32)),
        ("ids_s", pltpu.SMEM((R, LANE), jnp.int32)),
        ("ids_sem", pltpu.SemaphoreType.DMA((1,))),
        ("seen", pltpu.SMEM((c.n_experts_padded,), jnp.int32)),
        # y_early: each row's last expert (index into uniq), and the rows grouped by it:
        # rows done at u are dn_rows[dn_end[u] - dn_cnt[u] : dn_end[u]]
        ("row_done", pltpu.SMEM((R,), jnp.int32)),
        ("dn_cnt", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("dn_end", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("dn_rows", pltpu.SMEM((R,), jnp.int32)),
        # y_block_send: uniq index past the last expert first used by owner block b (b counted
        # from the next rank on), so block b's rows are final once expert blk_end[b] - 1 is done
        ("blk_end", pltpu.SMEM((16,), jnp.int32)),
        # collective pads: bf16 pairs packed into u32 (_pack_bf16x2)
        ("pad_o_rs", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("pad_o_ag", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("pad_pool", pltpu.VMEM((c.tp, R, c.expert_hidden_local // 2), jnp.uint32)),
        ("pad_y", pltpu.VMEM((c.tp, c.rows_per_dev, K, He // 2), jnp.uint32)),
        ("pad_u", pltpu.VMEM((c.tp, c.rows_per_dev, He // 2), jnp.uint32)),
        ("pad_f", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("row1_send", pltpu.VMEM((few_rows(c), H // 2), jnp.uint32)),
        ("pad_o1", pltpu.VMEM((c.tp, few_rows(c), H // 2), jnp.uint32)),
        ("pad_f1", pltpu.VMEM((c.tp, few_rows(c), c.hidden_local // 2), jnp.uint32)),
        ("coll_ssems", pltpu.SemaphoreType.DMA((8,))),
        ("coll_rsems", pltpu.SemaphoreType.DMA((8,))),
    ]
    if c.scale_fp8:
        items += [
            # the fp8 layers' bf16 channel scales, which the e4m3 slabs cannot hold
            ("w13c", pltpu.VMEM((nslot, 2, 2 * Il), jnp.bfloat16)),
            ("w2c", pltpu.VMEM((nslot, 2, He), jnp.bfloat16)),
            ("gs_s", pltpu.SMEM((2, 2 * c.n_experts_padded), jnp.float32)),
            ("gs_sem", pltpu.SemaphoreType.DMA((1,))),
        ]
    return items


# ----------------------------------------------------------------------------- the program
WEIGHT_NAMES = (
    "rope",
    "final_norm",
    "smalls",
    "wqkvg",
    "wo",
    "wpool",
    "wrouter",
    "wunpool",
)


def build_kernel(cfg: MKConfig, axis: str = "model"):
    """Per-device body: ``f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv)``.

    ``meta`` int32 SMEM: [n_live_rows, tp_rank]. ``emb`` [R, H] bf16 is the token
    embedding (gathered and reduced by sglang). ``experts`` is 4 refs per layer
    (w13, w13_scale, w2, w2_scale) in layer order. Outputs: hidden ``[R, H]`` bf16, aux
    ``[R, ncap*H]`` bf16 (or ``[R, 128]`` dummy), then the aliased KV refs.
    """
    c = cfg.validate()
    R, K, H, E = c.rows, c.top_k, c.hidden, c.n_experts
    hd, hq, hkv, g = c.head_dim, c.q_heads, c.kv_heads, c.g
    ph = c.pool_kv_heads
    He, Il, Hel, Hl = (
        c.expert_hidden,
        c.inter_local,
        c.expert_hidden_local,
        c.hidden_local,
    )
    L = c.n_layers
    q = c.rows_per_req
    ps = c.page_size
    cpt = c.chunk_pages * ps
    pd = c.pool_padded_head_dim
    scratch = scratch_plan(c)
    snames = [n for n, _ in scratch]
    nw = len(WEIGHT_NAMES)
    ncap = len(c.capture_layers)
    fp4_layers = [l for l in range(L) if c.expert_kinds[l] == "fp4"]
    # rolled range: the contiguous block of fp4 layers
    roll_lo = fp4_layers[0] if fp4_layers else None
    roll_hi = fp4_layers[-1] + 1 if fp4_layers else None
    if fp4_layers and fp4_layers != list(range(roll_lo, roll_hi)):
        raise NotImplementedError("fp4 expert layers must be one contiguous range")
    gl = tuple(sorted(c.global_layers))

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, emb_ref, *rest):
        W = dict(zip(WEIGHT_NAMES, rest[:nw]))
        rest = rest[nw:]
        EX = rest[: 4 * L]
        rest = rest[4 * L :]
        KV = rest[:L]  # read-only here; see kv_write_call
        hid_out, aux_out, newkv_out = rest[L], rest[L + 1], rest[L + 2]
        S = dict(zip(snames, rest[L + 3 :]))
        n_live = meta_ref[0]
        tp_rank = meta_ref[1]
        me = jax.lax.axis_index(axis) if c.tp > 1 else 0
        coll = _Coll(
            c.tp,
            me,
            axis,
            {
                "o_rs": S["pad_o_rs"],
                "o_ag": S["pad_o_ag"],
                "pool": S["pad_pool"],
                "y": S["pad_y"],
                "u": S["pad_u"],
                "f": S["pad_f"],
                "o1": S["pad_o1"],
                "f1": S["pad_f1"],
            },
            S["coll_ssems"],
            S["coll_rsems"],
        )
        ws = _WeightStream(W, S["ring"], S["ring_sems"], dense_tiles(c), c.stream_depth)
        msem = S["misc_sems"]

        def clear_seen(e, carry):
            S["seen"][e] = 0
            return carry

        jax.lax.fori_loop(0, c.n_experts_padded, clear_seen, 0)
        # Entry barrier: no peer's remote write lands in this device's VMEM before it has
        # entered the kernel. The call aliases nothing, so its collective_id costs no copy.
        if c.tp > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, c.tp):
                pl.semaphore_signal(
                    barrier,
                    device_id=(jnp.mod(me + d, c.tp),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, c.tp - 1)

        F = few_rows(c)
        few = c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max

        def single_row(x, multi, one):
            """``multi(x)`` -> [R, H], or in a step of at most few_rows_max rows ``one(x)``'s [F, H]
            rows (the rest zero)."""
            if not few:
                return multi(x)
            y = one(x)
            return jnp.concatenate([y, jnp.zeros((R - F, y.shape[1]), y.dtype)], axis=0)

        def allreduce_row(x):
            S["row1_send"][...] = _pack_bf16x2(x[0:F, :])
            pad = coll.gather_few("o1", S["row1_send"], me)
            tot = _unpack_bf16x2(pad[0])
            for d in range(1, c.tp):
                tot = tot + _unpack_bf16x2(pad[d])
            return _unpack_bf16x2(_pack_bf16x2(tot))

        def allreduce(x):
            return single_row(x, allreduce_rows, allreduce_row)

        def allreduce_rows(x):
            """f32 ``[R, H]`` partials -> bf16-exact totals: column reduce-scatter + all-gather.
            Partials travel as bf16 and are summed in f32, as the native bf16 all-reduce."""
            for d in range(c.tp):
                S["xpack"][:, d * Hl // 2 : (d + 1) * Hl // 2] = _pack_bf16x2(
                    x[:, d * Hl : (d + 1) * Hl]
                )
            pad = coll.reduce_scatter_cols("o_rs", S["xpack"], Hl // 2)
            mine = _unpack_bf16x2(pad[0])
            for d in range(1, c.tp):
                mine = mine + _unpack_bf16x2(pad[d])
            S["o_send"][...] = _pack_bf16x2(mine)
            pad2 = coll.all_gather("o_ag", S["o_send"], me)
            return jnp.concatenate(
                [_unpack_bf16x2(pad2[d]) for d in range(c.tp)], axis=1
            )

        # ---------------- embedding: weightless RMS norm of the bf16 rows, kept in bf16 precision
        stream = (
            _rms(emb_ref[...].astype(jnp.float32), c.rms_eps)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )

        # ---------------- rotary rows for every row's position
        cps = []
        for b in range(R):
            p = jnp.clip(pos_ref[b], 0, c.max_position - 1)
            cp = pltpu.make_async_copy(
                W["rope"].at[pl.ds(pl.multiple_of((p // 8) * 8, 8), 8)],
                S["rope_blk"].at[b],
                msem.at[b],
            )
            cp.start()
            cps.append(cp)
        rr = []
        for b in range(R):
            cps[b].wait()
            p = jnp.clip(pos_ref[b], 0, c.max_position - 1)
            sel = jax.lax.broadcasted_iota(jnp.int32, (8, 2 * hd), 0) == (
                p - (p // 8) * 8
            )
            rr.append(
                jnp.sum(jnp.where(sel, S["rope_blk"][b], 0.0), axis=0, keepdims=True)
            )
        rtab = jnp.concatenate(rr, axis=0)
        cos_x, sin_x = rtab[:, :hd], rtab[:, hd:]
        row_live = jax.lax.broadcasted_iota(jnp.int32, (R, 1), 0) < n_live

        def flags(l):
            if isinstance(l, int):
                return (l not in gl), (l in gl)
            is_g = False
            for x in gl:
                is_g = jnp.logical_or(is_g, l == x)
            return jnp.logical_not(is_g), is_g

        def smalls_copy(l, slot):
            return pltpu.make_async_copy(
                W["smalls"].at[l], S["smalls"].at[slot], S["small_sems"].at[slot]
            )

        # ------------------------------------------------------------ attention
        def attention(l, x_bf, use_rope, is_global, nxt, sm):
            per_geom(is_global, lambda geo: kv_prologue(geo, l, 0))
            with _scope("qkvg_mv"):
                qkvg = ws.mv(x_bf, l, "wqkvg", nxt, sm=sm)
            qd, kd = c.q_dim, c.kv_dim
            qv, kv_, vv, og = (
                qkvg[:, :qd],
                qkvg[:, qd : qd + kd],
                qkvg[:, qd + kd : qd + 2 * kd],
                qkvg[:, qd + 2 * kd :],
            )
            qn = _heads_rms(qv, hq, hd, c.rms_eps, gain=c.scale_query_by)
            kn = _heads_rms(kv_, hkv, hd, c.rms_eps)
            qn = jnp.where(use_rope, _rope_roll(qn, hq, hd, cos_x, sin_x), qn)
            kn = jnp.where(use_rope, _rope_roll(kn, hkv, hd, cos_x, sin_x), kn)
            # the pool's bf16 rounding of K/V, as in the native path (K after norm+rope)
            kb = kn.astype(c.kv_dtype).astype(jnp.bfloat16)
            vb = vv.astype(c.kv_dtype).astype(jnp.bfloat16)
            # ---- this step's K/V rows go out to newkv (scattered into the pools by kv_write_call);
            # attention reads them from VMEM, the cache only for positions before them.
            if (
                c.kv_sal
            ):  # [slot * words + word, row]: an identity matmul transposes exactly
                eye = (
                    jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 0)
                    == jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 1)
                ).astype(jnp.bfloat16)
                S["kv_row"][...] = jnp.concatenate(
                    [
                        pltpu.bitcast(
                            jax.lax.dot_general(
                                eye,
                                x[:, h * hd : (h + 1) * hd],
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            ).astype(c.kv_dtype),
                            jnp.uint32,
                        )
                        for h in range(hkv)
                        for x in (kb, vb)
                    ],
                    axis=0,
                )
            for r in range(0 if c.kv_sal else R):
                if (
                    c.kv_packed
                ):  # the projection row already holds the heads side by side
                    S["kv_row"][r] = jnp.concatenate(
                        [kb[r : r + 1], vb[r : r + 1]], axis=0
                    ).reshape(1, 2, pd)
                    continue
                zpad = jnp.zeros((R, pd - hd), jnp.bfloat16)
                parts = []
                for h in range(hkv):
                    parts.append(
                        jnp.concatenate(
                            [kb[r : r + 1, h * hd : (h + 1) * hd], zpad[r : r + 1]],
                            axis=1,
                        )
                    )
                    parts.append(
                        jnp.concatenate(
                            [vb[r : r + 1, h * hd : (h + 1) * hd], zpad[r : r + 1]],
                            axis=1,
                        )
                    )
                S["kv_row"][r] = jnp.concatenate(parts, axis=0).reshape(hkv, 2, pd)
            kv_out_cp = pltpu.make_async_copy(
                S["kv_row"], newkv_out.at[l], S["kv_row_sem"].at[0]
            )
            kv_out_cp.start()

            # ---- attention per live request (rows of one request share their page fetches), in a
            # runtime loop: the body is emitted once, not once per bucket slot
            S["qn_s"][...] = qn
            S["kb_s"][...] = kb.astype(jnp.float32)
            S["vb_s"][...] = vb.astype(jnp.float32)
            S["attn_buf"][...] = jnp.zeros((R, c.q_dim), jnp.float32)
            with _scope("attn_core"):
                n_req_live = jnp.minimum((n_live + q - 1) // q, c.n_req)

                def requests(geo):
                    def req_loop(j, carry):
                        request_attention_body(geo, l, j)
                        return carry

                    jax.lax.fori_loop(0, n_req_live, req_loop, 0)

                per_geom(is_global, requests)
                kv_out_cp.wait()
            attn = S["attn_buf"][...]
            attn = (
                _heads_rms(attn, hq, hd, c.rms_eps)
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            attn = (
                jax.nn.sigmoid(og.astype(jnp.bfloat16).astype(jnp.float32)) * attn
            ).astype(jnp.bfloat16)
            with _scope("o_mv"):
                o = ws.mv(attn, l, "wo", nxt, sm=sm)
            with _scope("o_allreduce"):
                return allreduce(o)

        def req_geom(geo, j):
            """First cache chunk and chunk count of request j (cached keys: [first_tok, p_lo))."""
            cpt_g = geo.cp * ps
            r0 = j * q
            p_lo = pos_ref[r0]
            first_tok = _sel(geo.is_global, 0, jnp.maximum(p_lo - c.window + 1, 0))
            first_ch = first_tok // cpt_g
            n_ch = jnp.where(
                jnp.logical_and(r0 < n_live, p_lo > 0),
                (p_lo - 1) // cpt_g - first_ch + 1,
                0,
            )
            return p_lo, first_ch, n_ch

        KG, KR, NS = c.kv_group, c.kv_ring, c.attn_streams
        if KG % NS:
            raise ValueError(
                f"{NS}-stream attention prefetches on group starts: kv_group must be a multiple of {NS}"
            )
        geo_global = _AttnGeom(c.chunk_pages, KG, KR, True)
        if c.swa_chunk_pages:
            geo_swa = _AttnGeom(
                c.swa_chunk_pages, c.swa_kv_group, 2 * c.swa_kv_group, False
            )

        def per_geom(is_global, fn):
            """``fn(geometry)`` for a layer: one shared body, or with swa_chunk_pages a body per
            layer type."""
            if not c.swa_chunk_pages:
                fn(_AttnGeom(c.chunk_pages, KG, KR, is_global))
            elif isinstance(is_global, bool):
                fn(geo_global if is_global else geo_swa)
            else:
                pl.when(is_global)(lambda: fn(geo_global))
                pl.when(jnp.logical_not(is_global))(lambda: fn(geo_swa))

        def ring_at(geo, slot, pi):
            """Ring entry and page of page ``pi`` of chunk slot ``slot``: a shorter chunk is a
            sub-slot of consecutive pages of an entry."""
            f = c.chunk_pages // geo.cp
            if f == 1:
                return slot, pi
            return slot // f, (slot % f) * geo.cp + pi

        def chunk_pages_of(geo, j, ci):
            pages = []
            for pi in range(geo.cp):
                pg_no = ci * geo.cp + pi
                ent = jnp.minimum(
                    j * c.pages_per_seq + jnp.minimum(pg_no, c.pages_per_seq - 1),
                    c.n_pages_entries - 1,
                )
                pages.append(
                    _sel(
                        geo.is_global,
                        _page(pf_ref, ent, c.packed_pages),
                        _page(psw_ref, ent, c.packed_pages),
                    )
                )
            return pages

        def chunk_dst(geo, slot, pi):
            # SEQ_ALONG_LANE: a whole page block; fused: the page's token rows
            if c.kv_sal:
                return S["kv_chunk"].at[ring_at(geo, slot, pi)]
            return S["kv_chunk"].at[slot, pl.ds(pi * ps, ps)]

        def chunk_sem(geo, slot, pi):
            return S["kv_chunk_sems"].at[ring_at(geo, slot, pi)]

        def issue_range(geo, l, j, ci0, slot0, n):
            """Chunks ``ci0 .. ci0+n`` of request j into ring slots ``slot0 ..``: ONE layer
            branch-tree walk whose leaf is a small runtime loop, so the 62
            leaves stay small (unrolled per-chunk code in every leaf bloats the kernel)."""

            def leaf(ref):
                def one(u, carry):
                    slot = (slot0 + u) % geo.kr
                    for pi, page in enumerate(chunk_pages_of(geo, j, ci0 + u)):
                        pltpu.make_async_copy(
                            ref.at[page], chunk_dst(geo, slot, pi), chunk_sem(geo, slot, pi)
                        ).start()
                    return carry

                jax.lax.fori_loop(0, n, one, 0)

            only = None
            if isinstance(geo.is_global, bool) and c.swa_chunk_pages:
                only = [x for x in range(c.n_layers) if (x in gl) == geo.is_global]
            _dispatch(l, KV, leaf, only)

        def wait_chunk(geo, slot):
            for pi in range(geo.cp):
                pltpu.make_async_copy(
                    KV[0].at[0], chunk_dst(geo, slot, pi), chunk_sem(geo, slot, pi)
                ).wait()

        def kv_prologue(geo, l, j):
            # the request's first two groups of chunks
            _, first_ch, n_ch = req_geom(geo, j)
            issue_range(geo, l, j, first_ch, 0, jnp.minimum(n_ch, geo.kr))

        def rows_at(ref, r0, n):
            """``n`` rows from a dynamic start: one-row accesses, as a multi-row dynamic sublane
            window must start at a multiple of 8."""
            return (
                jnp.concatenate([ref[pl.ds(r0 + t, 1), :] for t in range(n)], axis=0)
                if n > 1
                else ref[pl.ds(r0, 1), :]
            )

        def request_attention_body(geo, l, j):
            """Both local KV heads in one MXU op per step: the query rows of all heads are
            stacked ``[hkv*q*g, .]`` (rows ``(h, t, gi)``). Padded pools: scored against the
            chunk's interleaved key rows ``(token, h)``, scores of the other head masked. Packed
            pools (kv_packed): a token's one row holds every head side by side and a query row
            carries its q in its head's lanes, zeros elsewhere, so the op scores only same-head
            pairs: one key row per token and no head mask."""
            is_global, KG, KR, cpt = geo.is_global, geo.kg, geo.kr, geo.cp * ps
            r0 = pl.multiple_of(j * q, q)
            qg = q * g
            p_lo, first_ch, n_ch = req_geom(geo, j)
            qrows = rows_at(S["qn_s"], r0, q)  # [q, q_dim]

            def qrow(h, t, gi):
                x = qrows[t : t + 1, (h * g + gi) * hd : (h * g + gi + 1) * hd]
                if not c.packed_attn:
                    return x
                z = jnp.zeros((1, hd), x.dtype)
                return jnp.concatenate(
                    [x if hh == h else z for hh in range(hkv)], axis=1
                )

            q2 = jnp.concatenate(
                [
                    qrow(h, t, gi)
                    for h in range(hkv)
                    for t in range(q)
                    for gi in range(g)
                ],
                axis=0,
            ).astype(jnp.bfloat16)  # [hkv*qg, hd or pd]
            if c.fold_attn_scale:
                q2 = q2 * jnp.asarray(c.attn_scale, jnp.bfloat16)
            rowpos = jnp.concatenate(
                [jnp.broadcast_to(pos_ref[r0 + t], (g, 1)) for t in range(q)] * hkv,
                axis=0,
            )  # [hkv*qg, 1]
            rowhead = jax.lax.broadcasted_iota(jnp.int32, (hkv * qg, 1), 0) // qg

            @pl.when(j > 0)  # request 0's chunks were requested before the qkv matvec
            def _():
                kv_prologue(geo, l, j)

            init = (
                jnp.full((hkv * qg, 1), NEG, jnp.float32),
                jnp.zeros((hkv * qg, 1), jnp.float32),
                jnp.zeros((hkv * qg, c.kv_dim if c.packed_attn else pd), jnp.float32),
            )
            col = jax.lax.broadcasted_iota(jnp.int32, (1, cpt * ph), 1)
            colhead = col % ph
            coltok = col // ph
            kdim = c.kv_dim if c.packed_attn else hd

            def online(carry, s_, v, v_t=False):
                """``v_t``: v is [kv_dim, tokens] (SEQ_ALONG_LANE chunks), else [tokens, dim]."""
                m, lsum, acc = carry
                with _scope("softmax"):
                    if c.attn_fixed_max:  # m stays at its initial value; streams merge with weight 1
                        m_new, corr = m, None
                        pexp = jnp.exp(s_ - c.attn_fixed_max)
                        lsum = lsum + jnp.sum(pexp, axis=1, keepdims=True)
                    else:
                        m_new = jnp.maximum(m, jnp.max(s_, axis=1, keepdims=True))
                        corr = jnp.exp(m - m_new)
                        pexp = jnp.exp(s_ - m_new)
                        lsum = lsum * corr + jnp.sum(pexp, axis=1, keepdims=True)
                with _scope("pv"):
                    acc = acc[:, : v.shape[0 if v_t else 1]]
                    acc = (acc if corr is None else acc * corr) + jax.lax.dot_general(
                        pexp.astype(jnp.bfloat16),
                        v,
                        (((1,), (1 if v_t else 0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                return m_new, lsum, acc

            def chunk_scores(i, valid, clean=False):
                """Masked scores and V of chunk i; ``clean`` chunks lie wholly in the past of
                every row and skip the position and window masks."""
                slot = i % KR
                ci = first_ch + i
                with _scope("kv_unpack"):
                    if c.kv_sal:  # page block -> rows slot * hd + dim, tokens on lanes
                        kts, vts = [], []
                        for pi in range(geo.cp):
                            w = (
                                S["kv_chunk"]
                                .at[ring_at(geo, slot, pi)]
                                .bitcast(jnp.uint32)
                                .reshape(c.sal_words, ps)[...]
                            )
                            x8 = pltpu.bitcast(w, c.kv_dtype)
                            fp8_k = c.kv_sal_fp8 and c.sal_fp8_dot >= 1
                            fp8_v = c.kv_sal_fp8 and c.sal_fp8_dot >= 2
                            xb = None if (fp8_k and fp8_v) else x8.astype(jnp.bfloat16)
                            xk, xv = (x8 if fp8_k else xb), (x8 if fp8_v else xb)
                            kts.append(
                                jnp.concatenate(
                                    [
                                        xk[2 * h * hd : (2 * h + 1) * hd]
                                        for h in range(hkv)
                                    ],
                                    axis=0,
                                )
                            )
                            vts.append(
                                jnp.concatenate(
                                    [
                                        xv[(2 * h + 1) * hd : (2 * h + 2) * hd]
                                        for h in range(hkv)
                                    ],
                                    axis=0,
                                )
                            )
                        kk = jnp.concatenate(kts, axis=1)  # [kv_dim, cpt]
                        vk = jnp.concatenate(vts, axis=1)
                    else:
                        u = (
                            S["kv_chunk"]
                            .at[slot]
                            .bitcast(jnp.uint32)
                            .reshape(cpt * ph, pd)[...]
                        )  # rows (token, h)
                        kk = pltpu.bitcast(
                            (u & 0xFFFF).astype(jnp.uint16), jnp.bfloat16
                        )
                        vk = pltpu.bitcast((u >> 16).astype(jnp.uint16), jnp.bfloat16)
                live = None if c.packed_attn else colhead == rowhead
                if not clean:
                    keypos = coltok + ci * cpt
                    if c.attn_col_mask and is_global is True:
                        # a cached key (before p_lo) precedes every verify row: [1, cpt] mask
                        pos_live = (keypos < p_lo) & valid
                    else:
                        pos_live = (keypos < p_lo) & (keypos <= rowpos) & valid
                        in_win = _in_window(is_global, rowpos - keypos < c.window)
                        if in_win is not None:
                            pos_live = pos_live & in_win
                    live = pos_live if live is None else live & pos_live
                with _scope("qk"):
                    s_ = jax.lax.dot_general(
                        q2,
                        kk if c.kv_sal else kk[:, :kdim],
                        (((1,), (0 if c.kv_sal else 1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    if not c.fold_attn_scale:
                        s_ = s_ * c.attn_scale
                return (s_ if live is None else jnp.where(live, s_, NEG)), vk

            def body_two_stream(k, carry2, clean=False):
                i0 = 2 * k
                i1 = i0 + 1
                valid_b = i1 < n_ch

                # entering group m >= 1: group m - 1's slots are free; fetch group m + 1
                @pl.when(
                    jnp.logical_and(
                        jnp.logical_and(i0 % KG == 0, i0 >= KG), i0 + KG < n_ch
                    )
                )
                def _():
                    issue_range(
                        geo,
                        l,
                        j,
                        first_ch + i0 + KG,
                        (i0 + KG) % KR,
                        jnp.minimum(KG, n_ch - i0 - KG),
                    )

                with _scope("kv_wait"):
                    wait_chunk(geo, i0 % KR)

                    @pl.when(
                        valid_b
                    )  # an odd tail re-reads chunk i0, already waited for
                    def _():
                        wait_chunk(geo, i1 % KR)

                s_a, v_a = chunk_scores(i0, True, clean)
                if c.attn_skip_idle_streams and not clean:
                    # a past-the-end stream skips its chunk instead of scoring it fully masked
                    carry_b = jax.lax.cond(
                        valid_b,
                        lambda cr: online(cr, *chunk_scores(i1, True, False), v_t=c.kv_sal),
                        lambda cr: cr,
                        carry2[1],
                    )
                    return online(carry2[0], s_a, v_a, v_t=c.kv_sal), carry_b
                s_b, v_b = chunk_scores(
                    i1 if clean else jnp.minimum(i1, n_ch - 1), valid_b, clean
                )
                return (
                    online(carry2[0], s_a, v_a, v_t=c.kv_sal),
                    online(carry2[1], s_b, v_b, v_t=c.kv_sal),
                )

            def body_streams(k, carries, clean=False):
                """NS streams over chunks NS*k .. NS*k+NS-1: every wait before any compute."""
                i0 = NS * k

                @pl.when(
                    jnp.logical_and(
                        jnp.logical_and(i0 % KG == 0, i0 >= KG), i0 + KG < n_ch
                    )
                )
                def _():
                    issue_range(
                        geo,
                        l,
                        j,
                        first_ch + i0 + KG,
                        (i0 + KG) % KR,
                        jnp.minimum(KG, n_ch - i0 - KG),
                    )

                valid = [i0 + s < n_ch for s in range(NS)]
                with _scope("kv_wait"):
                    wait_chunk(geo, i0 % KR)
                    for s in range(1, NS):

                        @pl.when(valid[s])  # a past-the-end stream re-reads the last chunk
                        def _(s=s):
                            wait_chunk(geo, (i0 + s) % KR)

                out = []
                for s in range(NS):
                    if c.attn_skip_idle_streams and not clean and s > 0:
                        out.append(
                            jax.lax.cond(
                                valid[s],
                                lambda cr, i=i0 + s: online(
                                    cr, *chunk_scores(i, True, False), v_t=c.kv_sal
                                ),
                                lambda cr: cr,
                                carries[s],
                            )
                        )
                        continue
                    i = i0 + s if (clean or s == 0) else jnp.minimum(i0 + s, n_ch - 1)
                    s_, v = chunk_scores(i, True if s == 0 else valid[s], clean)
                    out.append(online(carries[s], s_, v, v_t=c.kv_sal))
                return tuple(out)

            # Global-layer chunks before a request's last one lie wholly in the past of every
            # verify row: chunks [0, n_clean) need only the cross-head mask.
            n_clean = _sel(is_global, jnp.maximum(n_ch - 1, 0), 0)
            if NS != 2:
                with _scope("chunk_loop"):
                    n_it, carries = (n_ch + NS - 1) // NS, (init,) * NS
                    if not isinstance(n_clean, int):  # SWA-only bodies have no clean chunks
                        carries = jax.lax.fori_loop(
                            0, n_clean // NS, functools.partial(body_streams, clean=True), carries
                        )
                    carries = jax.lax.fori_loop(n_clean // NS, n_it, body_streams, carries)
                    m_all = functools.reduce(jnp.maximum, [cr[0] for cr in carries])
                    ws_ = [jnp.exp(cr[0] - m_all) for cr in carries]
                    carry = (
                        m_all,
                        sum(cr[1] * w for cr, w in zip(carries, ws_)),
                        sum(cr[2] * w for cr, w in zip(carries, ws_)),
                    )
            else:
              with _scope("chunk_loop"):
                # Two independent online-softmax streams over even and odd chunks, merged at
                # the end: two dependency chains per iteration for the scheduler to interleave.
                # A stream that saw no live key keeps m = NEG and gets weight exp(NEG - m) = 0.
                n_pairs, carry2 = (n_ch + 1) // 2, (init, init)
                if not isinstance(n_clean, int):  # SWA-only bodies have no clean chunks
                    carry2 = jax.lax.fori_loop(
                        0,
                        n_clean // 2,
                        functools.partial(body_two_stream, clean=True),
                        carry2,
                    )
                (m_a, l_a, acc_a), (m_b, l_b, acc_b) = jax.lax.fori_loop(
                    n_clean // 2, n_pairs, body_two_stream, carry2
                )
                m_ab = jnp.maximum(m_a, m_b)
                w_a, w_b = jnp.exp(m_a - m_ab), jnp.exp(m_b - m_ab)
                carry = (m_ab, l_a * w_a + l_b * w_b, acc_a * w_a + acc_b * w_b)
            # the request's own new rows (causal within a verify block), padded to a sublane
            # tile (a 1-row operand does not lower), filler keys masked. Padded pools: key rows
            # (h, t); packed: the projection rows as they are, heads side by side.
            q8 = -(-q // 8) * 8
            krows = rows_at(S["kb_s"], r0, q).astype(jnp.bfloat16)
            vrows = rows_at(S["vb_s"], r0, q).astype(jnp.bfloat16)
            if c.packed_attn:
                zrows = (
                    jnp.zeros((q8 - q, c.kv_dim), jnp.bfloat16) if q8 > q else None
                )  # Mosaic has no 0-row vectors
                kn = jnp.concatenate([krows, zrows], axis=0) if q8 > q else krows
                vn = jnp.concatenate([vrows, zrows], axis=0) if q8 > q else vrows
                nt = jax.lax.broadcasted_iota(jnp.int32, (1, q8), 1)
                head_ok = True
            else:
                zrows = jnp.zeros((q8 - q, hd), jnp.bfloat16) if q8 > q else None
                kn = jnp.concatenate(
                    [
                        x
                        for h in range(hkv)
                        for x in (
                            [krows[:, h * hd : (h + 1) * hd]]
                            + ([zrows] if q8 > q else [])
                        )
                    ],
                    axis=0,
                )
                vn = jnp.concatenate(
                    [
                        x
                        for h in range(hkv)
                        for x in (
                            [vrows[:, h * hd : (h + 1) * hd]]
                            + ([zrows] if q8 > q else [])
                        )
                    ],
                    axis=0,
                )
                ncol = jax.lax.broadcasted_iota(jnp.int32, (1, hkv * q8), 1)
                nt = ncol % q8
                head_ok = ncol // q8 == rowhead
            newpos = jnp.zeros(nt.shape, jnp.int32) + 2**30
            for t in range(q):
                newpos = jnp.where(nt == t, pos_ref[r0 + t], newpos)
            live_new = head_ok & (newpos <= rowpos)
            in_win = _in_window(is_global, rowpos - newpos < c.window)
            if in_win is not None:
                live_new = live_new & in_win
            s_ = jax.lax.dot_general(
                q2, kn, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            if not c.fold_attn_scale:
                s_ = s_ * c.attn_scale
            _, lsum, acc = online(carry, jnp.where(live_new, s_, NEG), vn)
            res = acc / jnp.maximum(lsum, 1e-30)  # [hkv*qg, hd or pd]
            lanes = (
                (lambda h: slice(h * hd, (h + 1) * hd))
                if c.packed_attn
                else (lambda h: slice(0, hd))
            )
            for t in range(q):
                S["attn_buf"][pl.ds(r0 + t, 1), :] = jnp.concatenate(
                    [
                        res[h * qg + t * g + gi : h * qg + t * g + gi + 1, lanes(h)]
                        for h in range(hkv)
                        for gi in range(g)
                    ],
                    axis=1,
                )

        # ------------------------------------------------------------ MoE
        def unique_experts(ids_list, l):
            """Distinct experts of the live rows' slots, in first-use order, into ``uniq`` (SMEM),
            with each expert's (row, slot) pairs grouped in ``perm``."""
            S["ids_v"][:, 0:K] = jnp.concatenate(ids_list, axis=1)
            cp = pltpu.make_async_copy(S["ids_v"], S["ids_s"], S["ids_sem"].at[0])
            cp.start()
            cp.wait()
            stamp = l + 1  # "seen" is cleared once per call; one stamp per layer

            def count(slot, n):
                e = S["ids_s"][slot // K, slot % K]
                new = S["seen"][e] != stamp

                @pl.when(new)
                def _():
                    S["seen"][e] = stamp
                    S["uniq"][n] = e
                    S["ex_of"][e] = n
                    S["ex_cnt"][n] = 0

                u = S["ex_of"][e]
                S["ex_cnt"][u] = S["ex_cnt"][u] + 1
                return n + new.astype(jnp.int32)

            if c.y_block_send and c.tp > 1:
                # Scan the rows owner block by owner block, peers from me+1 on and own rows
                # last: uniq comes out grouped by the block that first uses each expert, and
                # blk_end[b] closes block b's group.
                Rc = c.rows_per_dev

                def count_in_block(row0):
                    def body(t, n):
                        r = row0 + t // K
                        e = S["ids_s"][r, t % K]
                        live = r < n_live
                        new = (S["seen"][e] != stamp) & live

                        @pl.when(new)
                        def _():
                            S["seen"][e] = stamp
                            S["uniq"][n] = e
                            S["ex_of"][e] = n
                            S["ex_cnt"][n] = 0

                        @pl.when(live)
                        def _():
                            u = S["ex_of"][e]
                            S["ex_cnt"][u] = S["ex_cnt"][u] + 1

                        return n + new.astype(jnp.int32)

                    return body

                n_u = 0
                for b in range(c.tp):
                    owner = jax.lax.rem(me + 1 + b, c.tp)
                    n_u = jax.lax.fori_loop(
                        0, Rc * K, count_in_block(owner * Rc), n_u
                    )
                    S["blk_end"][b] = n_u
            else:
                n_u = jax.lax.fori_loop(0, n_live * K, count, 0)

            def offsets(u, off):
                S["ex_off"][u] = off
                S["ex_pos"][u] = off
                return off + S["ex_cnt"][u]

            jax.lax.fori_loop(0, n_u, offsets, 0)

            def place(slot, carry):
                u = S["ex_of"][S["ids_s"][slot // K, slot % K]]
                p = S["ex_pos"][u]
                S["perm"][p] = slot
                S["ex_pos"][u] = p + 1
                return carry

            jax.lax.fori_loop(0, n_live * K, place, 0)
            if c.y_early:
                # rows past n_live (sent only for a partly live owner) go with the last expert
                for r in range(R):
                    S["row_done"][r] = jnp.where(r < n_live, 0, jnp.maximum(n_u - 1, 0))

                def last_use(slot, carry):
                    r = slot // K
                    u = S["ex_of"][S["ids_s"][r, slot % K]]
                    S["row_done"][r] = jnp.maximum(S["row_done"][r], u)
                    return carry

                jax.lax.fori_loop(0, n_live * K, last_use, 0)

                def zero(u, carry):
                    S["dn_cnt"][u] = 0
                    return carry

                jax.lax.fori_loop(0, jnp.maximum(n_u, 1), zero, 0)
                for r in range(R):
                    d = S["row_done"][r]
                    S["dn_cnt"][d] = S["dn_cnt"][d] + 1

                def ends(u, off):
                    S["dn_end"][u] = off
                    return off + S["dn_cnt"][u]

                jax.lax.fori_loop(0, jnp.maximum(n_u, 1), ends, 0)
                for r in range(R):
                    d = S["row_done"][r]
                    S["dn_rows"][S["dn_end"][d]] = r
                    S["dn_end"][d] = S["dn_end"][d] + 1
            return n_u

        def moe(l, ffn_in32, sm, kind, nxt):
            ffn_in = ffn_in32.astype(jnp.bfloat16)
            Rc = c.rows_per_dev
            # pooler (column-parallel by TP rank) first: its gather flies during routing
            with _scope("pool_mv"):
                pooled = (
                    ws.mv(ffn_in, l, "wpool", nxt, sm=sm)
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
            S["pool_send"][...] = _pack_bf16x2(pooled)
            pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
            # router: one bf16 pass (as torch_tpu's DEFAULT-precision fp32 F.linear)
            Ep = c.n_experts_padded
            with _scope("router_mv"):
                logits = ws.mv(ffn_in32, l, "wrouter", nxt) * c.router_mult
            with _scope("topk"):
                sc = jax.nn.sigmoid(logits)
                # padded lanes carry a -1e30 bias, so they are never selected
                bias = sm[SMALL["expert_bias"] : SMALL["expert_bias"] + 1, :Ep]
                ids_list, picks = _topk_by_rank(
                    sc + bias,
                    K,
                    n_live,
                    S["rank"],
                    S["topk_s"] if c.row_bounded_temps else None,
                )
                wsel = jnp.concatenate(
                    [
                        jnp.sum(jnp.where(p, sc, 0.0), axis=1, keepdims=True)
                        for p in picks
                    ],
                    axis=1,
                )
                wsel = (
                    wsel
                    / (jnp.sum(wsel, axis=1, keepdims=True) + c.router_eps)
                    * c.route_scale
                )  # [R, K]
            with _scope("unique_experts"):
                n_u = unique_experts(ids_list, l)
            if kind == "fp4":
                nslot, W13, W2 = c.expert_depth + 1, S["w13"], S["w2"]
            else:
                nslot = (c.expert_depth + 1) // 2
                W13 = S["w13"].reshape(nslot, He // 4, 2 * Il)
                W2 = S["w2"].reshape(nslot, Il // 4, He)
            depth = nslot - 1
            base = 4 * l if isinstance(l, int) else None
            # scale slabs: e4m3 block scales (fp4, scale_fp8) or bf16 (fp4 folded, fp8 channel)
            chan = c.scale_fp8 and kind != "fp4"
            S13 = S["w13c"] if chan else S["w13s"]
            S2 = S["w2c"] if chan else S["w2s"]
            gscale = c.scale_fp8 and kind == "fp4"
            if gscale:
                gs_copy = pltpu.make_async_copy(
                    W["smalls"].at[l, pl.ds(SMALL["expert_gs"], 2), pl.ds(0, 2 * Ep)],
                    S["gs_s"],
                    S["gs_sem"].at[0],
                )
                gs_copy.start()

            def ex_refs(li):
                return [EX[4 * li + j] for j in range(4)]

            def copies(i, slot):
                e = S["uniq"][i]
                out = []
                pack = FP4_PACK if kind == "fp4" else 4
                kw13, kw2 = He // pack, Il // pack
                for j, (dst, sem, rows_, cols, is_scale) in enumerate(
                    (
                        (W13, S["w13_sems"], kw13, 2 * Il, False),
                        (
                            S13,
                            S["w13s_sems"],
                            He // FP4_GROUP if kind == "fp4" else 2,
                            2 * Il,
                            True,
                        ),
                        (W2, S["w2_sems"], kw2, He, False),
                        (
                            S2,
                            S["w2s_sems"],
                            Il // FP4_GROUP if kind == "fp4" else 2,
                            He,
                            True,
                        ),
                    )
                ):
                    out.append((j, e, dst, sem, rows_, cols, is_scale))
                return out

            def expert_dmas(refs, i, slot):
                """The four slab DMAs of distinct expert ``i`` into ``slot``, from one layer's refs."""
                for j, e, dst, sem, rows_, cols, is_scale in copies(i, slot):
                    src = refs[j] if is_scale else refs[j].bitcast(jnp.uint32)
                    pltpu.make_async_copy(
                        src.at[e, pl.ds(0, rows_), pl.ds(0, cols)],
                        dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                        sem.at[slot],
                    ).start()

            def for_layer(fn):
                """``fn(refs)`` on this layer's expert refs: ONE layer branch-tree walk."""
                if isinstance(l, int):
                    fn([EX[4 * l + jj] for jj in range(4)])
                else:
                    _branch(
                        l,
                        [
                            functools.partial(fn, [EX[4 * li + jj] for jj in range(4)])
                            for li in range(roll_lo, roll_hi)
                        ],
                        lo=roll_lo,
                    )

            def start_expert(i, slot):
                for_layer(lambda refs: expert_dmas(refs, i, slot))

            def wait_expert(slot):
                pack = FP4_PACK if kind == "fp4" else 4
                kw13, kw2 = He // pack, Il // pack
                tmpl = [
                    EX[
                        4
                        * (
                            roll_lo
                            if roll_lo is not None and not isinstance(l, int)
                            else l
                        )
                        + jj
                    ]
                    for jj in range(4)
                ]
                for dst, sem, rows_, cols, j in (
                    (W13, S["w13_sems"], kw13, 2 * Il, 0),
                    (
                        S13,
                        S["w13s_sems"],
                        He // FP4_GROUP if kind == "fp4" else 2,
                        2 * Il,
                        1,
                    ),
                    (W2, S["w2_sems"], kw2, He, 2),
                    (
                        S2,
                        S["w2s_sems"],
                        Il // FP4_GROUP if kind == "fp4" else 2,
                        He,
                        3,
                    ),
                ):
                    src = tmpl[j] if j in (1, 3) else tmpl[j].bitcast(jnp.uint32)
                    pltpu.make_async_copy(
                        src.at[0, pl.ds(0, rows_), pl.ds(0, cols)],
                        dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                        sem.at[slot],
                    ).wait()

            # the first `depth` experts in one walk; the leaf is a small loop (code size)
            def prologue(refs):
                def one(t, carry):
                    expert_dmas(refs, t, t % nslot)
                    return carry

                jax.lax.fori_loop(0, jnp.minimum(depth, n_u), one, 0)

            for_layer(prologue)

            with _scope("pool_ag"):
                pad = pool_wait()
            with _scope("ex_setup"):
                x_full = jnp.concatenate(
                    [_unpack_bf16x2(pad[d]) for d in range(c.tp)], axis=1
                )  # [R, He]
                x_full = _rms(
                    x_full,
                    c.rms_eps,
                    sm[SMALL["pre_expert_w"] : SMALL["pre_expert_w"] + 1, :He],
                ).astype(jnp.bfloat16)
                ids3 = jnp.concatenate(
                    [i_[:, :, None] for i_ in ids_list], axis=1
                )  # [R, K, 1]
                # every live (row, slot) gets exactly one expert output, written bf16-packed;
                # rows past n_live stay zero
                S["ysend"][...] = jnp.zeros((R, K, He // 2), jnp.uint32)
                if c.xfull_packed:
                    S["xfull"][...] = _pack_bf16x2(x_full.astype(jnp.float32))
                else:
                    S["xfull"][...] = x_full.astype(jnp.float32)

            def ebody(i, carry):
                slot = i % nslot

                @pl.when(i + depth < n_u)
                def _():
                    start_expert(i + depth, (i + depth) % nslot)

                with _scope("ex_wait"):
                    wait_expert(slot)
                e = S["uniq"][i]

                def full_expert():
                    if kind == "fp4":
                        gu = _fp4_dot(
                            x_full,
                            W13.at[slot],
                            S["w13s"].at[slot],
                            He,
                            2 * Il,
                            c.fp4_chunk_words,
                        )
                    else:
                        gu = _fp8_dot(
                            x_full,
                            W13.at[slot],
                            S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(jnp.float32),
                            He,
                            2 * Il,
                            c.fp4_chunk_words,
                        )
                    gate, up = gu[:, :Il], gu[:, Il:]
                    if gscale:
                        gate, up = gate * S["gs_s"][0, e], up * S["gs_s"][0, Ep + e]
                    h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                    if kind == "fp4":
                        y = _fp4_dot(
                            h,
                            W2.at[slot],
                            S["w2s"].at[slot],
                            Il,
                            He,
                            min(Il // FP4_PACK, 16),
                        )
                    else:
                        y = _fp8_dot(
                            h,
                            W2.at[slot],
                            S2[slot, 0:1, :].astype(jnp.float32),
                            Il,
                            He,
                            min(Il // 4, 32),
                        )
                    if gscale:
                        y = y * S["gs_s"][1, e]
                    # slot space: row r, slot k gets y[r] where ids[r, k] == e
                    with _scope("ex_accum"):
                        if c.row_bounded_temps:
                            yp = _pack_bf16x2(y)
                            for k in range(K):
                                S["ysend"][:, k, :] = jnp.where(
                                    ids_list[k] == e, yp, S["ysend"][:, k, :]
                                )
                        else:
                            S["ysend"][...] = jnp.where(
                                ids3 == e, _pack_bf16x2(y)[:, None, :], S["ysend"][...]
                            )

                def compact_expert(nrows):
                    # one variant per row bucket: the expanded input and the scale work scale
                    # with the rows; bucket rows past the expert's count repeat its last pair
                    off, cnt = S["ex_off"][i], S["ex_cnt"][i]
                    pairs = [
                        S["perm"][off + jnp.minimum(j, cnt - 1)] for j in range(nrows)
                    ]
                    xrows = [S["xfull"][pl.ds(p // K, 1), :] for p in pairs]
                    if c.xfull_packed:
                        xrows = [_unpack_bf16x2(x) for x in xrows]
                    xc = jnp.concatenate(xrows, axis=0).astype(jnp.bfloat16)
                    gu = _fp4_blockdiag_dot(
                        xc, W13.at[slot], S["w13s"].at[slot], He, 2 * Il, c.blockdiag_kc
                    )
                    gate, up = gu[:, :Il], gu[:, Il:]
                    if gscale:
                        gate, up = gate * S["gs_s"][0, e], up * S["gs_s"][0, Ep + e]
                    h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                    yc = _fp4_blockdiag_dot(
                        h, W2.at[slot], S["w2s"].at[slot], Il, He, c.blockdiag_kc
                    )
                    yp = _pack_bf16x2(yc * S["gs_s"][1, e] if gscale else yc)
                    for j, p in enumerate(pairs):

                        @pl.when(j < cnt)
                        def _(j=j, p=p):
                            S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]

                def compact_scoped(nrows):
                    with _scope(f"ex_c{nrows}"):
                        compact_expert(nrows)

                def full_scoped():
                    with _scope("ex_full"):
                        full_expert()

                if kind == "fp4" and c.compact_rows:
                    cnt = S["ex_cnt"][i]
                    lo = 0
                    for b in (1, 2, 4, 8):
                        if b > c.compact_rows:
                            break
                        pl.when((cnt > lo) & (cnt <= b))(
                            functools.partial(compact_scoped, b)
                        )
                        lo = b
                    pl.when(cnt > c.compact_rows)(full_scoped)
                else:
                    full_scoped()
                if c.y_early:

                    def send_row(t, carry):
                        r = S["dn_rows"][S["dn_end"][i] - S["dn_cnt"][i] + t]
                        owner, cp = coll.row_copy("y", S["ysend"], r, Rc)

                        @pl.when((owner != me) & (owner * Rc < n_live))
                        def _():
                            cp.start()

                        return carry

                    jax.lax.fori_loop(0, S["dn_cnt"][i], send_row, 0)
                elif c.y_block_send and c.tp > 1:
                    # owner block b is final once its group's last expert is done (an empty
                    # group fires with the group before it)
                    for b in range(c.tp - 1):
                        owner = jax.lax.rem(me + 1 + b, c.tp)

                        @pl.when((i + 1 == S["blk_end"][b]) & (owner * Rc < n_live))
                        def _(owner=owner):
                            coll.block_copy("y", S["ysend"], owner, Rc).start()

                return carry

            if gscale:
                gs_copy.wait()
            with _scope("experts"):
                jax.lax.fori_loop(0, n_u, ebody, 0)
            # ---- token-sharded reduce of the per-rank partials, then norm + combine
            with _scope("y_rs"):
                if c.y_early:
                    pad = coll.reduce_scatter_rows_finish("y", S["ysend"], Rc, n_live)
                else:
                    pad = coll.reduce_scatter_rows(
                        "y",
                        S["ysend"],
                        Rc,
                        n_live,
                        started=c.y_block_send and c.tp > 1,
                    )
            with _scope("combine"):
                if c.row_bounded_temps:
                    # one owned row at a time: the same per-(row, slot) norm and weighted sum
                    pw = sm[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
                    r0 = pl.multiple_of(me * Rc, Rc) if c.tp > 1 else 0
                    wk = jnp.concatenate(
                        [wsel[:, k : k + 1][:, :, None] for k in range(K)], axis=1
                    )  # [R, K, 1]
                    S["wk_s"][...] = jnp.broadcast_to(wk, (R, K, LANE))
                    mine_live = me * Rc < n_live

                    def combine_row(r, carry):
                        ysum = _unpack_bf16x2(pad[0, pl.ds(r, 1)])
                        for d in range(1, c.tp):
                            ysum = ysum + _unpack_bf16x2(
                                pad[d, pl.ds(r, 1)]
                            )  # [1, K, He]
                        ysum = (
                            jnp.where(mine_live, ysum, 0.0)
                            .astype(jnp.bfloat16)
                            .astype(jnp.float32)
                        )
                        ysc = ysum * pw[None]
                        nrm = jax.lax.rsqrt(
                            jnp.mean(jnp.square(ysc), axis=-1, keepdims=True)
                            + c.post_eps
                        )
                        ynorm = (ysc * nrm).astype(jnp.bfloat16).astype(jnp.float32)
                        wrow = S["wk_s"][pl.ds(r0 + r, 1), :, 0:1]  # [1, K, 1]
                        u = jnp.sum(ynorm * wrow, axis=1)  # [1, He]
                        S["u_send"][pl.ds(r, 1), :] = _pack_bf16x2(u)
                        return carry

                    jax.lax.fori_loop(0, Rc, combine_row, 0)
                else:
                    ysum = _unpack_bf16x2(pad[0])
                    for d in range(1, c.tp):
                        ysum = ysum + _unpack_bf16x2(pad[d])  # [Rc, K, He]
                    # rows past n_live received nothing: keep them finite
                    ysum = (
                        jnp.where(me * Rc < n_live, ysum, 0.0)
                        .astype(jnp.bfloat16)
                        .astype(jnp.float32)
                    )
                    pw = sm[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
                    ysc = ysum * pw[None]
                    nrm = jax.lax.rsqrt(
                        jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
                    )
                    ynorm = (ysc * nrm).astype(jnp.bfloat16).astype(jnp.float32)
                    # this device's rows' routing weights
                    r0 = pl.multiple_of(me * Rc, Rc) if c.tp > 1 else 0
                    wk = jnp.concatenate(
                        [wsel[:, k : k + 1][:, :, None] for k in range(K)], axis=1
                    )  # [R, K, 1]
                    S["wk_s"][...] = jnp.broadcast_to(wk, (R, K, LANE))
                    wmine = S["wk_s"][pl.ds(r0, Rc), :, 0:1]  # [Rc, K, 1]
                    u = jnp.sum(ynorm * wmine, axis=1)  # [Rc, He]
                    S["u_send"][...] = _pack_bf16x2(u)
            with _scope("u_ag"):
                padu = coll.all_gather(
                    "u", S["u_send"], me, live=lambda m: m * Rc < n_live
                )
            u_full = jnp.concatenate(
                [_unpack_bf16x2(padu[d]) for d in range(c.tp)], axis=0
            )  # [R, He]
            u_full = jnp.where(row_live, u_full, 0.0)
            with _scope("unpool_mv"):
                f = (
                    ws.mv(u_full, l, "wunpool", nxt, sm=sm)
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
            if nxt is not None:
                smalls_copy(l + 1, (l + 1) % 2).start()

            def gather_rows(f):
                S["f_send"][...] = _pack_bf16x2(f)
                padf = coll.all_gather("f", S["f_send"], tp_rank)
                return jnp.concatenate(
                    [_unpack_bf16x2(padf[d]) for d in range(c.tp)], axis=1
                )  # [R, H]

            def gather_row(f):
                S["row1_send"][:, 0 : Hl // 2] = _pack_bf16x2(f[0:F, :])
                padf = coll.gather_few(
                    "f1", S["row1_send"].at[:, pl.ds(0, Hl // 2)], tp_rank
                )
                return jnp.concatenate(
                    [_unpack_bf16x2(padf[d]) for d in range(c.tp)], axis=1
                )  # [F, H]

            with _scope("f_ag"):
                return single_row(f, gather_rows, gather_row)

        def layer(l, stream, kind, next_layer, prefetched):
            nxt = l + 1 if next_layer else None
            slot = l % 2
            if not prefetched:
                smalls_copy(l, slot).start()
                ws.prologue(l)
            with _scope("smalls_wait"):
                smalls_copy(l, slot).wait()
            sm = S["smalls"][slot]
            use_rope, is_global = flags(l)

            def row(name, n=H):
                return sm[SMALL[name] : SMALL[name] + 1, :n]

            attn_in = _rms(stream, c.rms_eps, row("pre_attn_w")).astype(jnp.bfloat16)
            with _scope("attention"):
                a = attention(l, attn_in, use_rope, is_global, nxt, sm)
            a = a.astype(jnp.bfloat16).astype(jnp.float32)
            stream = row("attn_alpha") * stream + row("attn_beta") * _rms(
                a, c.post_eps
            ).astype(jnp.bfloat16).astype(jnp.float32)
            ffn_in32 = _rms(stream, c.rms_eps, row("pre_ffn_w"))
            with _scope("moe"):
                f = moe(l, ffn_in32, sm, kind, nxt)
            stream = row("ffn_alpha") * stream + row("ffn_beta") * _rms(
                f * row("post_ffn_w"), c.post_eps
            ).astype(jnp.bfloat16).astype(jnp.float32)
            for ci, cl in enumerate(c.capture_layers):
                if isinstance(l, int):
                    if l == cl:
                        aux_out[:, ci * H : (ci + 1) * H] = stream.astype(jnp.bfloat16)
                    continue

                @pl.when(l == cl)
                def _(ci=ci, stream=stream):
                    aux_out[:, ci * H : (ci + 1) * H] = stream.astype(jnp.bfloat16)

            return stream

        # every layer but the last prefetches the next one's smalls and tiles
        state = {"pref": False}

        def emit(l, stream, kind, next_layer):
            stream = layer(l, stream, kind, next_layer, state["pref"])
            state["pref"] = next_layer
            return stream

        for l in range(L):
            if roll_lo is not None and l == roll_lo and l < roll_hi:
                lo, hi = l, roll_hi
                if not state["pref"]:
                    stream = emit(lo, stream, "fp4", next_layer=lo + 1 < L)
                    lo += 1
                if hi > lo:
                    if hi < L:
                        stream = jax.lax.fori_loop(
                            lo, hi, lambda i, v: layer(i, v, "fp4", True, True), stream
                        )
                        state["pref"] = True
                    else:
                        if hi - 1 > lo:
                            stream = jax.lax.fori_loop(
                                lo,
                                hi - 1,
                                lambda i, v: layer(i, v, "fp4", True, True),
                                stream,
                            )
                        state["pref"] = True
                        stream = emit(hi - 1, stream, "fp4", next_layer=False)
            elif roll_lo is None or not (roll_lo <= l < roll_hi):
                stream = emit(l, stream, c.expert_kinds[l], next_layer=l + 1 < L)
        # ---------------- final norm (on the bf16-cast stream, as the native path)
        PH_final = pltpu.make_async_copy(
            W["final_norm"], S["final_norm"], msem.at[R + 2]
        )
        PH_final.start()
        PH_final.wait()
        hidden = _rms(
            stream.astype(jnp.bfloat16).astype(jnp.float32),
            c.rms_eps,
            S["final_norm"][0:1, :],
        )
        hid_out[...] = jnp.where(row_live, hidden, 0.0).astype(jnp.bfloat16)
        if ncap == 0:
            aux_out[...] = jnp.zeros(aux_out.shape, aux_out.dtype)

    return kernel, scratch


def megakernel_call(cfg: MKConfig, axis: str = "model"):
    """``f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv) -> (hidden, aux, newkv)``
    for one device (call inside ``shard_map``)."""
    kernel, scratch = build_kernel(cfg, axis)
    L = cfg.n_layers
    nw = len(WEIGHT_NAMES)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    ncap = len(cfg.capture_layers)
    aux_w = ncap * cfg.hidden if ncap else LANE

    def call(meta, pos, pages_full, pages_swa, emb, *rest):
        weights = rest[:nw]
        experts = rest[nw : nw + 4 * L]
        kv = rest[nw + 4 * L :]
        assert len(kv) == L
        out_shape = [
            jax.ShapeDtypeStruct((cfg.rows, cfg.hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((cfg.rows, aux_w), jnp.bfloat16),
            jax.ShapeDtypeStruct(
                newkv_shape(cfg), jnp.uint32 if cfg.kv_sal else jnp.bfloat16
            ),
        ]
        # The pools are read-only here. With any remote DMA in the kernel XLA treats the call
        # as a collective and copies every aliased operand: the in-place write is kv_write_call.
        f = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=[smem] * 4
            + [vmem]
            + [hbm] * (nw + 4 * L + L),  # emb: bf16 rows, or the f32 stream
            out_specs=[vmem, vmem, hbm],
            scratch_shapes=[s for _, s in scratch],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=cfg.vmem_limit_bytes,
                collective_id=_COLLECTIVE_ID if cfg.tp > 1 else None,
            ),
            name=f"muse_megakernel_r{cfg.rows}_q{cfg.rows_per_req}",
        )
        return f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv)

    return call


def newkv_shape(cfg: MKConfig):
    if cfg.kv_sal:  # new token columns of a page block, [slot * words + word, row]
        return (cfg.n_layers, cfg.sal_words, cfg.rows)
    return (cfg.n_layers, cfg.rows, *cfg.kv_block)


def _kv_write_sal_call(cfg: MKConfig):
    """SEQ_ALONG_LANE pools: a token is one lane column of its page block, so each live request
    read-modify-writes the (at most two) pages its consecutive rows land in, per layer."""
    L, R, ps, q = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req
    gl = set(cfg.global_layers)
    nw = cfg.sal_words

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        blk, new_v, sems = rest[2 * L], rest[2 * L + 1], rest[2 * L + 2]
        n_live = meta_ref[0]
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, ps), 1)
        for j in range(cfg.n_req):

            @pl.when(j * q < n_live)
            def _(j=j):
                first, last = pos_ref[j * q] // ps, pos_ref[j * q + q - 1] // ps
                for l in range(L):
                    tab = pf_ref if l in gl else psw_ref
                    cp = pltpu.make_async_copy(newkv.at[l], new_v, sems.at[2])
                    cp.start()
                    cp.wait()
                    for side in range(2):
                        pg_no = first if side == 0 else last

                        @pl.when((side == 0) | (last != first))
                        def _(pg_no=pg_no):
                            ent = jnp.minimum(
                                j * cfg.pages_per_seq + pg_no, cfg.n_pages_entries - 1
                            )
                            page = _page(tab, ent, cfg.packed_pages)
                            rd = pltpu.make_async_copy(
                                kv_out[l].at[page], blk, sems.at[0]
                            )
                            rd.start()
                            rd.wait()
                            words = blk.bitcast(jnp.uint32).reshape(nw, ps)
                            w = words[...]
                            nv = new_v[...]
                            for t in range(q):
                                r = j * q + t
                                p = pos_ref[r]
                                hit = (
                                    (lane == p % ps) & (p // ps == pg_no) & (r < n_live)
                                )
                                w = jnp.where(hit, nv[:, r : r + 1], w)
                            words[...] = w
                            wr = pltpu.make_async_copy(
                                blk, kv_out[l].at[page], sems.at[1]
                            )
                            wr.start()
                            wr.wait()

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 4 + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[
                pltpu.VMEM(cfg.kv_block, cfg.kv_dtype),
                pltpu.VMEM((nw, R), jnp.uint32),
                pltpu.SemaphoreType.DMA((3,)),
            ],
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(),
            name=f"muse_kv_write_sal_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call


def kv_write_call(cfg: MKConfig):
    """``f(meta, pos, pages_full, pages_swa, newkv, *kv) -> kv``: scatter this step's K/V rows
    into every layer's pool in place (no remote DMA, so XLA aliases the pools without a copy)."""
    if cfg.kv_sal:
        return _kv_write_sal_call(cfg)
    L, R, ps, q = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req
    gl = set(cfg.global_layers)

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        sems = rest[2 * L]
        n_live = meta_ref[0]
        cps = []
        for l in range(L):
            tab = pf_ref if l in gl else psw_ref
            flat = kv_out[l].reshape(kv_out[l].shape[0] * ps, *kv_out[l].shape[2:])
            for r in range(R):
                p = pos_ref[r]
                ent = jnp.minimum(
                    (r // q) * cfg.pages_per_seq + p // ps, cfg.n_pages_entries - 1
                )
                tok = _page(tab, ent, cfg.packed_pages) * ps + p % ps
                cp = pltpu.make_async_copy(
                    newkv.at[l, pl.ds(r, 1)], flat.at[pl.ds(tok, 1)], sems.at[l * R + r]
                )

                @pl.when(r < n_live)
                def _(cp=cp):
                    cp.start()

                cps.append((r, cp))
        for r, cp in cps:

            @pl.when(r < n_live)
            def _(cp=cp):
                cp.wait()

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 4 + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[pltpu.SemaphoreType.DMA((L * R,))],
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(),
            name=f"muse_kv_write_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call
