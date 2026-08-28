# SPDX-License-Identifier: Apache-2.0
"""TPU RPA-seq prefill specialized for the stacked rank-4 KV layout."""

import enum
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.layers.attention.pallas.ragged_paged_attention_seq.tuned_block_sizes import (
    resolve_block_sizes,
)
from sglang.srt.layers.attention.pallas.ragged_paged_attention_seq.util import (
    align_to,
    cdiv,
    get_dtype_packing,
    next_power_of_2,
)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
LOG2E = 1.4426950408889634

# Scoped VMEM budget requested by the RPA Pallas kernel.
# TPU v7x exposes a hard 64 MiB scoped-VMEM cap per core. The previous
# value (120 MiB) exceeded that cap; libtpu silently truncated the
# backend_config to 64 MiB while Mosaic codegen still emitted offsets
# assuming the full 120 MiB, producing out-of-bounds VMEM accesses
# (manifested as TC-sequencer halts on long-context decode). Keep this
# strictly under 64 MiB so the requested budget is honored as-is.
DEFAULT_VMEM_LIMIT_BYTES = 60 * 1024 * 1024


class KVLayout(enum.StrEnum):
    """KV cache memory layout.

    - HEAD_ALONG_SUBLANE: head_dim lives on the lane (minor) axis and is padded
      to 128 lanes. This is the default and matches the existing kernel.
    - SEQ_ALONG_LANE: the sequence (page_size, any positive multiple of 128)
      lives on the lane axis and head_dim lives on the sublane axis, aligned only to
      ``num_sublanes * kv_packing`` (8 or 16). This avoids padding head_dim up to
      128 lanes, removing KV-cache HBM waste for head dims like 80/96/192.
    """

    HEAD_ALONG_SUBLANE = enum.auto()
    SEQ_ALONG_LANE = enum.auto()


def _normalize_kv_head_scale(scale, num_kv_heads):
    """KV scale -> f32[num_kv_heads]: None->ones, scalar->broadcast, array->as-is."""
    if scale is None:
        return jnp.ones((num_kv_heads,), jnp.float32)
    arr = jnp.asarray(scale, jnp.float32)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (num_kv_heads,))
    return arr


def detect_is_hd64(actual_num_kv_heads, actual_head_dim, kv_cache):
    """Auto-detect whether the kv_cache uses the packed hd64 layout.

    head_dim=64 always uses the packed hd64 layout where K and V are
    concatenated along the head dimension (cache_head_dim=128).
    """
    if actual_head_dim == 64:
        assert (
            kv_cache.shape[-1] == 128
        ), f"hd64 requires cache_head_dim=128, got {kv_cache.shape[-1]}"
        return True
    kv_packing = get_dtype_packing(kv_cache.dtype)
    third_dim = kv_cache.shape[2] * kv_packing
    expected = align_to(actual_num_kv_heads * 2, kv_packing)
    assert third_dim == expected, (
        f"Standard layout: expected dim2*packing={expected}, "
        f"got {third_dim} (num_kv_heads={actual_num_kv_heads}, packing={kv_packing})"
    )
    return False


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
    *,
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE,
):
    kv_packing = get_dtype_packing(kv_dtype)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        # Sequence on the lane axis (page_size any positive multiple of 128),
        # head_dim on the sublane axis aligned only to num_sublanes * kv_packing
        # (no padding to 128 lanes). page_size occupies the lane (minor) axis, so
        # it must be a whole number of 128-lane tiles.
        if page_size <= 0 or page_size % 128 != 0:
            raise ValueError(
                "SEQ_ALONG_LANE requires page_size to be a positive multiple of "
                f"128, got {page_size=}."
            )
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        return (
            total_num_pages,
            actual_num_kv_heads * 2,
            align_to(actual_head_dim, num_sublanes * kv_packing),
            page_size,
        )
    if actual_head_dim == 64:
        return (
            total_num_pages,
            page_size,
            align_to(actual_num_kv_heads, kv_packing) // kv_packing,
            kv_packing,
            128,
        )
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


# ---------------------------------------------------------------------------
# SEQ_ALONG_LANE helpers (sequence on the lane axis, head_dim on the sublane).
#
# Cache shape: [pages, 2*kv_heads, hd_aligned, page_size] where the 2*kv_heads
# axis is interleaved [k0, v0, k1, v1, ...], head_dim occupies the sublane axis
# and page_size the lane (minor) axis. This is the same rank-4 layout the
# stacked KV cache uses, so stacked can hand its cache to this kernel directly.
# ---------------------------------------------------------------------------
def _ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [page_indices_block_size] for the current sequence
    next_page_indices_ref,  # [page_indices_block_size] for the next sequence
    cu_q_lens_ref,  # [max_num_seqs + 1]
    # TODO(jevinjiang): merge these into one so we can save SMEM.
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [8] head: (sem0/1_seq_idx, sem0/1_offset, sem0/1_sz); seq decode reuses these + sem0/1_bkv_idx for deferred writeback
    k_scale_ref,  # [actual_num_kv_heads] f32 per-head K scale (prefetch)
    v_scale_ref,  # [actual_num_kv_heads] f32 per-head V scale (prefetch)
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_hbm_ref,  # [2*kv_heads, hd_aligned, padded_tokens]
    kv_cache_hbm_ref,  # [total_num_pages, 2*kv_heads, hd_aligned, page_size]
    # Output
    o_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    updated_kv_cache_hbm_ref,  # [total_num_pages, 2*kv_heads, hd_aligned, page_size]
    # Scratch
    bkv_x2_ref,  # [2, 2*kv_heads, hd_aligned, bkv_sz + slack*page_size]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    sems,  # [4, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim],
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    has_k_scale: bool = False,
    has_v_scale: bool = False,
    bkv_p,
    bq_sz,
    pages_per_seq: int,
    update_kv_cache: bool = True,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    # Specialization of the general RPA kernel: SEQ_ALONG_LANE layout, causal
    # (with optional sliding window, but no visibility / attention sink), and the
    # combined prefill + mixed region (q_len <= kv_len). No decode.
    (
        actual_num_kv_heads,
        _max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    # SEQ_ALONG_LANE cache: [pages, 2*kv_heads, hd_aligned, page_size].
    # Same rank and layout as the stacked KV cache, so no reshape is needed
    # when stacked delegates its prefill region to this kernel.
    (
        _total_num_pages,
        _two_kv_heads_cache,
        cache_head_dim,
        page_size,
    ) = kv_cache_hbm_ref.shape
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert page_size % num_lanes == 0
    assert kv_hbm_ref.shape[-1] % num_lanes == 0
    # Each program stages its current and next sequence's page-table rows. The
    # next row is required because KV prefetch crosses sequence boundaries.
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    kv_packing = get_dtype_packing(kv_dtype)
    assert head_dim % 128 == 0
    bkv_sz = bkv_p * page_size
    program_seq_idx = -1
    prefill_sw_bound = sliding_window is not None

    def physical_page_index(target_seq_idx, logical_page_idx):
        page_idx = jnp.minimum(logical_page_idx, pages_per_seq - 1)
        return jnp.where(
            target_seq_idx == program_seq_idx,
            page_indices_ref[page_idx],
            next_page_indices_ref[page_idx],
        )

    def get_seq_info(seq_idx, seq_idx_offset):
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens_ref[seq_idx]

        if sliding_window is None:
            bkv_idx_start = next_seq_bkv_idx_start = 0
        else:
            bkv_idx_start = jnp.maximum(kv_len - q_len - sliding_window, 0) // bkv_sz

            # The final program's next-sequence prefetch must stay within the
            # scalar metadata arrays. It is never consumed, but the index still
            # has to be valid while tracing the pipelined loop.
            next_seq_idx = jnp.minimum(seq_idx + 1, num_seqs + seq_idx_offset - 1)
            next_kv_len = kv_lens_ref[next_seq_idx]
            next_q_len = cu_q_lens_ref[next_seq_idx + 1] - q_end
            next_seq_bkv_idx_start = (
                jnp.maximum(next_kv_len - next_q_len - sliding_window, 0) // bkv_sz
            )

        return q_start, q_end, q_len, kv_len, bkv_idx_start, next_seq_bkv_idx_start

    @jax.named_scope("flash_attention_step1_qk_softmax")
    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_sz * num_q_heads_per_kv_head, head_dim]
        kv_or_k,  # [bkv_sz, head_dim] - combined kv (hd64) or k (standard)
        v=None,  # [bkv_sz, head_dim] - only for standard layout
        *,
        bq_idx,
        bkv_idx,
        kv_head_idx,
        skip_mask=False,  # True when the whole KV block is visible to every query
        apply_causal_mask=True,
        apply_window_mask=prefill_sw_bound,
        skip_kv_tail_mask=False,  # True when the whole KV block is below kv_len
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == head_dim
        # K/V come in transposed already as [cache_head_dim, bkv_sz]; the einsum
        # below contracts head_dim, avoiding an explicit transpose on load. K/V
        # are the NATIVE (un-padded) cache head_dim (hd64-native, ported from a2a).
        k_for_dot = kv_or_k
        assert kv_or_k.shape == v.shape == (cache_head_dim, bkv_sz)
        assert kv_or_k.dtype == v.dtype
        # V that step2 consumes for the PV product. Masked to the effective
        # kv_len below when a mask applies, then returned so step2 needs no
        # position logic of its own.
        v_for_pv = v
        head_l_ref = l_ref.at[kv_head_idx, : q.shape[0]]
        head_m_ref = m_ref.at[kv_head_idx, : q.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(
                bkv_idx == bkv_idx_start, jnp.full_like(ref, init_val), ref[...]
            )

        # Follow FlashAttention-2 forward pass.
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k_for_dot.dtype, jnp.floating):
                dtype_info = jnp.finfo(k_for_dot.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k_for_dot.dtype)

        # Slice Q down to the native cache_head_dim so the contraction matches
        # the un-padded K (hd64-native). Mosaic allows this VREG lane slice.
        q_dot = q[:, :cache_head_dim] if cache_head_dim < head_dim else q
        s = jnp.einsum(
            "nd,dm->nm", q_dot, k_for_dot, preferred_element_type=jnp.float32
        )
        combined_scale = sm_scale
        if has_k_scale:
            combined_scale = combined_scale * k_scale_ref[kv_head_idx]
        if q_scale is not None:
            combined_scale *= q_scale
        log2e_scale = LOG2E * combined_scale

        # skip_mask: the caller guarantees every key in this KV block is visible to
        # every query in the Q block, so the iota/compare/where and the defensive
        # V-tail conversion can be skipped.
        if not skip_mask:
            kv_base = bkv_idx * bkv_sz
            k_span = kv_base + lax.broadcasted_iota(jnp.int32, s.shape, 1)

            q_span = (
                kv_len
                - q_len
                + bq_idx * bq_sz
                + lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
            )
            if apply_causal_mask:
                mask = k_span <= q_span
            else:
                assert apply_window_mask
                mask = q_span - sliding_window < k_span

            if not skip_kv_tail_mask:
                # Bound the score to the effective kv_len and mask V to it BEFORE
                # the sliding-window mask, matching upstream v3's ordering.
                mask = jnp.logical_and(mask, k_span < kv_len)
                # The score mask forces p==0 on tail positions, but 0 * Inf/NaN
                # is NaN. Zero the final partial block's V values so the PV sum
                # remains finite. V has sequence on lane axis 1; Mosaic rejects
                # the packed-tile sublane broadcast, so select in f32 and cast back.
                v_span = kv_base + lax.broadcasted_iota(jnp.int32, v_for_pv.shape, 1)
                v_for_pv = jnp.where(
                    v_span < kv_len, v_for_pv.astype(jnp.float32), 0.0
                ).astype(v_for_pv.dtype)
            if apply_window_mask and apply_causal_mask:
                mask = jnp.logical_and(mask, q_span - sliding_window < k_span)
            s = jnp.where(mask, s, mask_value)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)

        m_prev = load_with_init(head_m_ref, -jnp.inf)

        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp2((s - broadcast_minor(m_curr, s.shape)) * log2e_scale)

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp2((m_prev - m_curr) * log2e_scale)
        l_prev = load_with_init(head_l_ref, 0.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr

        return p, exp_m_diff, v_for_pv

    @jax.named_scope("flash_attention_step2_pv")
    def flash_attention_step2_pv(
        q_shape_0,
        v_or_kv,  # v (standard) or combined kv (hd64), pre-masked to kv_len by step1
        p,  # from step1
        exp_m_diff,  # from step1
        *,
        bkv_idx,
        kv_head_idx,
    ):
        head_acc_ref = acc_ref.at[kv_head_idx, :q_shape_0]

        def load_with_init(ref, init_val):
            return jnp.where(
                bkv_idx == bkv_idx_start, jnp.full_like(ref, init_val), ref[...]
            )

        pv = jnp.einsum("nm,dm->nd", p, v_or_kv, preferred_element_type=jnp.float32)
        # v is the native cache_head_dim; pad pv up to the compute head_dim so
        # it matches the (128-padded, lane-tiled) accumulator (hd64-native).
        if cache_head_dim < head_dim:
            pv = jnp.pad(pv, ((0, 0), (0, head_dim - cache_head_dim)))
        if has_v_scale:
            pv *= v_scale_ref[kv_head_idx]
        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    @jax.named_scope("async_copy")
    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        # In-kernel KV writeback: fetch cached tokens as whole cache pages into
        # [0, cache_pages*page_size), then stage the new-token span rounded to
        # 128 lanes in the following slack region. The new tokens are later
        # rolled into place (stitch) and the updated pages are DMA'd back to the
        # cache HBM. Mosaic requires lane-axis DMA offsets and sizes to be
        # 128-aligned, but they do not need to cover a whole cache page.
        kv_len = kv_lens_ref[seq_idx]
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        kv_left = kv_len - kv_len_start
        if update_kv_cache:
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        else:
            kv_left_frm_cache = kv_left
        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        cache_pages = cdiv(bkv_sz_frm_cache, page_size)
        new_kv_len_start = q_end - kv_left_frm_new
        new_dma_offset = new_kv_len_start % num_lanes
        new_dma_base = new_kv_len_start - new_dma_offset
        new_dma_size = jnp.where(
            bkv_sz_frm_new > 0,
            align_to(new_dma_offset + bkv_sz_frm_new, num_lanes),
            0,
        )
        if not wait:
            # A writeback to this slot may have been deferred
            # (start_writeback_seq); wait for it before overwriting the slot
            # (WAR) / reading possibly-updated cache pages (RAW). All kinds
            # defer now (decode and prefill/mixed).
            wait_writeback_seq(bkv_sem_idx)
            with jax.named_scope("fetch_bkv_seq_cached"):
                # Cached pages -> [0, cache_pages).
                for i in range(bkv_p):
                    sz = pl.multiple_of(
                        jnp.where(i < cache_pages, page_size, 0), page_size
                    )
                    page_idx = physical_page_index(seq_idx, kv_p_start + i)
                    _async_copy(
                        kv_cache_hbm_ref.at[page_idx, :, :, pl.ds(0, sz)],
                        vmem_ref.at[:, :, pl.ds(i * page_size, sz)],
                        sem,
                        wait=False,
                    )
            with jax.named_scope("fetch_bkv_seq_stage_new"):
                # The new tokens are contiguous in kv_hbm_ref and in the VMEM
                # slack, so issue one coalesced DMA for the smallest 128-lane
                # window containing them. The input is padded to 128 lanes, so
                # this window cannot run past the HBM buffer.
                src = pl.multiple_of(new_dma_base, num_lanes)
                total_new = pl.multiple_of(new_dma_size, num_lanes)
                dst_off = pl.multiple_of(cache_pages * page_size, page_size)

                @pl.when(bkv_sz_frm_new > 0)
                def _stage_new():
                    _async_copy(
                        kv_hbm_ref.at[:, :, pl.ds(src, total_new)],
                        vmem_ref.at[:, :, pl.ds(dst_off, total_new)],
                        sem,
                        wait=False,
                    )

        else:
            total = pl.multiple_of(
                cache_pages * page_size + new_dma_size,
                num_lanes,
            )
            dst = vmem_ref.at[:, :, pl.ds(0, total)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)
        return kv_len_start, bkv_sz_frm_new

        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        if update_kv_cache:
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    @jax.named_scope("fetch_bq")
    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    @jax.named_scope("send_bo")
    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    @jax.named_scope("start_fetch_bkv")
    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    @jax.named_scope("wait_fetch_bkv")
    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    @jax.named_scope("start_fetch_bq")
    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    @jax.named_scope("wait_fetch_bq")
    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    @jax.named_scope("start_send_bo")
    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    @jax.named_scope("wait_send_bo")
    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(seq_idx_offset <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    @jax.named_scope("load_bq")
    def load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        return pltpu.bitcast(
            q_ref[: actual_bq_sz * num_q_heads_per_kv_head_per_packing], q_dtype
        )

    @jax.named_scope("load_bkv_seq")
    def load_bkv_seq(bkv_sem_idx, kv_head_idx):
        """Load one KV head from the SEQ_ALONG_LANE buffer as [head_dim, bkv_sz].

        The buffer is [2, 2*kv_heads, hd_aligned, bkv_sz]. For head h, the K rows
        live at 2h and V rows at 2h+1, so the slice is already [head_dim, bkv_sz]
        and needs no reshape (the old (hd//pk, pk)->hd merge forced a relayout).
        We keep the [head_dim, bkv_sz] orientation (head on sublanes, seq on
        lanes) and let the seq-layout QK/PV einsums contract head_dim directly --
        this avoids the explicit seq<->head_dim transpose, which was a VPU op on
        the critical path. head_dim is padded (on the sublane axis) up to the
        128-aligned compute head_dim.
        """
        with jax.named_scope("load_bkv_seq_read"):
            k = bkv_x2_ref[bkv_sem_idx, 2 * kv_head_idx, :, :bkv_sz][...]
            v = bkv_x2_ref[bkv_sem_idx, 2 * kv_head_idx + 1, :, :bkv_sz][...]
        return k, v

    @jax.named_scope("stitch_seq")
    def stitch_seq(bkv_sem_idx, bkv_sz_cache, cache_pages, new_dma_offset):
        """Roll staged new-token lanes into place in the bkv buffer (lane axis).

        Mirrors the batched SEQ_ALONG_LANE stitch: single-token decode
        (decode_q_len == 1) rolls only the boundary VREG (O(1)); spec decode
        (decode_q_len > 1) rolls a bounded 3-page window at the boundary
        (O(decode_q_len)); prefill/mixed roll the whole staged region (O(N)).
        Operates on the u32 bitcast view so head/head_dim structure collapses
        into the row axis.
        """
        # The cached and new-token regions span at most bkv_p+2 pages in total
        # when both ends are misaligned. Two slack pages are therefore sufficient
        # for every prefill/mixed split. This must match the bkv_double_buf
        # allocation in _ragged_paged_attention_impl.
        stage_slack_pages = 2
        v_len = bkv_sz + stage_slack_pages * page_size
        # With the rank-4 cache the bitcast view is already [2*kv_heads,
        # hd_aligned // packing, lanes] -- rows on the sublane axis, tokens on
        # the lane axis -- so the roll applies directly to axis 2. (The rank-5
        # layout needed an extra reshape here to collapse the packing axis;
        # that reshape regrouped sublanes and Mosaic rejects it on rank 4.)
        vmem_u32 = bkv_x2_ref.at[bkv_sem_idx].bitcast(jnp.uint32)
        roll_shift = (bkv_sz_cache - (cache_pages * page_size + new_dma_offset)) % v_len

        # Skip the O(N) roll when the staged data is already aligned
        # (roll_shift == 0, e.g. fresh / page-aligned prefill): the roll would
        # be an identity over the whole buffer at full VPU cost.
        @pl.when(roll_shift != 0)
        def _do_roll():
            rolled = pltpu.roll(vmem_u32[...], roll_shift, axis=2)
            lane_idx = lax.broadcasted_iota(jnp.int32, rolled[..., :bkv_sz].shape, 2)
            merged = lax.select(
                lane_idx >= bkv_sz_cache,
                rolled[..., :bkv_sz],
                vmem_u32[..., :bkv_sz],
            )
            vmem_u32[..., :bkv_sz] = merged

    def _writeback_seq_copy(seq_idx, bkv_idx, bkv_sem_idx, p):
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]
        kv_p_start = bkv_idx * bkv_p
        page_idx = physical_page_index(seq_idx, kv_p_start + p)
        return pltpu.make_async_copy(
            vmem_ref.at[
                :, :, pl.ds(pl.multiple_of(p * page_size, page_size), page_size)
            ],
            kv_cache_hbm_ref.at[page_idx],
            sems.at[3, bkv_sem_idx],
        )

    @jax.named_scope("start_writeback_seq")
    def start_writeback_seq(
        seq_idx, bkv_idx, bkv_sem_idx, bkv_sz_cache, bkv_sz_frm_new
    ):
        """Issue the writeback DMAs async and record the pending op (all kinds).

        The wait is deferred to wait_writeback_seq (before the slot is reused, and
        in the epilogue) so the writeback overlaps later compute. Safe for decode
        AND prefill/mixed: WAR (slot reuse) is guarded by the fetch-time
        wait_writeback_seq and the epilogue flush; there is no within-call RAW
        (new tokens are consumed from VMEM / re-staged from the input kv_hbm, never
        re-read from cache this call); and every bkv block -- including
        causally-masked ones routed through compute_with_bkv_no_attn -- is
        persisted at bq=0 (see _seq_persist_bkv).

        Pending state is recorded in bkv_update_ids_ref[bkv_sem_idx + {0,2,4,6}] =
        (seq_idx, bkv_sz_cache, bkv_sz_frm_new, bkv_idx); a non-zero +4 slot
        (bkv_sz_frm_new) flags a pending writeback for wait_writeback_seq.
        """
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = bkv_sz_cache
        bkv_update_ids_ref[bkv_sem_idx + 4] = bkv_sz_frm_new
        bkv_update_ids_ref[bkv_sem_idx + 6] = bkv_idx
        first_wb = bkv_sz_cache // page_size
        last_wb = (bkv_sz_cache + bkv_sz_frm_new - 1) // page_size
        n_wb = last_wb - first_wb + 1

        def start_body(i, _):
            _writeback_seq_copy(seq_idx, bkv_idx, bkv_sem_idx, first_wb + i).start()

        lax.fori_loop(0, n_wb, start_body, None, unroll=False)

    @jax.named_scope("wait_writeback_seq")
    def wait_writeback_seq(bkv_sem_idx):
        bkv_sz_frm_new = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(bkv_sz_frm_new > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            bkv_sz_cache = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_idx = bkv_update_ids_ref[bkv_sem_idx + 6]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            first_wb = bkv_sz_cache // page_size
            last_wb = (bkv_sz_cache + bkv_sz_frm_new - 1) // page_size
            n_wb = last_wb - first_wb + 1

            def wait_body(i, _):
                _writeback_seq_copy(seq_idx, bkv_idx, bkv_sem_idx, first_wb + i).wait()

            lax.fori_loop(0, n_wb, wait_body, None, unroll=False)

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        # no-op concatenation.
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])], axis=-1
        )[..., : shape[-1]]

    @jax.named_scope("process")
    def process(static_q_len=None):
        prefetch_num_bkv = num_bkv = cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        def _sw_bq_start(bq_idx):
            # Start each query block at the KV block containing its own window
            # lower bound. This avoids revisiting blocks that are fully masked
            # for later query blocks in the same prefill chunk.
            return (
                jnp.maximum(kv_len - q_len + bq_idx * actual_bq_sz - sliding_window, 0)
                // bkv_sz
            )

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        @jax.named_scope("get_next_bkv_ids")
        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == prefetch_num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            if sliding_window is None:
                next_bkv_idx_start = 0
            else:
                same_seq_start = (
                    _sw_bq_start(next_bq_idx) if prefill_sw_bound else bkv_idx_start
                )
                next_bkv_idx_start = lax.select(
                    is_last_bq,
                    next_seq_bkv_idx_start,
                    same_seq_start,
                )
            next_bkv_idx = lax.select(is_last_bkv, next_bkv_idx_start, next_bkv_idx)

            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        @jax.named_scope("compute_with_bq")
        def compute_with_bq(bq_idx, _):
            # Tighten the sliding-window lower bound to THIS q-block's window.
            # bkv_idx_start is a kernel-body nonlocal read by the flash-init
            # conditions, wait_cur_bq, and the bkv loop, so rebinding it here
            # propagates the tightened start to all of them (same pattern as the
            # nonlocal prefetch_num_bkv below). bq 0 keeps the seq-level start.
            nonlocal bkv_idx_start
            if prefill_sw_bound:
                bkv_idx_start = _sw_bq_start(bq_idx)
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            # Prefetch next bq
            @pl.when(next_seq_idx < num_seqs + seq_idx_offset)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @jax.named_scope("prefetch_with_bkv")
            def prefetch_with_bkv(bkv_idx):
                # Create bitmask for KV.
                assert bkv_sz % kv_packing == 0

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx
                )

                # Prefetch next bkv
                @pl.when(next_seq_idx < num_seqs + seq_idx_offset)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == bkv_idx_start)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                _offset, _update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                # Start updating bkv to kv cache if applicable.
                # Only needed in first bq loop.
                # SEQ_ALONG_LANE does its own in-kernel stitch + writeback inside
                # compute_with_bkv_seq, so there is no row writeback here.
                return bkv_sem_idx

            def _compute_with_bkv_preamble(bkv_idx):
                """Shared prefetch + debug logic for compute_with_bkv variants."""
                bkv_sem_idx = prefetch_with_bkv(bkv_idx)
                return bkv_sem_idx

            @jax.named_scope("flush_prev_pv")
            def _flush_prev_pv(
                prev_bq_shape_0,
                prev_kv_head_bkv,
                prev_kv_head_p,
                prev_kv_head_exp_m_diff,
                prev_kv_head_idx,
                bkv_idx,
            ):
                """Execute pv of last attention head."""
                assert prev_bq_shape_0 is not None
                flash_attention_step2_pv(
                    prev_bq_shape_0,
                    prev_kv_head_bkv,
                    prev_kv_head_p,
                    prev_kv_head_exp_m_diff,
                    bkv_idx=bkv_idx,
                    kv_head_idx=prev_kv_head_idx,
                )

            def _seq_persist_bkv(bkv_idx, bkv_sem_idx):
                """Stitch staged new-token pages into the bkv buffer (every bq
                block, needed for the attention read) and write the updated whole
                pages back to the cache HBM in-place (first bq block only).

                Called for BOTH attended (compute_with_bkv_seq) and causally-masked
                (compute_with_bkv_no_attn) blocks, so every bkv block's new KV is
                persisted at bq=0 -- masked blocks are only reached at bq>0 by the
                attention loop, where the writeback is gated off, so without this
                the cache would be left incomplete for multi-bq prefill.
                """
                kv_len_start = bkv_idx * bkv_sz
                kv_left = kv_len - kv_len_start
                if update_kv_cache:
                    kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
                else:
                    kv_left_frm_cache = kv_left
                kv_left_frm_new = kv_left - kv_left_frm_cache
                bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
                bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
                cache_pages = cdiv(bkv_sz_frm_cache, page_size)
                new_kv_len_start = cu_q_lens_ref[seq_idx + 1] - kv_left_frm_new
                new_dma_offset = new_kv_len_start % num_lanes

                @pl.when(bkv_sz_frm_new > 0)
                def _do_stitch():
                    stitch_seq(
                        bkv_sem_idx, bkv_sz_frm_cache, cache_pages, new_dma_offset
                    )

                if update_kv_cache:

                    @pl.when(jnp.logical_and(bkv_sz_frm_new > 0, bq_idx == 0))
                    def _do_writeback():
                        # Defer the wait (async) for ALL kinds so the writeback
                        # overlaps later compute. WAR/RAW are guarded by the
                        # fetch-time wait_writeback_seq + the epilogue flush, and
                        # every bkv block -- including causally-masked ones routed
                        # through compute_with_bkv_no_attn -- is persisted at bq=0.
                        start_writeback_seq(
                            seq_idx,
                            bkv_idx,
                            bkv_sem_idx,
                            bkv_sz_frm_cache,
                            bkv_sz_frm_new,
                        )

            @jax.named_scope("compute_with_bkv_seq")
            def compute_with_bkv_seq(
                bkv_idx,
                _,
                skip_mask=False,
                apply_causal_mask=True,
                apply_window_mask=prefill_sw_bound,
                skip_kv_tail_mask=False,
            ):
                bkv_sem_idx = _compute_with_bkv_preamble(bkv_idx)
                _seq_persist_bkv(bkv_idx, bkv_sem_idx)

                prev_bq_shape_0 = None
                prev_kv_head_bkv = None
                prev_kv_head_idx = None
                prev_kv_head_p = None
                prev_kv_head_exp_m_diff = None

                for cur_kv_head_idx in range(actual_num_kv_heads):
                    cur_kv_head_bq = load_bq(
                        bq_sem_idx, cur_kv_head_idx, actual_bq_sz=actual_bq_sz
                    )
                    bk, bv = load_bkv_seq(bkv_sem_idx, cur_kv_head_idx)
                    cur_kv_head_p, cur_kv_head_exp_m_diff, bv = (
                        flash_attention_step1_qk_softmax(
                            cur_kv_head_bq,
                            bk,
                            bv,
                            bq_idx=bq_idx,
                            bkv_idx=bkv_idx,
                            kv_head_idx=cur_kv_head_idx,
                            skip_mask=skip_mask,
                            apply_causal_mask=apply_causal_mask,
                            apply_window_mask=apply_window_mask,
                            skip_kv_tail_mask=skip_kv_tail_mask,
                        )
                    )
                    if prev_bq_shape_0 is not None:
                        flash_attention_step2_pv(
                            prev_bq_shape_0,
                            prev_kv_head_bkv,
                            prev_kv_head_p,
                            prev_kv_head_exp_m_diff,
                            bkv_idx=bkv_idx,
                            kv_head_idx=prev_kv_head_idx,
                        )
                    prev_bq_shape_0 = cur_kv_head_bq.shape[0]
                    prev_kv_head_bkv = bv
                    prev_kv_head_p = cur_kv_head_p
                    prev_kv_head_exp_m_diff = cur_kv_head_exp_m_diff
                    prev_kv_head_idx = cur_kv_head_idx

                _flush_prev_pv(
                    prev_bq_shape_0,
                    prev_kv_head_bkv,
                    prev_kv_head_p,
                    prev_kv_head_exp_m_diff,
                    prev_kv_head_idx,
                    bkv_idx,
                )

            @jax.named_scope("compute_with_bkv_no_attn")
            def compute_with_bkv_no_attn(bkv_id, _):
                # Persist (stitch + writeback) this causally-masked block's new
                # KV to the cache. Head persists inside prefetch_with_bkv (so
                # this path covers it for free), but seq's writeback lives
                # outside prefetch, so without doing it here multi-bq prefill
                # would leave these blocks unwritten in the cache.
                bkv_sem_idx = _compute_with_bkv_preamble(bkv_id)
                _seq_persist_bkv(bkv_id, bkv_sem_idx)

            compute_with_bkv_fn = compute_with_bkv_seq

            def compute_masked_range(
                start,
                end,
                *,
                apply_causal_mask=True,
                apply_window_mask=prefill_sw_bound,
                may_include_kv_tail=False,
            ):
                # Every block except the final physical KV block is completely
                # below kv_len. Avoid constructing the kv_len score mask and the
                # expensive packed V f8->f32->f8 tail select for those blocks.
                if not may_include_kv_tail:
                    lax.fori_loop(
                        start,
                        end,
                        lambda i, c: compute_with_bkv_fn(
                            i,
                            c,
                            apply_causal_mask=apply_causal_mask,
                            apply_window_mask=apply_window_mask,
                            skip_kv_tail_mask=True,
                        ),
                        None,
                        unroll=False,
                    )
                    return

                # floor(kv_len / bkv_sz) is the first partial block. It equals
                # num_bkv when kv_len is block-aligned, so aligned sequences skip
                # the tail path entirely instead of redundantly masking a full
                # final block.
                tail_start = jnp.clip(kv_len // bkv_sz, start, end)
                lax.fori_loop(
                    start,
                    tail_start,
                    lambda i, c: compute_with_bkv_fn(
                        i,
                        c,
                        apply_causal_mask=apply_causal_mask,
                        apply_window_mask=apply_window_mask,
                        skip_kv_tail_mask=True,
                    ),
                    None,
                    unroll=False,
                )
                lax.fori_loop(
                    tail_start,
                    end,
                    lambda i, c: compute_with_bkv_fn(
                        i,
                        c,
                        apply_causal_mask=apply_causal_mask,
                        apply_window_mask=apply_window_mask,
                    ),
                    None,
                    unroll=False,
                )

            # Limit KV block processing to avoid fully masked regions.
            # This optimization skips KV blocks beyond the last query position in the
            # current batch, avoiding unnecessary computation for fully masked regions.
            # For causal masking queries can only attend to keys at positions <= their own position.
            adjusted_num_bkv = jnp.minimum(
                cdiv(kv_len - q_len + bq_idx * bq_sz + actual_bq_sz, bkv_sz),
                num_bkv,
            )
            nonlocal prefetch_num_bkv
            prefetch_num_bkv = lax.select(bq_idx == 0, num_bkv, adjusted_num_bkv)

            # Mask-skip: split the attended KV range around blocks that are fully
            # visible to every query in this Q block. The upper edge is causal:
            # block_max <= min_q. For sliding-window attention the lower edge is
            # block_min > max_q - window. Boundary blocks still run the exact mask;
            # the prefetch chain stays continuous across all loop splits.
            q_block_start = kv_len - q_len + bq_idx * bq_sz
            visible_end = jnp.clip(
                (q_block_start + 1) // bkv_sz,
                bkv_idx_start,
                adjusted_num_bkv,
            )
            if sliding_window is None:
                lax.fori_loop(
                    bkv_idx_start,
                    visible_end,
                    lambda i, c: compute_with_bkv_fn(i, c, skip_mask=True),
                    None,
                    unroll=False,
                )
                compute_masked_range(
                    visible_end,
                    adjusted_num_bkv,
                    may_include_kv_tail=True,
                )
            else:
                # The latest query has the tightest sliding-window lower bound.
                # Start the mask-free region at the first block whose minimum key
                # is strictly above that bound. This is a block-level optimization;
                # the exact element mask remains on both boundary regions.
                window_visible_start = jnp.clip(
                    cdiv(
                        jnp.maximum(
                            q_block_start + actual_bq_sz - sliding_window,
                            0,
                        ),
                        bkv_sz,
                    ),
                    bkv_idx_start,
                    adjusted_num_bkv,
                )
                first_boundary_end = jnp.minimum(window_visible_start, visible_end)
                boundary_split_end = jnp.maximum(window_visible_start, visible_end)
                compute_masked_range(
                    bkv_idx_start,
                    first_boundary_end,
                    apply_causal_mask=False,
                    apply_window_mask=True,
                )
                lax.fori_loop(
                    window_visible_start,
                    boundary_split_end,
                    lambda i, c: compute_with_bkv_fn(i, c, skip_mask=True),
                    None,
                    unroll=False,
                )
                # If the window spans at least one KV block beyond a complete Q
                # block, the two boundary masks cannot overlap for any alignment.
                # Avoid materializing an otherwise unreachable copy of the full
                # attention body in that common case. Narrower windows retain the
                # exact overlap path.
                if sliding_window < bq_sz + bkv_sz:
                    compute_masked_range(
                        visible_end,
                        boundary_split_end,
                        apply_causal_mask=True,
                        apply_window_mask=True,
                    )
                compute_masked_range(
                    boundary_split_end,
                    adjusted_num_bkv,
                    apply_causal_mask=True,
                    apply_window_mask=False,
                    may_include_kv_tail=True,
                )

            @pl.when(bq_idx == 0)
            def _():
                lax.fori_loop(
                    adjusted_num_bkv,
                    num_bkv,
                    compute_with_bkv_no_attn,
                    None,
                    unroll=False,
                )

            # Load acc and calculate final output.
            with jax.named_scope("output_epilogue"):
                acc = acc_ref[...]
                l = broadcast_minor(l_ref[...], acc.shape)
                out = (
                    lax.div(acc, l)
                    if q_dtype == jnp.float32
                    else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)
                )

                # Wait for previous bo to be fully sent before storing new bo.
                bo_sem_idx = sem_ids_ref[2]
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
                wait_send_bo(bo_sem_idx)

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                    actual_num_kv_heads,
                    bq_sz * num_q_heads_per_kv_head_per_packing,
                    head_dim,
                )[...] = pltpu.bitcast(out, jnp.int32)

                # Send cur bo
                start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    def run_kernel(_seq_idx, _):
        nonlocal q_len, kv_len, bkv_idx_start, next_seq_bkv_idx_start, seq_idx
        nonlocal program_seq_idx

        seq_idx = _seq_idx + seq_idx_offset
        program_seq_idx = seq_idx
        _q_start, _q_end, q_len, kv_len, bkv_idx_start, next_seq_bkv_idx_start = (
            get_seq_info(seq_idx, seq_idx_offset)
        )

        @pl.when(_seq_idx == 0)
        def prologue():
            with jax.named_scope("prologue"):
                start_fetch_bq(seq_idx, 0, 0)

                # Initialize bkv_x2_ref to zeros to avoid NaN issues from accessing
                # uninitialized memory. Bitcast into int32 to avoid tiling issues.
                # Zero in the buffer's natural shape: with the rank-4 cache a
                # reshape to (2, -1, 8, 128) is a relayout that Mosaic rejects.
                bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32)
                zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)

                # To pipeline VST and DMA, we divide the initialization into two steps.
                bkv_x2_int32_ref[0] = zeros
                start_fetch_bkv(seq_idx, bkv_idx_start, 0)
                bkv_x2_int32_ref[1] = zeros

        # Single region: prefill + mixed (q_len <= kv_len). num_bq is dynamic.
        process()

        @pl.when(_seq_idx == num_seqs - 1)
        def epilogue():
            with jax.named_scope("epilogue"):
                for i in range(2):
                    wait_send_bo(i)
                    # Prefill/mixed write back synchronously; this is a no-op
                    # unless a writeback is still outstanding.
                    wait_writeback_seq(i)

    q_len = kv_len = bkv_idx_start = next_seq_bkv_idx_start = seq_idx = -1
    # The grid covers distribution[0]..distribution[2], i.e. every non-decode
    # sequence, so sequence indices are offset past the decode region.
    seq_idx_offset = distribution_ref[0]
    num_seqs = pl.num_programs(0)
    run_kernel(pl.program_id(0), num_seqs)

    ### ------- Kernel end ------- ###


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    is_hd64=False,
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    if is_hd64:
        num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
        kv = jnp.pad(
            jnp.concat([k, v], axis=-1),
            (
                (0, 0),
                (0, num_kv_heads - actual_num_kv_heads),
                (0, 0),
            ),
            constant_values=0,
        ).reshape(
            max_num_tokens,
            num_kv_heads // kv_packing,
            kv_packing,
            actual_head_dim * 2,
        )
    else:
        actual_num_kv_heads_x2 = actual_num_kv_heads * 2
        num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)
        head_dim = align_to(actual_head_dim, 128)
        kv = jnp.pad(
            jnp.concat([k, v], axis=-1).reshape(
                max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
            ),
            (
                (0, 0),
                (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            max_num_tokens,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            head_dim,
        )
    return kv


def _seq_along_lane_new_kv(k, v):
    """Build the new K/V in SEQ_ALONG_LANE layout for in-kernel DMA writeback.

    Returns [2*kv_heads, hd_aligned, padded_tokens] (tokens on the lane
    axis). Tokens are padded only to the TPU lane count because the kernel stages
    new K/V with 128-lane-aligned DMAs rather than whole cache-page DMAs.
    """
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    hd_aligned = align_to(actual_head_dim, num_sublanes * kv_packing)
    padded = align_to(max_num_tokens, num_lanes)
    merged = jnp.concatenate([k, v], axis=-1).reshape(
        max_num_tokens, actual_num_kv_heads * 2, actual_head_dim
    )
    merged = jnp.pad(
        merged,
        ((0, padded - max_num_tokens), (0, 0), (0, hd_aligned - actual_head_dim)),
        constant_values=0,
    )
    return merged.transpose(1, 2, 0)


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    attention_sink: jax.Array | None = None,
    is_hd64=False,
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE,
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        )
        # TODO(jevinjiang): Explore fusing swapping non-tiling axis to DMA.
        .swapaxes(0, 1)
    )
    # TODO(kyuyeunk, chengjiyao): Add kv quantization here.
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        kv = _seq_along_lane_new_kv(k, v)
    else:
        kv = merge_kv(k, v, is_hd64=is_hd64)

    if attention_sink is not None:
        attention_sink = attention_sink.reshape((-1, num_q_heads_per_kv_head, 1))
        attention_sink = jnp.repeat(attention_sink, 128, -1)

    return q, kv, attention_sink


def prepare_outputs(
    out,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    is_hd64=False,
):
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    if is_hd64:
        # For hd64, output is in the second half of the head_dim (value portion).
        return (
            out.swapaxes(0, 1)
            .reshape(
                max_num_tokens,
                actual_num_kv_heads,
                num_q_heads_per_kv_head_per_q_packing * q_packing,
                head_dim,
            )[:, :, :actual_num_q_heads_per_kv_head, actual_head_dim:]
            .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
        )
    return (
        out.swapaxes(0, 1)
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing * q_packing,
            head_dim,
        )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim]
        .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
    )


# Expect to run this validation during runtime.
def dynamic_validate_inputs(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # SEQ_ALONG_LANE: [total_num_pages, 2*kv_heads, hd_aligned, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sink: jax.Array | None = None,  # f32[actual_num_q_heads]
    visibility: jax.Array | None = None,  # i32[max_num_tokens, 2]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
):
    q, k, v = queries, keys, values
    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink,
        visibility,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )

    # Expect to run this validation during compile time.
    max_num_tokens = q.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    i, j, k = distribution
    if not (i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")

    if k > max_num_seqs:
        raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")

    if cu_q_lens[k] > max_num_tokens:
        raise ValueError(f"Total q tokens {cu_q_lens[k]} must be <= {max_num_tokens=}.")
    for i in range(k):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        kv_len = kv_lens[i]
        if not (0 < q_len <= kv_len):
            raise ValueError(f"Require 0 < {q_len=} <= {kv_len=} at sequence {i}.")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Require {page_cnt=} <= {pages_per_seq=} at sequence {i} where"
                f" {kv_len=} and {page_size=}."
            )
        for p in range(page_cnt):
            page_idx = page_indices[i * pages_per_seq + p]
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence"
                    f" {i} where {kv_len=} and {page_size=}."
                )


# Expect to run this validation during compile time.
def static_validate_inputs(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # SEQ_ALONG_LANE: [total_num_pages, 2*kv_heads, hd_aligned, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sink: jax.Array | None = None,  # f32[actual_num_q_heads]
    visibility: jax.Array | None = None,  # i32[max_num_tokens, 2]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the RPA kernel statically."""
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(
            f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}"
        )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(
            f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}"
        )

    if attention_sink is not None:
        if attention_sink.shape[0] != q.shape[1]:
            raise ValueError(
                f"Expected {attention_sink.shape[0]=} to be equal to"
                f" {q.shape[1]=} (num_q_heads)."
            )
        if attention_sink.dtype != jnp.float32:
            raise ValueError(
                f"Expected {attention_sink.dtype=} to be equal to {jnp.float32=}."
            )

    if visibility is not None:
        if visibility.shape != (q.shape[0], 2):
            raise ValueError(f"Expected {visibility.shape=} to be ({q.shape[0]}, 2).")
        if visibility.dtype != jnp.int32:
            raise ValueError(f"Expected {visibility.dtype=} to be {jnp.int32=}.")
        if sliding_window is not None:
            raise ValueError("visibility and sliding_window are mutually exclusive.")

    actual_head_dim = q.shape[2]
    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(
            f"Expected {actual_num_q_heads=} to be divisible by {actual_num_kv_heads=}."
        )

    is_hd64 = detect_is_hd64(actual_num_kv_heads, actual_head_dim, kv_cache)

    (
        _,
        page_size,
        num_kv_heads_per_kv_packing,
        kv_packing,
        head_dim,
    ) = kv_cache.shape

    if is_hd64:
        if actual_head_dim != 64:
            raise ValueError(f"Expected {actual_head_dim=} to be 64 for hd64 layout.")
        if head_dim != 128:
            raise ValueError(f"Expected {head_dim=} to be 128 for hd64 layout.")
        num_kv_heads = num_kv_heads_per_kv_packing * kv_packing
        if align_to(actual_num_kv_heads, kv_packing) != num_kv_heads:
            raise ValueError(
                f"Invalid {num_kv_heads=}, {actual_num_kv_heads=}, {kv_packing=}"
            )
    else:
        if head_dim != align_to(actual_head_dim, 128):
            raise ValueError(
                f"Expected {head_dim=} is equal to {align_to(actual_head_dim, 128)=}"
            )
        num_kv_heads_x2 = num_kv_heads_per_kv_packing * kv_packing
        if num_kv_heads_x2 % 2 != 0:
            raise ValueError(
                f"Combined KV heads must be divisible by 2, but got {num_kv_heads_x2}"
            )
        if align_to(actual_num_kv_heads * 2, kv_packing) != num_kv_heads_x2:
            raise ValueError(
                f"Invalid {num_kv_heads_x2=}, {actual_num_kv_heads=}, {kv_packing=}"
            )

    # Note: we expect the kv quantization happens outside of the RPA kernel.
    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"Expected {kv_cache.dtype=} to be equal to {k.dtype=} and {v.dtype=}."
        )
    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
    if kv_packing != get_dtype_packing(kv_cache.dtype):
        raise ValueError(f"{kv_packing=} does not match with {kv_cache.dtype=}")

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == distribution.dtype
    ):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}"
        )

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}"
        )

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_block is not None and num_kv_pages_per_block <= 0:
        raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_block is not None and num_queries_per_block <= 0:
        raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del debug_mode


def get_kernel_scope_name(bq_size, bkv_p, page_size):
    return f"RPA-prefill-bq_{bq_size}-bkvp_{bkv_p}-p_{page_size}"


def _ragged_paged_attention_impl(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # SEQ_ALONG_LANE: [total_num_pages, 2*kv_heads, hd_aligned, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sink: jax.Array | None = None,  # f32[actual_num_q_heads]
    visibility: jax.Array | None = None,  # i32[max_num_tokens, 2]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    bkv_p,
    bq_sz,
    chunk_prefill_size: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
    # KV-share: when False, skip the new-K/V writeback into kv_cache and read
    # the entire kv_left from cache (the source layer is assumed to have
    # populated it).
    update_kv_cache: bool = True,
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE,
    prepacked_q_hbm: jax.Array | None = None,
    prepacked_new_kv_hbm: jax.Array | None = None,
):
    q, k, v = queries, keys, values
    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            "page_indices length must be divisible by max_num_seqs, got "
            f"{num_page_indices=} and {max_num_seqs=}"
        )
    pages_per_seq = num_page_indices // max_num_seqs
    # TPU BlockSpec dimensions must be a power of two >= 128 or a multiple of
    # 1024. Use the smallest legal power of two to limit per-program SMEM use.
    page_indices_block_size = next_power_of_2(max(pages_per_seq, 128))
    page_indices_by_seq = page_indices.reshape(max_num_seqs, pages_per_seq)
    # Add a sentinel row so the final program's next-row BlockSpec remains in
    # bounds, just like every preceding program's current/next pair.
    page_indices_by_seq = jnp.pad(
        page_indices_by_seq,
        ((0, 1), (0, page_indices_block_size - pages_per_seq)),
    )
    page_indices_for_kernel = page_indices_by_seq.reshape(-1)
    seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE
    # SEQ_ALONG_LANE writes the new K/V into the cache in-place via DMA inside
    # the pallas kernel (see _fetch_bkv / stitch / writeback), so there is no
    # out-of-kernel scatter and update_kv_cache stays True.
    is_hd64 = False

    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    if (prepacked_q_hbm is None) != (prepacked_new_kv_hbm is None):
        raise ValueError("prepacked Q and K/V must be provided together")
    if prepacked_q_hbm is None:
        q, kv, attention_sink = prepare_inputs(
            q,
            k,
            v,
            attention_sink,
            is_hd64=is_hd64,
            kv_layout=kv_layout,
        )
    else:
        if attention_sink is not None:
            raise ValueError("prepacked RPA-seq does not support attention_sink")
        q = prepacked_q_hbm
        kv = prepacked_new_kv_hbm
    (
        _,
        _max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache.shape[-1] if seq_along_lane else kv_cache.shape[1]
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    if vmem_limit_bytes is None:
        # TODO (jevinjiang/jacobplatin): change this to use
        # `get_vmem_estimate_bytes` when VREG spilling is fixed.
        vmem_limit_bytes = DEFAULT_VMEM_LIMIT_BYTES

    def _launch_kernel():
        # Single region: prefill + mixed, i.e. every non-decode sequence.
        grid = (distribution[2] - distribution[0],)
        bkv_sz = bkv_p * page_size

        def _page_indices_index_map(
            program_id,
            _kv_lens_ref,
            _cu_q_lens_ref,
            distribution_ref,
            *_unused_scalar_refs,
        ):
            # Grid covers distribution[0]..distribution[2]; skip the decode region.
            seq_idx_offset = distribution_ref[0]
            return (program_id + seq_idx_offset,)

        def _next_page_indices_index_map(
            program_id,
            _kv_lens_ref,
            _cu_q_lens_ref,
            distribution_ref,
            *_unused_scalar_refs,
        ):
            # Grid covers distribution[0]..distribution[2]; skip the decode region.
            seq_idx_offset = distribution_ref[0]
            return (program_id + seq_idx_offset + 1,)

        in_specs = [
            pl.BlockSpec(
                (page_indices_block_size,),
                _page_indices_index_map,
                memory_space=pltpu.SMEM,
            ),
            pl.BlockSpec(
                (page_indices_block_size,),
                _next_page_indices_index_map,
                memory_space=pltpu.SMEM,
            ),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        if attention_sink is not None:
            in_specs.append(pl.BlockSpec(memory_space=pltpu.VMEM))

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        # [2, 2*kv_heads, hd_aligned, bkv_sz + slack*page_size]
        # The two slack pages cover the worst-case cached/new split where both
        # sides straddle a page boundary. Must match stage_slack_pages in
        # stitch_seq.
        stage_slack_pages = 2
        bkv_double_buf = pltpu.VMEM(
            (2, *kv_cache.shape[1:3], bkv_sz + stage_slack_pages * page_size),
            kv_cache.dtype,
        )

        bq_double_buf = pltpu.VMEM(
            (2, actual_num_kv_heads, bq_sz, *q.shape[2:]),
            q.dtype,
        )

        bo_double_buf = bq_double_buf

        l_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
            jnp.float32,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
            jnp.float32,
        )

        scratch_shapes = [
            bkv_double_buf,  # Double buffering for kv block.
            bq_double_buf,  # Double buffering for q block.
            bo_double_buf,  # Double buffering for output block.
            # Semaphores for double buffering of bkv, bq, bo and bkv_update.
            pltpu.SemaphoreType.DMA((4, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]
        if visibility is not None:
            scratch_shapes[3] = pltpu.SemaphoreType.DMA((5, 2))
            scratch_shapes.append(
                pltpu.VMEM((2, bq_sz, 128), jnp.int32),
            )

        scalar_prefetches = (
            kv_lens,
            cu_q_lens,
            distribution,
            # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
            jnp.zeros((3,), jnp.int32),
            # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
            jnp.full((4,), -1, jnp.int32),
            # (bkv_sem_*_seq_idx, *_offset/bkv_sz_cache, *_sz/bkv_sz_frm_new,
            #  *_bkv_idx) -- last 2 slots used by the seq deferred writeback.
            jnp.full((8,), -1, jnp.int32),
            _normalize_kv_head_scale(k_scale, actual_num_kv_heads),
            _normalize_kv_head_scale(v_scale, actual_num_kv_heads),
        )

        scope_name = get_kernel_scope_name(bq_sz, bkv_p, page_size)
        n_scalar = len(scalar_prefetches)
        base_kernel_fn = functools.partial(
            _ragged_paged_attention_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            mask_value=mask_value,
            q_scale=q_scale,
            has_k_scale=k_scale is not None,
            has_v_scale=v_scale is not None,
            pages_per_seq=pages_per_seq,
            bkv_p=bkv_p,
            bq_sz=bq_sz,
            update_kv_cache=update_kv_cache,
        )

        if visibility is not None:
            # Pad max_num_tokens by bq_sz to prevent any OOB DMA
            # Pad 2 to 2 + 126 to satisfy DMA requirement of 128 in the minor dimension
            visibility_hbm = jnp.pad(visibility, ((0, bq_sz), (0, 126)))
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))

        def pallas_kernel_fn(*args, _base=base_kernel_fn, _n_scalar=n_scalar):
            idx = _n_scalar
            page_indices_ref = args[idx]
            next_page_indices_ref = args[idx + 1]
            idx += 2
            base_inputs = args[idx : idx + 3]
            idx += 3
            scalar_args = args[:_n_scalar]
            return _base(
                scalar_args[0],
                page_indices_ref,
                next_page_indices_ref,
                *scalar_args[1:],
                *base_inputs,
                *args[idx:],
            )

        kernel = pl.pallas_call(
            pallas_kernel_fn,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=n_scalar,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                # TODO(jevinjiang): since each sequence depends on the previous
                # one, we need some extra work to support Megacore mode.
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                # Paged attention invokes multiple small DMAs for each pages instead
                # of a single large DMA. Therefore, the overhead of bounds checking
                # becomes too significant so we disable it.
                disable_bounds_checks=True,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                jax.ShapeDtypeStruct(shape=kv_cache.shape, dtype=kv_cache.dtype),
            ],
            input_output_aliases={n_scalar + 2: 0, n_scalar + 4: 1},
            name=scope_name,
        )

        extra_inputs = ()
        if attention_sink is not None:
            extra_inputs = extra_inputs + (attention_sink,)
        if visibility is not None:
            extra_inputs = extra_inputs + (visibility_hbm,)
        return kernel(
            *scalar_prefetches,
            page_indices_for_kernel,
            page_indices_for_kernel,
            q,
            kv,
            kv_cache,
            *extra_inputs,
        )

    # output shares its buffer with q, updated_kv_cache with kv_cache
    output, updated_kv_cache = _launch_kernel()

    if prepacked_q_hbm is not None:
        return output, updated_kv_cache
    return (
        prepare_outputs(
            output,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
            is_hd64=is_hd64,
        ),
        updated_kv_cache,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "use_causal_mask",
        "skip_kv_mask",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "out_dtype",
        "mask_value",
        "q_scale",
        "chunk_prefill_size",
        "decode_q_len",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "debug_mode",
        "update_kv_cache",
        "kv_layout",
    ),
    donate_argnames=("kv_cache",),
    compiler_options={
        # uncomment the following line for named_scopes to show up in traces
        # this incurs some performance loss in benchmarks, so only enable
        # when needed
        # "xla_enable_transpose_trace": True,
        # uncomment the following line to allow debug_print to show up in
        # logs, note that this changes the behavior of the kernel completely,
        # must turn off for accuracy tests
        # "xla_tpu_enable_log_recorder": True,
    },
)
def ragged_paged_attention_prefill(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # SEQ_ALONG_LANE: [total_num_pages, 2*kv_heads, hd_aligned, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sink: jax.Array | None = None,  # f32[actual_num_q_heads]
    visibility: jax.Array | None = None,  # i32[max_num_tokens, 2]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Decode region static q-block length = num_speculative_tokens + 1.
    # Default 1 reproduces the single-token decode path exactly.
    decode_q_len: int = 1,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    # tpu-inference compatible tuning params (accepted but mapped internally).
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
    # KV-share: when False, the kernel will SKIP the new-K/V writeback into
    # kv_cache and read the entire kv_left from cache (assuming a source layer
    # already populated the slot). Mirrors upstream tpu-inference v3 behavior.
    # Useful for KV-share patterns and speculative-decoding rollback.
    update_kv_cache: bool = True,
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE,
    prepacked_q_hbm: jax.Array | None = None,
    prepacked_new_kv_hbm: jax.Array | None = None,
):
    """Run causal SEQ_ALONG_LANE prefill and cached-prefix attention.

    This specialization processes only the non-decode region and supports causal
    full-context or sliding-window attention. Attention sinks, visibility ranges,
    and logit soft caps remain unsupported.

    Args:
      queries: concatenated all sequences' queries.
      keys: concatenated all sequences' keys (quantized).
      values: concatenated all sequences' values (quantized).
      kv_cache: paged KV cache with TPU-friendly shape.
      kv_lens: padded kv lengths. Only the first num_seqs values are valid.
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      attention_sink: optional attention sink for each q head.
      visibility: optional per-token visibility ranges. Shape [max_num_tokens, 2]
        of i32, where visibility[t] = (start, end) means token t can attend to KV
        positions in [start, end]. When provided, replaces causal and sliding window
        masking. Padding positions should be zero-filled.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      sliding_window: the sliding window size for the attention.
      soft_cap: the logit soft cap for the attention.
      mask_value: mask value for causal mask.
      q_scale: the scale for the query.
      k_scale: the scale for the key cache.
      v_scale: the scale for the value cache.
      chunk_prefill_size: the chunk prefill size for the attention.
      decode_q_len: static q-block length for the decode region
        (num_speculative_tokens + 1). Default 1 reproduces the single-token decode
        path. Must be <= the decode num_queries_per_block so num_bq == 1.
      num_kv_pages_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      num_queries_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      vmem_limit_bytes: the vmem limit for the pallas kernel.
      debug_mode: if true, RPA does not issue any DMAs or run flash attention but
        print debug info. Need to compile with `--xla_tpu_enable_log_recorder`.
      prepacked_q_hbm: optional Q already in the stacked/RPA-seq packed layout.
        Must be supplied together with ``prepacked_new_kv_hbm``. This internal
        path returns packed output so a caller can preserve its existing
        decode/prefill dispatch state without another layout conversion.
      prepacked_new_kv_hbm: optional K/V already in SEQ_ALONG_LANE layout,
        lane-padded through ``max_num_tokens``.

    Returns:
      The output of the attention. It remains packed when prepacked inputs are
      supplied; otherwise it has the same shape as ``queries``.
    """
    if kv_layout != KVLayout.SEQ_ALONG_LANE:
        raise ValueError("rpa-seq prefill requires KVLayout.SEQ_ALONG_LANE")
    if not use_causal_mask:
        raise ValueError("rpa-seq prefill supports only causal attention")
    if skip_kv_mask:
        raise ValueError("skip_kv_mask is not supported by rpa-seq prefill")
    if attention_sink is not None:
        raise ValueError("attention_sink is not supported by rpa-seq prefill")
    if visibility is not None:
        raise ValueError("visibility is not supported by rpa-seq prefill")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive")
    if soft_cap is not None:
        raise ValueError("soft_cap is not supported by rpa-seq prefill")
    if chunk_prefill_size is not None:
        raise ValueError("chunk_prefill_size is not supported by rpa-seq prefill")
    if debug_mode:
        raise ValueError("debug_mode is not supported by rpa-seq prefill")
    if out_dtype is not None and jnp.dtype(out_dtype) != queries.dtype:
        raise ValueError(
            "rpa-seq prefill supports only query-dtype output, got "
            f"{jnp.dtype(out_dtype)} for {queries.dtype} queries"
        )
    if (prepacked_q_hbm is None) != (prepacked_new_kv_hbm is None):
        raise ValueError("prepacked Q and K/V must be provided together")

    actual_num_q_heads = queries.shape[1]
    actual_head_dim = queries.shape[2]
    actual_num_kv_heads = keys.shape[1]
    max_num_tokens = queries.shape[0]
    seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE
    page_size = kv_cache.shape[-1] if seq_along_lane else kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    use_prepacked_inputs = prepacked_q_hbm is not None
    if use_prepacked_inputs:
        q_packing = get_dtype_packing(queries.dtype)
        num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
        aligned_q_heads = align_to(num_q_heads_per_kv_head, q_packing)
        expected_q_shape = (
            actual_num_kv_heads,
            max_num_tokens,
            aligned_q_heads // q_packing,
            q_packing,
            align_to(actual_head_dim, 128),
        )
        if prepacked_q_hbm.shape != expected_q_shape:
            raise ValueError(
                f"Expected {prepacked_q_hbm.shape=} to equal {expected_q_shape=}"
            )
        kv_packing = get_dtype_packing(kv_cache.dtype)
        expected_new_kv_shape = (
            2 * actual_num_kv_heads,
            align_to(
                actual_head_dim,
                pltpu.get_tpu_info().num_sublanes * kv_packing,
            ),
            align_to(max_num_tokens, pltpu.get_tpu_info().num_lanes),
        )
        if prepacked_new_kv_hbm.shape != expected_new_kv_shape:
            raise ValueError(
                f"Expected {prepacked_new_kv_hbm.shape=} to equal "
                f"{expected_new_kv_shape=}"
            )
        if prepacked_q_hbm.dtype != queries.dtype:
            raise ValueError(
                f"Expected {prepacked_q_hbm.dtype=} to equal {queries.dtype=}"
            )
        if prepacked_new_kv_hbm.dtype != kv_cache.dtype:
            raise ValueError(
                f"Expected {prepacked_new_kv_hbm.dtype=} to equal {kv_cache.dtype=}"
            )

    # --- Per-channel fp8 KV scale ([num_kv_heads, head_dim]) -----------------
    # A per-(head, channel) dequant scale folds OUT of the flash-attention
    # kernel, so no Mosaic hot-loop change is needed: the K scale commutes onto
    # Q  (q . (k_q (x) s_k) == (q (x) s_k) . k_q)  and the V scale factors out
    # of the PV sum + softmax normalize  (o == s_v (x) ((P @ v_q) / l)).  We
    # pre-scale Q here, post-scale the output below, and hand the inner kernel
    # k_scale=v_scale=None.  Scalar / per-kv-head ([num_kv_heads]) scales keep
    # the fast in-kernel scalar fold (this branch is skipped for them).
    _pc_v_scale = None
    _pc_k_is_2d = k_scale is not None and getattr(k_scale, "ndim", 0) == 2
    _pc_v_is_2d = v_scale is not None and getattr(v_scale, "ndim", 0) == 2
    if _pc_k_is_2d or _pc_v_is_2d:
        assert q_scale is None, "per-channel KV scale + q_scale unsupported"
        _nq_per_kv = actual_num_q_heads // actual_num_kv_heads
        if _pc_k_is_2d:
            if use_prepacked_inputs:
                _kq = jnp.pad(
                    jnp.asarray(k_scale, jnp.float32),
                    ((0, 0), (0, prepacked_q_hbm.shape[-1] - actual_head_dim)),
                    constant_values=1.0,
                )
                prepacked_q_hbm = (
                    prepacked_q_hbm.astype(jnp.float32) * _kq[:, None, None, None, :]
                ).astype(prepacked_q_hbm.dtype)
            else:
                _kq = jnp.repeat(jnp.asarray(k_scale, jnp.float32), _nq_per_kv, axis=0)
                queries = (queries.astype(jnp.float32) * _kq[None]).astype(
                    queries.dtype
                )
            k_scale = None
        if _pc_v_is_2d:
            if use_prepacked_inputs:
                _pc_v_scale = jnp.asarray(v_scale, jnp.float32)
            else:
                _pc_v_scale = jnp.repeat(
                    jnp.asarray(v_scale, jnp.float32), _nq_per_kv, axis=0
                )
            v_scale = None

    # Validate decode_q_len here at the entrypoint so it covers BOTH KV layouts
    # (HEAD_ALONG_SUBLANE and SEQ_ALONG_LANE): static_validate_inputs only runs on
    # the non-seq_along_lane branch in _ragged_paged_attention_impl, so moving these
    # checks there would skip the SEQ_ALONG_LANE path.
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")
    if num_queries_per_block is not None and decode_q_len > num_queries_per_block:
        raise ValueError(
            f"decode_q_len={decode_q_len} must be <= num_queries_per_block="
            f"{num_queries_per_block} so the decode region keeps num_bq == 1"
        )

    bkv_p, bq_sz = resolve_block_sizes(
        queries.dtype,
        kv_cache.dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        actual_head_dim,
        page_size,
        max_num_tokens,
        pages_per_seq,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )

    _pc_out = _ragged_paged_attention_impl(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink,
        visibility,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        bkv_p=bkv_p,
        bq_sz=bq_sz,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        update_kv_cache=update_kv_cache,
        kv_layout=kv_layout,
        prepacked_q_hbm=prepacked_q_hbm,
        prepacked_new_kv_hbm=prepacked_new_kv_hbm,
    )

    if _pc_v_scale is not None:
        # Per-channel V dequant: out (x)= v_scale (factors out of PV+normalize).
        _o, _upd = _pc_out
        if use_prepacked_inputs:
            _pc_v_scale = jnp.pad(
                _pc_v_scale,
                ((0, 0), (0, _o.shape[-1] - actual_head_dim)),
                constant_values=1.0,
            )
            _o = (_o.astype(jnp.float32) * _pc_v_scale[:, None, None, None, :]).astype(
                _o.dtype
            )
        else:
            _o = (_o.astype(jnp.float32) * _pc_v_scale[None]).astype(_o.dtype)
        _pc_out = (_o, _upd)
    return _pc_out
