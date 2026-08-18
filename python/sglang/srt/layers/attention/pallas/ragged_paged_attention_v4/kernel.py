"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.

The kernel supports two KV cache layouts:
- Standard layout: K and V heads are interleaved. Works with any head_dim.
- Packed layout (head_dim=64): K and V are concatenated along the head dimension
  (actual_head_dim_x2=128), halving the number of cache heads. This is more
  efficient for head_dim=64 models.

The layout is auto-detected from the kv_cache shape.
"""

import enum
import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tuned_block_sizes import (
    get_tuned_block_sizes,
)
from .util import (
    align_to,
    cdiv,
    get_dtype_bitwidth,
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


def _ref_per_head_scale(scale, num_q_heads_per_kv_head, ndim, head_axis, dtype):
    """Broadcast scalar or per-kv-head [num_kv_heads] scale to the attn/out head axis."""
    arr = jnp.asarray(scale, dtype)
    if arr.ndim == 0:
        return arr
    arr = jnp.repeat(arr, num_q_heads_per_kv_head)
    shape = [1] * ndim
    shape[head_axis] = arr.shape[0]
    return arr.reshape(tuple(shape))


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


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,
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
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]

    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        return _ref_ragged_paged_attention_seq_along_lane(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    is_hd64 = detect_is_hd64(actual_num_kv_heads, actual_head_dim, kv_cache)

    dynamic_validate_inputs(
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
    )

    _, page_size, _, kv_packing, cache_head_dim = kv_cache.shape
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    if is_hd64:
        assert actual_head_dim == 64
        assert cache_head_dim == 128
        padded_num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
        merged_kv = merge_kv(keys, values, is_hd64=True)
        queries = jnp.pad(queries, ((0, 0), (0, 0), (0, 64)), constant_values=0.0)
    else:
        head_dim = align_to(actual_head_dim, 128)
        num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
        merged_kv = merge_kv(keys, values, is_hd64=False)
        assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    outputs = []

    for i in range(distribution[-1]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        indices_start = i * pages_per_seq
        indices_end = indices_start + cdiv(kv_len, page_size)
        indices = page_indices[indices_start:indices_end]
        q = queries[q_start:q_end, :, :]

        # Update the kv cache.
        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        gathered_kv = gathered_kv.at[kv_len - q_len : kv_len].set(
            merged_kv[q_start:q_end]
        )
        kv_cache = kv_cache.at[indices].set(gathered_kv.reshape(gathered_shape))

        if is_hd64:
            kv = gathered_kv.reshape(-1, padded_num_kv_heads, 128)[
                :, :actual_num_kv_heads, :
            ]
            kv = kv[:kv_len, :, :]
            kv = jnp.repeat(kv, actual_num_q_heads_per_kv_head, axis=1)

            if q_scale is not None:
                q = q / q_scale
                if jnp.issubdtype(kv.dtype, jnp.floating):
                    dtype_info = jnp.finfo(kv.dtype)
                    q = jnp.clip(
                        q, min=float(dtype_info.min), max=float(dtype_info.max)
                    )
                q = q.astype(kv.dtype)

            attn = jnp.einsum("qhd,khd->hqk", q, kv, preferred_element_type=jnp.float32)
        else:
            kv = gathered_kv.reshape(-1, num_kv_heads_x2, head_dim)[
                :, : actual_num_kv_heads * 2, :
            ].reshape(-1, actual_num_kv_heads, head_dim * 2)
            k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
            v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
            k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
            v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

            if q_scale is not None:
                q = q / q_scale
                if jnp.issubdtype(k.dtype, jnp.floating):
                    dtype_info = jnp.finfo(k.dtype)
                    q = jnp.clip(
                        q, min=float(dtype_info.min), max=float(dtype_info.max)
                    )
                q = q.astype(k.dtype)

            attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)

        attn *= sm_scale
        if k_scale is not None:
            attn *= _ref_per_head_scale(
                k_scale, actual_num_q_heads_per_kv_head, attn.ndim, 0, attn.dtype
            )
        if q_scale is not None:
            attn *= q_scale

        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        if visibility is not None:
            vis = visibility[q_start:q_end]
            vis_start = vis[:, 0].reshape(1, q_len, 1)
            vis_end = vis[:, 1].reshape(1, q_len, 1)
            mask = jnp.logical_or(kv_span < vis_start, kv_span > vis_end)
        else:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1
            )
            mask = q_span < kv_span
            if sliding_window is not None:
                mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn = jnp.where(mask, mask_value, attn)

        if attention_sink is not None:
            reshaped_attention_sink = attention_sink.reshape(actual_num_q_heads, 1, 1)
            reshaped_attention_sink = jnp.repeat(reshaped_attention_sink, q_len, axis=1)
            attn = jnp.concat([reshaped_attention_sink, attn], axis=2)
            # Keep softmax in f32: casting probs to the KV dtype underflows to 0 for
            # fp8 at large context (uniform prob 1/kv_len < fp8 min subnormal).
            attn = jax.nn.softmax(attn, axis=-1).astype(jnp.float32)
            attn = attn[..., 1:]
        else:
            attn = jax.nn.softmax(attn, axis=-1).astype(jnp.float32)

        if is_hd64:
            out = jnp.einsum("hqk,khd->qhd", attn, kv.astype(jnp.float32)).astype(
                queries.dtype
            )
        else:
            out = jnp.einsum("hqk,khd->qhd", attn, v.astype(jnp.float32)).astype(
                queries.dtype
            )
        if v_scale is not None:
            out *= _ref_per_head_scale(
                v_scale, actual_num_q_heads_per_kv_head, out.ndim, 1, out.dtype
            )

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    if is_hd64:
        result = result[:, :, actual_head_dim:]
    return result, kv_cache


def get_smem_estimate_bytes(max_num_seqs, pages_per_seq):
    total_bits = (
        # kv_lens_ref: i32[max_num_seqs]
        align_to(max_num_seqs, 128) * 32
        +
        # page_indices_ref: padded current + next i32[pages_per_seq] rows
        2 * next_power_of_2(max(pages_per_seq, 128)) * 32
        +
        # cu_q_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32
        +
        # distribution_ref: i32[3]
        128 * 32
        +
        # sem_ids_ref: i32[3]
        128 * 32
        +
        # bo_ids_ref: i32[4]
        128 * 32
        +
        # bkv_update_ids_ref: i32[6]
        128 * 32
    )
    return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
    is_hd64=False,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    if is_hd64:
        num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
        head_dim = actual_head_dim * 2
    else:
        num_kv_heads = align_to(actual_num_kv_heads * 2, kv_packing)
        head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * num_kv_heads * head_dim) * (32 // kv_packing)
        +
        # bq_x2_ref + bo_x2_ref
        2
        * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim)
        * (32 // q_packing)
        +
        # l_ref + m_ref
        2 * (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32
        +
        # acc_ref
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * 32
    )
    return cdiv(total_bits, 8)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
    *,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
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
            align_to(actual_head_dim, num_sublanes * kv_packing) // kv_packing,
            kv_packing,
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
# Cache shape: [pages, 2*kv_heads, hd_aligned // kv_packing, kv_packing, page_size]
# where the 2*kv_heads axis is interleaved [k0, v0, k1, v1, ...] and the head_dim
# axis is laid out as (hd_aligned // kv_packing, kv_packing) so a reshape recovers
# the contiguous head_dim. page_size occupies the lane (minor) axis.
# ---------------------------------------------------------------------------
def _seq_along_lane_cache_dims(kv_cache):
    total_num_pages, two_kv_heads, hdp, kv_packing, page_size = kv_cache.shape
    assert two_kv_heads % 2 == 0, two_kv_heads
    return total_num_pages, two_kv_heads // 2, hdp, kv_packing, page_size


def _seq_along_lane_merge_new_kv(keys, values, hd_aligned, kv_packing):
    """Pack new K/V tokens into the cache's per-token sublane layout.

    Returns [max_num_tokens, 2*kv_heads, hd_aligned // kv_packing, kv_packing],
    matching a single page-lane slice of the SEQ_ALONG_LANE cache.
    """
    max_num_tokens, kv_heads, actual_head_dim = keys.shape
    merged = jnp.concatenate([keys, values], axis=-1).reshape(
        max_num_tokens, 2 * kv_heads, actual_head_dim
    )
    merged = jnp.pad(
        merged, ((0, 0), (0, 0), (0, hd_aligned - actual_head_dim)), constant_values=0
    )
    return merged.reshape(
        max_num_tokens, 2 * kv_heads, hd_aligned // kv_packing, kv_packing
    )


def unpack_seq_along_lane_kv_cache(kv_cache, actual_head_dim):
    """Convert a SEQ_ALONG_LANE cache to logical token-major K and V.

    Input:  5D [pages, 2*kv_heads, hd_aligned // pk, pk, page_size] or the
            collapsed 4D [pages, 2*kv_heads, hd_aligned, page_size] ([64] layout).
    Returns (k, v), each [pages, page_size, kv_heads, actual_head_dim].
    """
    if kv_cache.ndim == 4:
        pages, two_kv_heads, hd_aligned, page_size = kv_cache.shape
        kv_heads = two_kv_heads // 2
        merged = kv_cache
    else:
        pages, kv_heads, hdp, pk, page_size = _seq_along_lane_cache_dims(kv_cache)
        hd_aligned = hdp * pk
        merged = kv_cache.reshape(pages, 2 * kv_heads, hd_aligned, page_size)
    # [pages, 2*kv_heads, hd_aligned, page_size] -> [pages, page_size, 2*kv_heads, hd]
    merged = merged.transpose(0, 3, 1, 2)
    merged = merged.reshape(pages, page_size, kv_heads, 2, hd_aligned)
    k = merged[:, :, :, 0, :actual_head_dim]
    v = merged[:, :, :, 1, :actual_head_dim]
    return k, v


def append_seq_along_lane_kv_cache(
    keys: jax.Array,  # [max_num_tokens, kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # SEQ_ALONG_LANE layout
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
):
    """Scatter new K/V tokens into a SEQ_ALONG_LANE cache (done outside the kernel).

    Each new token writes one page-lane column. Token t belongs to sequence s and
    lands at logical position kv_lens[s] - q_len[s] + local_offset.
    """
    is_4d = kv_cache.ndim == 4
    if is_4d:
        pages, two_kv_heads, hd_aligned, page_size = kv_cache.shape
        kv_heads = two_kv_heads // 2
        max_num_tokens, _, actual_head_dim = keys.shape
        merged = jnp.concatenate([keys, values], axis=-1).reshape(
            max_num_tokens, 2 * kv_heads, actual_head_dim
        )
        merged = jnp.pad(
            merged,
            ((0, 0), (0, 0), (0, hd_aligned - actual_head_dim)),
            constant_values=0,
        )
    else:
        pages, kv_heads, hdp, pk, page_size = _seq_along_lane_cache_dims(kv_cache)
        hd_aligned = hdp * pk
        merged = _seq_along_lane_merge_new_kv(keys, values, hd_aligned, pk)
    max_num_tokens = keys.shape[0]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    token_indices = jnp.arange(max_num_tokens, dtype=jnp.int32)
    num_seqs = distribution[-1]
    # Map each token to its sequence via searchsorted on cu_q_lens.
    active_cu_q_lens = jnp.where(
        jnp.arange(max_num_seqs + 1, dtype=jnp.int32) <= num_seqs,
        cu_q_lens,
        max_num_tokens + 1,
    )
    seq_idx = jnp.searchsorted(active_cu_q_lens, token_indices, side="right") - 1
    seq_idx = jnp.clip(seq_idx, 0, max_num_seqs - 1)
    active = token_indices < cu_q_lens[num_seqs]

    q_start = cu_q_lens[seq_idx]
    q_len = cu_q_lens[seq_idx + 1] - q_start
    local = token_indices - q_start
    pos = kv_lens[seq_idx] - q_len + local
    pos = jnp.where(active, pos, 0)

    page_off = pos // page_size
    lane = pos % page_size
    lut = jnp.where(active, seq_idx * pages_per_seq + page_off, 0)
    target_page = jnp.where(active, page_indices[lut], pages)  # OOB page -> dropped

    if is_4d:
        return kv_cache.at[target_page, :, :, lane].set(merged, mode="drop")
    return kv_cache.at[target_page, :, :, :, lane].set(merged, mode="drop")


def _ref_ragged_paged_attention_seq_along_lane(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    sm_scale,
    sliding_window,
    soft_cap,
    mask_value,
    q_scale,
    k_scale,
    v_scale,
):
    """Reference attention for the SEQ_ALONG_LANE layout."""
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    kv_cache = append_seq_along_lane_kv_cache(
        keys, values, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
    )
    k_cache, v_cache = unpack_seq_along_lane_kv_cache(kv_cache, actual_head_dim)
    # Compute the reference in fp32; upcast so fp8 (and low-precision) caches are
    # dequantized rather than fed straight into the matmul (which yields zeros).
    k_cache = k_cache.astype(jnp.float32)
    v_cache = v_cache.astype(jnp.float32)
    page_size = k_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    outputs = []
    for i in range(distribution[-1]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[
            i * pages_per_seq : i * pages_per_seq + cdiv(kv_len, page_size)
        ]
        q = queries[q_start:q_end, :, :]

        k = k_cache[indices].reshape(-1, actual_num_kv_heads, actual_head_dim)[:kv_len]
        v = v_cache[indices].reshape(-1, actual_num_kv_heads, actual_head_dim)[:kv_len]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                q = jnp.clip(q, min=float(dtype_info.min), max=float(dtype_info.max))
            q = q.astype(k.dtype)

        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn *= sm_scale
        if k_scale is not None:
            attn *= _ref_per_head_scale(
                k_scale, actual_num_q_heads_per_kv_head, attn.ndim, 0, attn.dtype
            )
        if q_scale is not None:
            attn *= q_scale

        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn = jnp.where(mask, mask_value, attn)
        # Keep softmax in f32: casting probs to v.dtype underflows to 0 for fp8 at
        # large context (uniform prob 1/kv_len < fp8 min subnormal ~2^-9).
        attn = jax.nn.softmax(attn, axis=-1).astype(jnp.float32)
        out = jnp.einsum("hqk,khd->qhd", attn, v.astype(jnp.float32)).astype(
            queries.dtype
        )
        if v_scale is not None:
            out *= _ref_per_head_scale(
                v_scale, actual_num_q_heads_per_kv_head, out.ndim, 1, out.dtype
            )
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0), kv_cache


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
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, head_dim]
    attention_sink_ref,  # [actual_num_kv_heads, num_q_heads_per_kv_head, 128] or None
    visibility_hbm_ref,  # [max_num_tokens, 128] or None
    # Output
    o_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    updated_kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, head_dim]
    # Scratch
    bkv_x2_ref,  # [2, bkv_sz, num_kv_heads // kv_packing, kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    sems,  # [4, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim],
    vis_scratch_ref,  # [2, bq_sz, 128] or None
    *,
    is_hd64: bool = False,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    has_k_scale: bool = False,
    has_v_scale: bool = False,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
    rpa_kind: int,  # 0: decode, 1: prefill, 2: mixed
    decode_q_len: int = 1,  # decode-region static q-block length (num_spec + 1)
    pages_per_seq: int,
    debug_mode: bool = False,
    update_kv_cache: bool = True,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    if seq_along_lane:
        # SEQ_ALONG_LANE cache: [pages, 2*kv_heads, hd_aligned//pk, pk, page_size].
        (
            total_num_pages,
            _two_kv_heads_cache,
            _hdp_cache,
            kv_packing,
            page_size,
        ) = kv_cache_hbm_ref.shape
        cache_head_dim = _hdp_cache * kv_packing
        num_kv_heads_per_kv_packing = _two_kv_heads_cache
    else:
        assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]
        (
            total_num_pages,
            page_size,
            num_kv_heads_per_kv_packing,
            kv_packing,
            _,
        ) = kv_cache_hbm_ref.shape
        cache_head_dim = head_dim
    # Each program stages its current and next sequence's page-table rows. The
    # next row is required because KV prefetch crosses sequence boundaries.
    num_kv_heads = num_kv_heads_per_kv_packing * kv_packing
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    if is_hd64:
        assert head_dim == 128
    else:
        assert head_dim % 128 == 0
    bkv_sz = bkv_p * page_size
    program_seq_idx = -1

    def physical_page_index(target_seq_idx, logical_page_idx):
        page_idx = jnp.minimum(logical_page_idx, pages_per_seq - 1)
        return jnp.where(
            target_seq_idx == program_seq_idx,
            page_indices_ref[page_idx],
            next_page_indices_ref[page_idx],
        )

    decode_end = distribution_ref[0]
    prefill_end = distribution_ref[1]
    mixed_end = distribution_ref[2]

    # --- Window-anchored SW decode (prototype, env-gated) ---
    # PR #962 sizes the decode KV block to next_pow2(2*window) (32 pages / 4096
    # tok for a 2048 window) because blocks are anchored to the GLOBAL page grid,
    # so the window straddles a block boundary. Instead, anchor the single block
    # at the page-aligned window start and read ~one window (bkv_p set to
    # cdiv(window+q+page-1, page) in the entry) in ONE iteration. ON by default;
    # set TPU_RPA_SW_BOUND=0 to disable. Only the seq_along_lane decode path.
    WINDOW_ANCHORED = (
        rpa_kind == 0
        and sliding_window is not None
        and seq_along_lane
        and os.environ.get("TPU_RPA_SW_BOUND", "1") == "1"
    )

    def _window_anchor_tok(kv_len, q_len):
        # Page-aligned start of the sliding window (in tokens).
        return (
            jnp.maximum(kv_len - q_len - sliding_window, 0) // page_size
        ) * page_size

    # --- Per-q-block sliding-window lower bound for PREFILL/MIXED (prototype) ---
    # bkv_idx_start is computed once per sequence from the EARLIEST query
    # (bq_idx=0's window), so every q-block iterates the bkv blocks below its own
    # window -- correctly masked but wasted compute (scales with num_bq). When
    # enabled, each q-block starts its bkv loop at its OWN window lower bound
    # (_sw_bq_start), skipping the fully-masked low blocks. ON by default; set
    # TPU_RPA_SW_BOUND=0 to disable. Prefill/mixed only (rpa_kind != 0), no vis.
    PREFILL_SW_BOUND = (
        rpa_kind != 0
        and sliding_window is not None
        and visibility_hbm_ref is None
        and os.environ.get("TPU_RPA_SW_BOUND", "1") == "1"
    )

    def get_seq_info(seq_idx, seq_idx_offset):
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens_ref[seq_idx]

        if visibility_hbm_ref is not None or sliding_window is None or WINDOW_ANCHORED:
            bkv_idx_start = next_seq_bkv_idx_start = 0
        else:
            bkv_idx_start = jnp.maximum(kv_len - q_len - sliding_window, 0) // bkv_sz

            # If seq_idx + 1 == num_seqs, kv_lens_ref[seq_idx + 1] will trigger a
            # out-of-bound error. To avoid this, we set upperbound of next_seq_idx
            # to be num_seqs - 1.
            next_seq_idx = jnp.minimum(seq_idx + 1, num_seqs + seq_idx_offset - 1)
            next_kv_len = kv_lens_ref[next_seq_idx]
            next_q_len = cu_q_lens_ref[next_seq_idx + 1] - q_end
            next_seq_bkv_idx_start = (
                jnp.maximum(next_kv_len - next_q_len - sliding_window, 0) // bkv_sz
            )

        return q_start, q_end, q_len, kv_len, bkv_idx_start, next_seq_bkv_idx_start

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    @jax.named_scope("flash_attention_step1_qk_softmax")
    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_sz * num_q_heads_per_kv_head, head_dim]
        kv_or_k,  # [bkv_sz, head_dim] - combined kv (hd64) or k (standard)
        v=None,  # [bkv_sz, head_dim] - only for standard layout
        *,
        bq_idx,
        bkv_idx,
        kv_head_idx,
        visibility=None,  # [2, actual_bq_sz] array or None
        skip_mask=False,  # True for KV blocks fully below the causal diagonal
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == head_dim
        if is_hd64:
            k_for_dot = kv_or_k
            assert kv_or_k.shape == (bkv_sz, head_dim)
        elif seq_along_lane:
            # K/V come in transposed already as [cache_head_dim, bkv_sz]; the einsum
            # below contracts head_dim, avoiding an explicit transpose on load. K/V
            # are the NATIVE (un-padded) cache head_dim (hd64-native, ported from a2a).
            k_for_dot = kv_or_k
            assert kv_or_k.shape == v.shape == (cache_head_dim, bkv_sz)
            assert kv_or_k.dtype == v.dtype
        else:
            k_for_dot = kv_or_k
            assert kv_or_k.shape == v.shape == (bkv_sz, head_dim)
            assert kv_or_k.dtype == v.dtype
        # V (standard/seq) or the combined KV (hd64) that step2 consumes for the
        # PV product. Masked to the effective kv_len below when a mask applies,
        # then returned so step2 needs no position logic of its own.
        v_for_pv = kv_or_k if is_hd64 else v
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

        if seq_along_lane:
            # Slice Q down to the native cache_head_dim so the contraction matches
            # the un-padded K (hd64-native). Mosaic allows this VREG lane slice.
            q_dot = q[:, :cache_head_dim] if cache_head_dim < head_dim else q
            s = jnp.einsum(
                "nd,dm->nm", q_dot, k_for_dot, preferred_element_type=jnp.float32
            )
        else:
            s = jnp.einsum(
                "nd,md->nm", q, k_for_dot, preferred_element_type=jnp.float32
            )
        combined_scale = sm_scale
        if has_k_scale:
            combined_scale = combined_scale * k_scale_ref[kv_head_idx]
        if q_scale is not None:
            combined_scale *= q_scale
        if soft_cap is not None:
            s *= combined_scale
            s = soft_cap * jnp.tanh(s / soft_cap)
            log2e_scale = LOG2E
        else:
            log2e_scale = LOG2E * combined_scale

        # skip_mask: the caller guarantees every key in this KV block is visible to
        # every query in the Q block (fully below the causal diagonal, no sliding
        # window, no custom visibility), so the iota/compare/where can be skipped.
        if not skip_mask:
            if WINDOW_ANCHORED:
                kv_base = _window_anchor_tok(kv_len, q_len)
            else:
                kv_base = bkv_idx * bkv_sz
            k_span = kv_base + lax.broadcasted_iota(jnp.int32, s.shape, 1)

            if visibility is not None:
                actual_bq = q.shape[0] // num_q_heads_per_kv_head
                bq_shape = (actual_bq, s.shape[1])
                k_span_bq = bkv_idx * bkv_sz + lax.broadcasted_iota(
                    jnp.int32, bq_shape, 1
                )
                vis_start = lax.broadcast_in_dim(visibility[:, 0], bq_shape, (0,))
                vis_end = lax.broadcast_in_dim(visibility[:, 1], bq_shape, (0,))
                mask_bq = jnp.logical_and(k_span_bq >= vis_start, k_span_bq <= vis_end)
                mask = jnp.repeat(mask_bq, num_q_heads_per_kv_head, axis=0)
            else:
                q_span = (
                    kv_len
                    - q_len
                    + bq_idx * bq_sz
                    + lax.broadcasted_iota(jnp.int32, s.shape, 0)
                    // num_q_heads_per_kv_head
                )
                mask = k_span <= q_span

            # Bound the score to the effective kv_len and mask V to it BEFORE the
            # sliding-window mask, matching upstream v3's ordering. Numerically
            # identical either way (score masks are AND-ed -- commutative -- and
            # the V-mask is independent of the score mask); this keeps the
            # structure aligned with upstream.
            mask = jnp.logical_and(mask, k_span < kv_len)
            # Mask V (or the combined KV for hd64) to the same effective kv_len
            # before step2. The score mask forces p==0 on tail positions, but
            # 0 * Inf/NaN == NaN -- so if the tail V ever holds Inf/NaN the PV sum
            # is still poisoned; zeroing V here keeps it 0 * 0 == 0.
            if seq_along_lane:
                # V is [cache_head_dim, bkv_sz] with the sequence on the LANE
                # axis (1) and a packed dtype. Selecting on the packed tile needs
                # a sublane broadcast of the lane mask, which Mosaic rejects
                # ("Not implemented: Sublane broadcast"), so do the select in
                # float32 -- exactly like the score mask on `s` -- and cast back.
                # Kept values roundtrip through f32 exactly.
                v_span = kv_base + lax.broadcasted_iota(jnp.int32, v_for_pv.shape, 1)
                v_for_pv = jnp.where(
                    v_span < kv_len, v_for_pv.astype(jnp.float32), 0.0
                ).astype(v_for_pv.dtype)
            else:
                # V is [bkv_sz, head_dim]; the kv_len bound is on the SUBLANE
                # axis (0), which Mosaic masks on the packed tile directly.
                v_span = kv_base + lax.broadcasted_iota(jnp.int32, v_for_pv.shape, 0)
                v_for_pv = jnp.where(
                    v_span < kv_len, v_for_pv, jnp.zeros((), v_for_pv.dtype)
                )
            # Sliding-window mask applied last (causal path only; visibility
            # replaces causal+sliding in this kernel, so it is guarded out here).
            if sliding_window is not None and visibility is None:
                mask = jnp.logical_and(mask, q_span - sliding_window < k_span)
            s = jnp.where(mask, s, mask_value)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)

        if attention_sink_ref is not None:
            sinks = attention_sink_ref[kv_head_idx]
            if soft_cap is None:
                sinks = sinks * (1.0 / combined_scale)
            actual_bq_sz = q.shape[0] // num_q_heads_per_kv_head
            m_prev_init = jnp.concat([sinks] * actual_bq_sz, axis=0)
            m_prev = jnp.where(bkv_idx == bkv_idx_start, m_prev_init, head_m_ref[...])
        else:
            m_prev = load_with_init(head_m_ref, -jnp.inf)

        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp2((s - broadcast_minor(m_curr, s.shape)) * log2e_scale)

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp2((m_prev - m_curr) * log2e_scale)
        # When attention_sink is used, m_prev is finite on the first iteration
        # (not -inf), so exp_m_diff is non-zero and l_prev matters. Init to 1.0
        # to account for the virtual sink token's probability mass.
        l_prev_init = 1.0 if attention_sink_ref is not None else 0.0
        l_prev = load_with_init(head_l_ref, l_prev_init)
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

        if seq_along_lane:
            pv = jnp.einsum("nm,dm->nd", p, v_or_kv, preferred_element_type=jnp.float32)
            # v is the native cache_head_dim; pad pv up to the compute head_dim so
            # it matches the (128-padded, lane-tiled) accumulator (hd64-native).
            if cache_head_dim < head_dim:
                pv = jnp.pad(pv, ((0, 0), (0, head_dim - cache_head_dim)))
        else:
            pv = jnp.einsum("nm,md->nd", p, v_or_kv, preferred_element_type=jnp.float32)
        if has_v_scale:
            pv *= v_scale_ref[kv_head_idx]
        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    @jax.named_scope("async_copy")
    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            # Skip DMA if debug mode is enabled.
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        if seq_along_lane:
            # In-kernel KV writeback: fetch the cached tokens (whole 128-lane
            # pages) into [0, cache_pages*page_size) and stage the new-token whole
            # pages into the slack region [cache_pages*page_size, ...). The new
            # tokens are later rolled into place (stitch) and the updated pages are
            # DMA'd back to the cache HBM. All lane-axis DMAs are whole pages
            # (128-aligned), as required by Mosaic.
            kv_len = kv_lens_ref[seq_idx]
            q_start = cu_q_lens_ref[seq_idx]
            q_end = cu_q_lens_ref[seq_idx + 1]
            q_len = q_end - q_start
            if WINDOW_ANCHORED:
                kv_len_start = _window_anchor_tok(kv_len, q_len)
                kv_p_start = kv_len_start // page_size
            else:
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
            new_tok_offset = new_kv_len_start % page_size
            new_page_base = new_kv_len_start - new_tok_offset
            num_new_pages = jnp.where(
                bkv_sz_frm_new > 0,
                (new_tok_offset + bkv_sz_frm_new - 1) // page_size + 1,
                0,
            )
            padded_new = kv_hbm_ref.shape[-1]
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
                            kv_cache_hbm_ref.at[page_idx, :, :, :, pl.ds(0, sz)],
                            vmem_ref.at[:, :, :, pl.ds(i * page_size, sz)],
                            sem,
                            wait=False,
                        )
                with jax.named_scope("fetch_bkv_seq_stage_new"):
                    # New-token pages -> slack [cache_pages, cache_pages+num_new).
                    # The new tokens are contiguous in kv_hbm_ref (from
                    # new_page_base) and in the VMEM slack (from cache_pages), so
                    # issue ONE coalesced whole-page DMA instead of a per-page
                    # descriptor. This cuts the scalar-core DMA-issue cost that
                    # dominates short-context prefill, and subsumes the decode case
                    # (num_new_pages <= 1). kv_hbm is padded by an extra page, so
                    # the num_new_pages whole pages never run past the buffer; the
                    # min() is a defensive page-aligned clamp.
                    src = pl.multiple_of(new_page_base, page_size)
                    total_new = pl.multiple_of(
                        jnp.minimum(num_new_pages * page_size, padded_new - src),
                        page_size,
                    )
                    dst_off = pl.multiple_of(cache_pages * page_size, page_size)

                    @pl.when(num_new_pages > 0)
                    def _stage_new():
                        _async_copy(
                            kv_hbm_ref.at[:, :, :, pl.ds(src, total_new)],
                            vmem_ref.at[:, :, :, pl.ds(dst_off, total_new)],
                            sem,
                            wait=False,
                        )

            else:
                total = pl.multiple_of(
                    (cache_pages + num_new_pages) * page_size, page_size
                )
                dst = vmem_ref.at[:, :, :, pl.ds(0, total)]
                _async_copy(src=dst, dst=dst, sem=sem, wait=True)
            return kv_len_start, bkv_sz_frm_new

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        if update_kv_cache:
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        else:
            # KV-share: source layer already wrote the new K/V into the
            # (redirected) cache slot before this layer's call, so read
            # everything from cache and skip writeback. Mirrors upstream
            # tpu-inference v3's update_kv_cache=False semantics.
            kv_left_frm_cache = kv_left
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        debug_print(
            f"[RPA debug] -----------{'wait' if wait else 'start'}_fetch_bkv-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_idx={}", bkv_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] kv_len_start={}", kv_len_start)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_left={}", kv_left)
        debug_print("[RPA debug] kv_left_frm_cache={}", kv_left_frm_cache)
        debug_print("[RPA debug] kv_left_frm_new={}", kv_left_frm_new)
        debug_print("[RPA debug] bkv_sz_frm_cache={}", bkv_sz_frm_cache)
        debug_print("[RPA debug] bkv_sz_frm_new={}", bkv_sz_frm_new)
        debug_print("[RPA debug] page_indices_offset={}", kv_p_start)

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop.
            for i in range(bkv_p):
                # Ensure only effective kvs are copied.
                sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
                # If the page index is out of bound, we set page_idx to the last page.
                # And there will be no copy since sz will be 0.
                page_idx = physical_page_index(seq_idx, kv_p_start + i)
                _async_copy(
                    cache_hbm_ref.at[pl.ds(page_idx * page_size, sz)],
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )
                debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)

            new_kv_len_start = q_end - kv_left_frm_new
            debug_print("[RPA debug] new_kv_len_start={}", new_kv_len_start)
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, bkv_sz_frm_cache + bkv_sz_frm_new)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    @jax.named_scope("update_kv_cache")
    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_update_kv_cache-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] offset={}", offset)
        debug_print("[RPA debug] update_sz={}", update_sz)
        debug_print("[RPA debug] bkv_id={}", bkv_id)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_p_end={}", kv_p_end)
        debug_print("[RPA debug] ignore={}", ignore)
        debug_print("[RPA debug] p_ignore={}", p_ignore)
        debug_print("[RPA debug] page_indices_offset={}", kv_p_start)

        if not wait:

            def loop_body(i, states):
                update_sz, ignore = states
                sz = jnp.minimum(page_size - ignore, update_sz)

                _async_copy(
                    vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                    cache_hbm_ref.at[
                        pl.ds(
                            physical_page_index(seq_idx, kv_p_start + i) * page_size
                            + ignore,
                            sz,
                        )
                    ],
                    sem,
                    wait=False,
                )
                debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)
                return update_sz - sz, 0

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),  # total transfer size
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )

    @jax.named_scope("fetch_bq")
    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            f"[RPA debug] -----------{'wait' if wait else 'start'}_fetch_bq-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bq_idx={}", bq_idx)
        debug_print("[RPA debug] bq_sem_idx={}", bq_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

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

        debug_print(
            f"[RPA debug] -----------{'wait' if wait else 'start'}_send_bo-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bo_idx={}", bo_idx)
        debug_print("[RPA debug] bo_sem_idx={}", bo_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

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

    @jax.named_scope("fetch_vis")
    def _fetch_vis(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        vis_sem = sems.at[4, bq_sem_idx]
        if not wait:
            q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
            _async_copy(
                visibility_hbm_ref.at[pl.ds(q_len_start, bq_sz)],
                vis_scratch_ref.at[bq_sem_idx],
                vis_sem,
                wait=False,
            )
        else:
            vis_dst = vis_scratch_ref.at[bq_sem_idx]
            _async_copy(src=vis_dst, dst=vis_dst, sem=vis_sem, wait=True)

    @jax.named_scope("start_fetch_vis")
    def start_fetch_vis(seq_idx, bq_idx, bq_sem_idx):
        _fetch_vis(seq_idx, bq_idx, bq_sem_idx)

    @jax.named_scope("wait_fetch_vis")
    def wait_fetch_vis(bq_sem_idx):
        _fetch_vis(0, 0, bq_sem_idx, wait=True)

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

    @jax.named_scope("start_update_kv_cache")
    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    @jax.named_scope("wait_update_kv_cache")
    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

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

    def strided_load(ref, start, step):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        if is_hd64:
            _, l = ref.shape  # noqa
            assert l == 128
            vec = ref[start::step]
        else:
            r, l = ref.shape  # noqa
            assert l % 128 == 0
            folds = l // 128
            ref = ref.reshape(r * folds, 128)
            start *= folds
            step *= folds
            vec = jnp.concat([ref[start + i :: step] for i in range(folds)], axis=1)
        return vec

    @jax.named_scope("strided_load_bkv")
    def strided_load_bkv(bkv_sem_idx, start, step):
        assert start % kv_packing == 0
        assert step % kv_packing == 0
        start //= kv_packing
        step //= kv_packing
        kv_ref = (
            bkv_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * step, head_dim)
        )

        if is_hd64:
            kv = strided_load(kv_ref, start, step)
            bitwidth = 32 // kv_packing
            repack_ty = jnp.dtype(f"uint{bitwidth}")
            lst = []
            for i in range(0, kv_packing):
                cur_kv = pltpu.bitcast(
                    (kv >> (i * bitwidth)).astype(repack_ty), kv_dtype
                )
                lst.append(cur_kv)
            return lst

        if kv_packing == 1:
            k = strided_load(kv_ref, start, step)
            v = strided_load(kv_ref, start + 1, step)

            k = pltpu.bitcast(k, kv_dtype)
            v = pltpu.bitcast(v, kv_dtype)
            return [(k, v)]

        kv = strided_load(kv_ref, start, step)
        bitwidth = 32 // kv_packing

        # If we want to convert 32-bits into 32//N number of N-bits value, naive
        # approach would be to perform 32//N number of 32-bits to N-bits conversion.
        # However, we can reduce number of instructions by utilizing binary tree.
        # 0: [32]
        # 1: [16, 16]
        # ...
        # log2(32//N): [N, N, ... N]

        def _convert_to_target_bitwidth(val, target_bitwidth: int):
            curr_dtype = val.dtype
            curr_bitwidth = get_dtype_bitwidth(curr_dtype)
            assert target_bitwidth != curr_bitwidth, "No conversion is needed."

            # We split val into two vals (left and right) where each have half of the
            # original bitwidth.
            next_bitwidth = curr_bitwidth // 2
            next_dtype = jnp.dtype(f"uint{next_bitwidth}")

            left = val.astype(next_dtype)

            # Bitwise shift is only supported in uint32.
            val_u32 = pltpu.bitcast(val, jnp.uint32)
            val_u32_shifted = val_u32 >> next_bitwidth
            # Convert back to original dtype.
            val_shifted = pltpu.bitcast(val_u32_shifted, curr_dtype)
            right = val_shifted.astype(next_dtype)

            if next_bitwidth == target_bitwidth:
                k = pltpu.bitcast(left, kv_dtype)
                v = pltpu.bitcast(right, kv_dtype)
                return [(k, v)]
            else:
                left_out = _convert_to_target_bitwidth(
                    left,
                    target_bitwidth=target_bitwidth,
                )
                right_out = _convert_to_target_bitwidth(
                    right,
                    target_bitwidth=target_bitwidth,
                )
                return left_out + right_out

        return _convert_to_target_bitwidth(kv, target_bitwidth=bitwidth)

    @jax.named_scope("load_bkv_seq")
    def load_bkv_seq(bkv_sem_idx, kv_head_idx):
        """Load one KV head from the SEQ_ALONG_LANE buffer as [head_dim, bkv_sz].

        The buffer is [2, 2*kv_heads, hd_aligned//pk, pk, bkv_sz]. For head h, the
        K rows live at 2h and V rows at 2h+1; reshape merges (hd//pk, pk) back into
        the contiguous head_dim. We keep the [head_dim, bkv_sz] orientation (head
        on sublanes, seq on lanes) and let the seq-layout QK/PV einsums contract
        head_dim directly -- this avoids the explicit seq<->head_dim transpose,
        which was a VPU op on the critical path. head_dim is padded (on the
        sublane axis) up to the 128-aligned compute head_dim.
        """
        with jax.named_scope("load_bkv_seq_read"):
            kbuf = bkv_x2_ref[bkv_sem_idx, 2 * kv_head_idx, :, :, :bkv_sz][...]
            vbuf = bkv_x2_ref[bkv_sem_idx, 2 * kv_head_idx + 1, :, :, :bkv_sz][...]
        k = kbuf.reshape(cache_head_dim, bkv_sz)
        v = vbuf.reshape(cache_head_dim, bkv_sz)
        return k, v

    @jax.named_scope("stitch_seq")
    def stitch_seq(bkv_sem_idx, bkv_sz_cache, cache_pages, new_tok_offset):
        """Roll staged new-token pages into place in the bkv buffer (lane axis).

        Mirrors the batched SEQ_ALONG_LANE stitch: single-token decode
        (decode_q_len == 1) rolls only the boundary VREG (O(1)); spec decode
        (decode_q_len > 1) rolls a bounded 3-page window at the boundary
        (O(decode_q_len)); prefill/mixed roll the whole staged region (O(N)).
        Operates on the u32 bitcast view so head/head_dim structure collapses
        into the row axis.
        """
        num_lanes = pltpu.get_tpu_info().num_lanes
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        # The single-token decode O(1) fast path rolls one 128-lane VREG, so it
        # needs 2 pages of staging slack AND requires the new token's within-page
        # offset to be < 128 (true only at page_size==128). Every other case --
        # spec decode (decode_q_len > 1) and page_size>128 single-token decode --
        # goes through the bounded 3-page-window roll, which needs 3 pages of
        # slack. prefill/mixed keep the general roll. Must match the
        # bkv_double_buf allocation in _ragged_paged_attention_impl.
        stage_slack_pages = 2 if (decode_q_len == 1 and page_size == 128) else 3
        v_len = bkv_sz + stage_slack_pages * page_size
        vmem_u32 = bkv_x2_ref.at[bkv_sem_idx].bitcast(jnp.uint32)
        if rpa_kind == 0 and decode_q_len == 1 and page_size == 128:
            # Decode (single new token): roll only the boundary VREG (O(1)).
            lanes_per_col = v_len // num_lanes
            strided = vmem_u32.reshape(-1, num_lanes)
            outer_dim = strided.shape[0] // lanes_per_col
            dst_chunk = bkv_sz_cache // num_lanes
            dst_vreg = strided[pl.ds(dst_chunk, outer_dim, lanes_per_col)]
            src_vreg = strided[pl.ds(cache_pages, outer_dim, lanes_per_col)]
            dst_rel = bkv_sz_cache % num_lanes
            rolled = pltpu.roll(src_vreg, dst_rel - new_tok_offset, axis=1)
            lane_idx = lax.broadcasted_iota(jnp.int32, dst_vreg.shape, 1)
            merged = lax.select(lane_idx >= dst_rel, rolled, dst_vreg)
            strided[pl.ds(dst_chunk, outer_dim, lanes_per_col)] = merged
        elif rpa_kind == 0:
            # Bounded-window decode roll. Handles spec decode (decode_q_len > 1)
            # and page_size>128 single-token decode. The new tokens are a short
            # run (<= decode_q_len) at the cache/new boundary, so only the 1-3
            # lane columns spanning [bkv_sz_cache, ...) plus the staged source need
            # to move. Roll a bounded 3-page window, page-aligned at the boundary
            # page, instead of the whole bkv_sz buffer -- O(decode_q_len) VPU work
            # instead of O(bkv_sz). The window is sized so it always covers both
            # the destination (boundary page[, +1]) and the staged source (cols
            # cache_pages[, +1], where cache_pages in {dst_page, dst_page+1}), i.e.
            # up to dst_page+2, and fits within the 3-page slack.
            window_lanes = 3 * page_size
            total_head_words = actual_num_kv_heads * 2 * cache_head_dim // kv_packing
            words_per_sublane = total_head_words // num_sublanes
            reshaped = vmem_u32.reshape(words_per_sublane, num_sublanes, v_len)
            # Page (not 128-lane-chunk) index of the boundary: this operates on the
            # full v_len lane axis in absolute lane units, so win_base must be the
            # page-aligned lane offset of the page containing bkv_sz_cache.
            dst_page = bkv_sz_cache // page_size
            win_base = pl.multiple_of(dst_page * page_size, page_size)
            dst_rel = bkv_sz_cache - win_base
            src_rel = cache_pages * page_size + new_tok_offset - win_base
            roll_shift = (dst_rel - src_rel) % window_lanes

            # Skip the roll when the staged data is already aligned (identity).
            @pl.when(roll_shift != 0)
            def _do_roll_bounded():
                window = reshaped[..., pl.ds(win_base, window_lanes)]
                rolled = pltpu.roll(window, roll_shift, axis=2)
                lane_idx = lax.broadcasted_iota(jnp.int32, window.shape, 2)
                # lanes < dst_rel keep the cached tail; >= dst_rel take the rolled
                # staged run (positions past the new tokens are masked downstream,
                # mirroring the general path's [bkv_sz_cache:] overwrite).
                merged = lax.select(lane_idx >= dst_rel, rolled, window)
                reshaped[..., pl.ds(win_base, window_lanes)] = merged

        else:
            # Prefill/mixed: general roll over the staged region (handles any
            # count). Spec decode (decode_q_len > 1) is handled by the bounded
            # window branch above; this path only runs for rpa_kind != 0.
            total_head_words = actual_num_kv_heads * 2 * cache_head_dim // kv_packing
            words_per_sublane = total_head_words // num_sublanes
            reshaped = vmem_u32.reshape(words_per_sublane, num_sublanes, v_len)
            roll_shift = (
                bkv_sz_cache - (cache_pages * page_size + new_tok_offset)
            ) % v_len

            # Skip the O(N) roll when the staged data is already aligned
            # (roll_shift == 0, e.g. fresh / page-aligned prefill): the roll would
            # be an identity over the whole buffer at full VPU cost.
            @pl.when(roll_shift != 0)
            def _do_roll():
                rolled = pltpu.roll(reshaped[...], roll_shift, axis=2)
                lane_idx = lax.broadcasted_iota(
                    jnp.int32, rolled[..., :bkv_sz].shape, 2
                )
                merged = lax.select(
                    lane_idx >= bkv_sz_cache,
                    rolled[..., :bkv_sz],
                    reshaped[..., :bkv_sz],
                )
                reshaped[..., :bkv_sz] = merged

    def _writeback_seq_copy(seq_idx, bkv_idx, bkv_sem_idx, p):
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]
        if WINDOW_ANCHORED:
            kv_len = kv_lens_ref[seq_idx]
            q_len = cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx]
            kv_p_start = _window_anchor_tok(kv_len, q_len) // page_size
        else:
            kv_p_start = bkv_idx * bkv_p
        page_idx = physical_page_index(seq_idx, kv_p_start + p)
        return pltpu.make_async_copy(
            vmem_ref.at[
                :, :, :, pl.ds(pl.multiple_of(p * page_size, page_size), page_size)
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
            return None

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
                return None

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
        prefetch_num_bkv = num_bkv = 1 if WINDOW_ANCHORED else cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        def _sw_bq_start(bq_idx):
            # Per-q-block sliding-window lower bound (bkv index). The earliest
            # query in this q-block is at (kv_len - q_len + bq_idx*actual_bq_sz);
            # it cannot see keys below (that - sliding_window), so start the bkv
            # loop at the block containing that position. _sw_bq_start(0) ==
            # the seq-level bkv_idx_start, so bq 0 (and the cache writeback gated
            # on bq_idx==0) is unchanged; only later q-blocks start higher.
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

            if visibility_hbm_ref is not None or sliding_window is None:
                # When visibility is used or sliding window is disabled, starting
                # bkv_idx of next request is always 0.
                next_bkv_idx_start = 0
            else:
                # Determine starting bkv_idx of next request based on whether next
                # request is from the same sequence or next sequence.
                same_seq_start = (
                    _sw_bq_start(next_bq_idx) if PREFILL_SW_BOUND else bkv_idx_start
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
            if PREFILL_SW_BOUND:
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
                if visibility_hbm_ref is not None:
                    start_fetch_vis(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            if visibility_hbm_ref is not None:
                wait_fetch_vis(bq_sem_idx)
                vis_block = vis_scratch_ref.at[bq_sem_idx, :actual_bq_sz][...]
            else:
                vis_block = None

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
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                # Start updating bkv to kv cache if applicable.
                # Only needed in first bq loop.
                # KV-share: skip the cache write when update_kv_cache=False
                # so shared layers don't overwrite the source layer's slot.
                # SEQ_ALONG_LANE does its own in-kernel stitch + writeback inside
                # compute_with_bkv_seq, so skip the head-along row writeback here.
                if update_kv_cache and not seq_along_lane:

                    @pl.when(jnp.logical_and(update_sz > 0, bq_idx == 0))
                    def update_cur_bkv_to_cache():
                        start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                return bkv_sem_idx

            def _compute_with_bkv_preamble(bkv_idx):
                """Shared prefetch + debug logic for compute_with_bkv variants."""
                bkv_sem_idx = prefetch_with_bkv(bkv_idx)
                debug_print("[RPA debug] -----------flash attention-----------")
                debug_print("[RPA debug] seq_idx={}", seq_idx)
                debug_print("[RPA debug] bq_idx={}", bq_idx)
                debug_print("[RPA debug] bkv_idx={}", bkv_idx)
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

            @jax.named_scope("compute_with_bkv_hd64")
            def compute_with_bkv_hd64(bkv_idx, _, skip_mask=False):
                bkv_sem_idx = _compute_with_bkv_preamble(bkv_idx)
                if debug_mode:
                    return

                prev_bq_shape_0 = None
                prev_kv_head_bkv = None
                prev_kv_head_idx = None
                prev_kv_head_p = None
                prev_kv_head_exp_m_diff = None

                for kv_head_start in range(0, actual_num_kv_heads, kv_packing):
                    bkv_lst = strided_load_bkv(
                        bkv_sem_idx,
                        kv_head_start,
                        num_kv_heads,
                    )
                    assert len(bkv_lst) == kv_packing
                    for i in range(kv_packing):
                        cur_kv_head_idx = kv_head_start + i
                        if cur_kv_head_idx >= actual_num_kv_heads:
                            break
                        cur_kv_head_bq = load_bq(
                            bq_sem_idx, cur_kv_head_idx, actual_bq_sz=actual_bq_sz
                        )
                        cur_kv_head_bkv = bkv_lst[i]
                        cur_kv_head_p, cur_kv_head_exp_m_diff, cur_kv_head_bkv = (
                            flash_attention_step1_qk_softmax(
                                cur_kv_head_bq,
                                cur_kv_head_bkv,
                                bq_idx=bq_idx,
                                bkv_idx=bkv_idx,
                                kv_head_idx=cur_kv_head_idx,
                                visibility=vis_block,
                                skip_mask=skip_mask,
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
                        prev_kv_head_bkv = cur_kv_head_bkv
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

            @jax.named_scope("compute_with_bkv")
            def compute_with_bkv(bkv_idx, _, skip_mask=False):
                bkv_sem_idx = _compute_with_bkv_preamble(bkv_idx)
                if debug_mode:
                    return

                prev_bq_shape_0 = None
                prev_kv_head_bkv = None
                prev_kv_head_idx = None
                prev_kv_head_p = None
                prev_kv_head_exp_m_diff = None

                # NOTE: kv_packing is divided by 2 because k and v are packed together.
                heads_per_load = max(1, kv_packing // 2)
                for kv_head_start in range(0, actual_num_kv_heads, heads_per_load):
                    bkv_lst = strided_load_bkv(
                        bkv_sem_idx,
                        kv_head_start * 2,
                        num_kv_heads,
                    )
                    assert len(bkv_lst) == heads_per_load
                    for i in range(heads_per_load):
                        cur_kv_head_idx = kv_head_start + i
                        if cur_kv_head_idx >= actual_num_kv_heads:
                            break

                        cur_kv_head_bq = load_bq(
                            bq_sem_idx, cur_kv_head_idx, actual_bq_sz=actual_bq_sz
                        )
                        bk, bv = bkv_lst[i]
                        cur_kv_head_p, cur_kv_head_exp_m_diff, bv = (
                            flash_attention_step1_qk_softmax(
                                cur_kv_head_bq,
                                bk,
                                bv,
                                bq_idx=bq_idx,
                                bkv_idx=bkv_idx,
                                kv_head_idx=cur_kv_head_idx,
                                visibility=vis_block,
                                skip_mask=skip_mask,
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
                if WINDOW_ANCHORED:
                    kv_len_start = _window_anchor_tok(kv_len, q_len)
                else:
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
                new_tok_offset = new_kv_len_start % page_size

                @pl.when(bkv_sz_frm_new > 0)
                def _do_stitch():
                    stitch_seq(
                        bkv_sem_idx, bkv_sz_frm_cache, cache_pages, new_tok_offset
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
            def compute_with_bkv_seq(bkv_idx, _, skip_mask=False):
                bkv_sem_idx = _compute_with_bkv_preamble(bkv_idx)
                if debug_mode:
                    return

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
                            visibility=vis_block,
                            skip_mask=skip_mask,
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
                if seq_along_lane:
                    # Persist (stitch + writeback) this causally-masked block's new
                    # KV to the cache. Head persists inside prefetch_with_bkv (so
                    # this path covers it for free), but seq's writeback lives
                    # outside prefetch, so without doing it here multi-bq prefill
                    # would leave these blocks unwritten in the cache.
                    bkv_sem_idx = _compute_with_bkv_preamble(bkv_id)
                    if debug_mode:
                        return
                    _seq_persist_bkv(bkv_id, bkv_sem_idx)
                else:
                    prefetch_with_bkv(bkv_id)

            if seq_along_lane:
                compute_with_bkv_fn = compute_with_bkv_seq
            elif is_hd64:
                compute_with_bkv_fn = compute_with_bkv_hd64
            else:
                compute_with_bkv_fn = compute_with_bkv

            # Limit KV block processing to avoid fully masked regions.
            # This optimization skips KV blocks beyond the last query position in the
            # current batch, avoiding unnecessary computation for fully masked regions.
            if visibility_hbm_ref is not None:
                # For visibility use the max vis_end across the Q block.
                max_vis_end = jnp.max(vis_block[:, 1])
                adjusted_num_bkv = jnp.minimum(cdiv(max_vis_end + 1, bkv_sz), num_bkv)
            else:
                # For causal masking queries can only attend to keys at positions <= their own position.
                adjusted_num_bkv = jnp.minimum(
                    cdiv(kv_len - q_len + bq_idx * bq_sz + actual_bq_sz, bkv_sz),
                    num_bkv,
                )
            nonlocal prefetch_num_bkv
            prefetch_num_bkv = lax.select(bq_idx == 0, num_bkv, adjusted_num_bkv)

            # Causal mask-skip: KV blocks strictly below the diagonal are fully visible
            # to every query in this Q block, so their mask can be skipped (saves the
            # iota/compare/where VPU work on the majority of blocks). Split the loop
            # into [start, visible_end) without mask and [visible_end, end) with mask.
            # Only valid for plain causal (no custom visibility, no sliding window); the
            # prefetch chain stays continuous across the split (next bkv = cur + 1).
            if visibility_hbm_ref is None and sliding_window is None:
                # Block j is fully visible iff (j+1)*bkv_sz-1 <= min q_span, where
                # min q_span = kv_len - q_len + bq_idx*bq_sz (first query in the block).
                visible_end = jnp.clip(
                    (kv_len - q_len + bq_idx * bq_sz + 1) // bkv_sz,
                    bkv_idx_start,
                    adjusted_num_bkv,
                )
                lax.fori_loop(
                    bkv_idx_start,
                    visible_end,
                    lambda i, c: compute_with_bkv_fn(i, c, skip_mask=True),
                    None,
                    unroll=False,
                )
                lax.fori_loop(
                    visible_end,
                    adjusted_num_bkv,
                    compute_with_bkv_fn,
                    None,
                    unroll=False,
                )
            else:
                lax.fori_loop(
                    bkv_idx_start,
                    adjusted_num_bkv,
                    compute_with_bkv_fn,
                    None,
                    unroll=False,
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
                l = broadcast_minor(l_ref[...], acc.shape)  # noqa
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
        q_start, q_end, q_len, kv_len, bkv_idx_start, next_seq_bkv_idx_start = (
            get_seq_info(seq_idx, seq_idx_offset)
        )

        debug_print("[RPA debug] ======= In loop seq_idx={}", seq_idx)
        debug_print("[RPA debug] num_seqs={}", num_seqs)
        debug_print("[RPA debug] decode_end={}", decode_end)
        debug_print("[RPA debug] prefill_end={}", prefill_end)
        debug_print("[RPA debug] mixed_end={}", mixed_end)
        debug_print("[RPA debug] bkv_p={}", bkv_p)
        debug_print("[RPA debug] page_size={}", page_size)
        debug_print("[RPA debug] pages_per_seq={}", pages_per_seq)
        debug_print("[RPA debug] bkv_sz={}", bkv_sz)
        debug_print("[RPA debug] bq_sz={}", bq_sz)
        debug_print("[RPA debug] q_start={}", q_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] q_len={}", q_len)
        debug_print("[RPA debug] kv_len={}", kv_len)
        debug_print("[RPA debug] rpa_kind={}", rpa_kind)

        @pl.when(_seq_idx == 0)
        def prologue():
            with jax.named_scope("prologue"):
                start_fetch_bq(seq_idx, 0, 0)
                if visibility_hbm_ref is not None:
                    start_fetch_vis(seq_idx, 0, 0)

                # Initialize bkv_x2_ref to zeros to avoid NaN issues from accessing
                # uninitialized memory. Bitcast into int32 to avoid tiling issues.
                bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32).reshape(
                    (2, -1, 8, 128)
                )
                zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)

                # To pipeline VST and DMA, we divide the initialization into two steps.
                bkv_x2_int32_ref[0] = zeros
                start_fetch_bkv(seq_idx, bkv_idx_start, 0)
                bkv_x2_int32_ref[1] = zeros

        @pl.when(rpa_kind == 0)
        def process_decode():
            process(static_q_len=decode_q_len)

        @pl.when(rpa_kind == 1)
        def process_prefill():
            process(static_q_len=chunk_prefill_size)

        @pl.when(rpa_kind == 2)
        def process_mixed():
            process()

        @pl.when(_seq_idx == num_seqs - 1)
        def epilogue():
            with jax.named_scope("epilogue"):
                for i in range(2):
                    wait_send_bo(i)
                    if seq_along_lane:
                        # Flush any deferred decode writeback before the kernel
                        # returns (no-op for prefill/mixed, which write back
                        # synchronously).
                        wait_writeback_seq(i)
                    else:
                        wait_update_kv_cache(i)

    q_len = kv_len = bkv_idx_start = next_seq_bkv_idx_start = seq_idx = -1
    seq_idx_offset = distribution_ref[rpa_kind - 1] if rpa_kind > 0 else 0
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


def _seq_along_lane_new_kv(k, v, page_size: int = 128):
    """Build the new K/V in SEQ_ALONG_LANE layout for in-kernel DMA writeback.

    Returns [2*kv_heads, hd_aligned//pk, pk, padded_tokens] (tokens on the lane
    axis). Tokens are padded to a ``page_size`` multiple plus one extra page so the
    kernel can fetch whole ``page_size``-lane new-token pages without going out of
    bounds (the fetch in ``_fetch_bkv`` clamps to this buffer with a
    ``multiple_of(..., page_size)`` hint, so the pad must be a page multiple).
    ``page_size`` is a positive multiple of 128, so the result stays 128-lane
    aligned. At page_size==128 this reproduces the original padding exactly.
    """
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    hd_aligned = align_to(actual_head_dim, num_sublanes * kv_packing)
    padded = align_to(max_num_tokens, page_size) + page_size
    merged = jnp.concatenate([k, v], axis=-1).reshape(
        max_num_tokens, actual_num_kv_heads * 2, actual_head_dim
    )
    merged = jnp.pad(
        merged,
        ((0, padded - max_num_tokens), (0, 0), (0, hd_aligned - actual_head_dim)),
        constant_values=0,
    )
    return merged.reshape(
        padded, actual_num_kv_heads * 2, hd_aligned // kv_packing, kv_packing
    ).transpose(1, 2, 3, 0)


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    attention_sink: jax.Array | None = None,
    is_hd64=False,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
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
        kv = _seq_along_lane_new_kv(k, v, page_size=page_size)
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
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, head_dim]
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
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, head_dim]
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
    if num_kv_pages_per_block is not None:
        if num_kv_pages_per_block <= 0:
            raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_block is not None:
        if num_queries_per_block <= 0:
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


def get_kernel_scope_name(bq_size, bkv_p, page_size, rpa_kind, is_hd64=False):
    prefix = "RPA-HD_64" if is_hd64 else "RPA"
    return f"{prefix}-bq_{bq_size}-bkvp_{bkv_p}-p_{page_size}-{rpa_kind}"


def _ragged_paged_attention_impl(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, head_dim]
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
    bkv_p_decode,
    bq_sz_decode,
    bkv_p_prefill,
    bq_sz_prefill,
    bkv_p_mixed,
    bq_sz_mixed,
    chunk_prefill_size: int | None = None,
    decode_q_len: int = 1,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
    # KV-share: when False, skip the new-K/V writeback into kv_cache and read
    # the entire kv_left from cache (the source layer is assumed to have
    # populated it).
    update_kv_cache: bool = True,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
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
    if seq_along_lane:
        # SEQ_ALONG_LANE writes the new K/V into the cache in-place via DMA inside
        # the pallas kernel (see _fetch_bkv / stitch / writeback), so there is no
        # out-of-kernel scatter and update_kv_cache stays True.
        is_hd64 = False
    else:
        is_hd64 = detect_is_hd64(actual_num_kv_heads, actual_head_dim, kv_cache)

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
        )

    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q, kv, attention_sink = prepare_inputs(
        q,
        k,
        v,
        attention_sink,
        is_hd64=is_hd64,
        kv_layout=kv_layout,
        # SEQ_ALONG_LANE new-KV padding must be a multiple of the cache page_size
        # (lane/minor axis) so the in-kernel whole-page fetch stays in bounds.
        page_size=(kv_cache.shape[-1] if seq_along_lane else 128),
    )
    (
        _,
        max_num_tokens,
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

    def _launch_kernel(i: int):
        match i:
            case 0:
                # decode only
                grid = (distribution[0],)
                bkv_p = bkv_p_decode
                bq_sz = bq_sz_decode
                # Backstops the autotuned-bq_sz path: when num_queries_per_block is
                # None the block size comes from get_tuned_block_sizes, so the
                # entrypoint ValueError can't see it.
                assert bq_sz >= decode_q_len, (
                    f"decode bq_sz={bq_sz} must be >= decode_q_len={decode_q_len} "
                    "to keep num_bq == 1"
                )
            # case 1:
            #     # prefill only
            #     grid = (distribution[1] - distribution[0],)
            #     bkv_p = bkv_p_prefill
            #     bq_sz = bq_sz_prefill
            case 1:
                # mixed
                grid = (distribution[2] - distribution[0],)
                bkv_p = bkv_p_mixed
                bq_sz = bq_sz_mixed
            case _:
                assert False

        bkv_sz = bkv_p * page_size

        def _page_indices_index_map(
            program_id,
            _kv_lens_ref,
            _cu_q_lens_ref,
            distribution_ref,
            *_unused_scalar_refs,
        ):
            seq_idx_offset = distribution_ref[i - 1] if i > 0 else 0
            return (program_id + seq_idx_offset,)

        def _next_page_indices_index_map(
            program_id,
            _kv_lens_ref,
            _cu_q_lens_ref,
            distribution_ref,
            *_unused_scalar_refs,
        ):
            seq_idx_offset = distribution_ref[i - 1] if i > 0 else 0
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

        if seq_along_lane:
            # [2, 2*kv_heads, hd_aligned//pk, pk, bkv_sz + slack*page_size]
            # The extra slack pages stage fetched new-token pages before they are
            # stitched into place (lane axis). Must match stage_slack_pages in
            # stitch_seq: the single-token decode O(1) fast path (decode_q_len==1
            # and page_size==128) uses 2; every other case (spec decode, or
            # page_size>128 single-token decode) uses the bounded 3-page window.
            stage_slack_pages = 2 if (decode_q_len == 1 and page_size == 128) else 3
            bkv_double_buf = pltpu.VMEM(
                (2, *kv_cache.shape[1:4], bkv_sz + stage_slack_pages * page_size),
                kv_cache.dtype,
            )
        else:
            bkv_double_buf = pltpu.VMEM(
                (2, bkv_sz, *kv_cache.shape[2:]),
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

        scope_name = get_kernel_scope_name(bq_sz, bkv_p, page_size, i, is_hd64)
        n_scalar = len(scalar_prefetches)
        has_sink = attention_sink is not None
        has_vis = visibility is not None
        base_kernel_fn = functools.partial(
            _ragged_paged_attention_kernel,
            is_hd64=is_hd64,
            kv_layout=kv_layout,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            has_k_scale=k_scale is not None,
            has_v_scale=v_scale is not None,
            chunk_prefill_size=chunk_prefill_size,
            decode_q_len=decode_q_len,
            pages_per_seq=pages_per_seq,
            bkv_p=bkv_p,
            bq_sz=bq_sz,
            rpa_kind=i,
            debug_mode=debug_mode,
            update_kv_cache=update_kv_cache,
        )

        if visibility is not None:
            # Pad max_num_tokens by bq_sz to prevent any OOB DMA
            # Pad 2 to 2 + 126 to satisfy DMA requirement of 128 in the minor dimension
            visibility_hbm = jnp.pad(visibility, ((0, bq_sz), (0, 126)))
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))

        def pallas_kernel_fn(
            *args,
            _base=base_kernel_fn,
            _n_scalar=n_scalar,
            _has_sink=has_sink,
            _has_vis=has_vis,
        ):
            idx = _n_scalar
            page_indices_ref = args[idx]
            next_page_indices_ref = args[idx + 1]
            idx += 2
            base_inputs = args[idx : idx + 3]
            idx += 3

            sink_ref = args[idx] if _has_sink else None
            if _has_sink:
                idx += 1

            vis_ref = args[idx] if _has_vis else None
            if _has_vis:
                idx += 1

            rest = list(args[idx:])
            if not _has_vis:
                # Inject None for vis_scratch_ref (last scratch arg).
                rest.append(None)
            scalar_args = args[:_n_scalar]
            return _base(
                scalar_args[0],
                page_indices_ref,
                next_page_indices_ref,
                *scalar_args[1:],
                *base_inputs,
                sink_ref,
                vis_ref,
                *rest,
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

    for i in range(2):
        # output shares buffer with q, updated_kv_cache shares buffer with kv_cache
        output, updated_kv_cache = _launch_kernel(i)
        q = output
        kv_cache = updated_kv_cache

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
        "sm_scale",
        "sliding_window",
        "soft_cap",
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
def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, head_dim]
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
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
):
    """Ragged paged attention that supports mixed prefill and decode.

    Supports both standard and packed (hd64) KV cache layouts. The layout is
    auto-detected from the kv_cache shape.

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

    Returns:
      The output of the attention.
    """
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
            _kq = jnp.repeat(jnp.asarray(k_scale, jnp.float32), _nq_per_kv, axis=0)
            queries = (queries.astype(jnp.float32) * _kq[None]).astype(queries.dtype)
            k_scale = None
        if _pc_v_is_2d:
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

    is_hd64 = (
        False
        if seq_along_lane
        else detect_is_hd64(actual_num_kv_heads, actual_head_dim, kv_cache)
    )

    if num_kv_pages_per_block is None or num_queries_per_block is None:
        (
            bkv_p_decode,
            bq_sz_decode,
            bkv_p_prefill,
            bq_sz_prefill,
            bkv_p_mixed,
            bq_sz_mixed,
        ) = get_tuned_block_sizes(
            queries.dtype,
            kv_cache.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            actual_head_dim,
            page_size,
            max_num_tokens,
            pages_per_seq,
            chunk_prefill_size,
            sliding_window,
            is_hd64=is_hd64,
            seq_along_lane=seq_along_lane,
            decode_q_len=decode_q_len,
        )
    else:
        bkv_p_decode = bkv_p_prefill = bkv_p_mixed = num_kv_pages_per_block
        bq_sz_decode = bq_sz_prefill = bq_sz_mixed = num_queries_per_block

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
        decode_q_len=decode_q_len,
        bkv_p_decode=bkv_p_decode,
        bq_sz_decode=bq_sz_decode,
        bkv_p_prefill=bkv_p_prefill,
        bq_sz_prefill=bq_sz_prefill,
        bkv_p_mixed=bkv_p_mixed,
        bq_sz_mixed=bq_sz_mixed,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        update_kv_cache=update_kv_cache,
        kv_layout=kv_layout,
    )

    if _pc_v_scale is not None:
        # Per-channel V dequant: out (x)= v_scale (factors out of PV+normalize).
        _o, _upd = _pc_out
        _o = (_o.astype(jnp.float32) * _pc_v_scale[None]).astype(_o.dtype)
        _pc_out = (_o, _upd)
    return _pc_out
