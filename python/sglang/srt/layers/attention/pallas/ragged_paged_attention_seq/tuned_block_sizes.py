# SPDX-License-Identifier: Apache-2.0
"""Tuned block sizes for stacked RPA-seq prefill."""

import logging

import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info

logger = logging.getLogger(__name__)

# Canonical global entries retained for the generic SEQ_ALONG_LANE backend.
# (device, page_size, dtypes, head_key, (max_model_len, None)) -> (bkv_p, bq)
GLOBAL_TUNED_BLOCK_SIZE_OVERRIDES = {
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (1024, None),
    ): (8, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (8192, None),
    ): (16, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (32768, None),
    ): (16, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (131072, None),
    ): (16, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (8192, None),
    ): (8, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (32768, None),
    ): (8, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-32_kv_head-4_head-128",
        (8192, None),
    ): (8, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-32_kv_head-4_head-128",
        (32768, None),
    ): (8, 16),
    **{
        (
            "TPU7x",
            512,
            "q_bfloat16_kv_float8_e4m3fn",
            head_key,
            (max_model_len, None),
        ): (1, 256)
        for head_key in (
            "q_head-8_kv_head-2_head-128",
            "q_head-16_kv_head-2_head-128",
        )
        for max_model_len in (8192, 32768, 131072)
    },
}

# Tuned for the current rank-4 cache kernel on TPU7x.
# (device, page_size, q_dtype, kv_dtype, q_heads, kv_heads, head_dim, q_tokens)
# -> (bkv_p, bq)
_QUERY_TOKEN_BUCKETS = (
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)
_TUNED_KV_DTYPES = ("bfloat16", "float8_e4m3fn")

TUNED_BLOCK_SIZES = {
    # For the 8-query-head, 1-KV-head, 64-wide shape, BF16 and FP8 8k/1k
    # sweeps agree on the dominant large-query tile. bq=256 is the
    # balanced full/sliding-attention choice: bq=512 is only 1-1.5% faster for
    # full attention and 0.5-1.2% slower at sw=2048.
    **{
        ("TPU7x", 512, "bfloat16", kv_dtype, 8, 1, 64, q_tokens): (
            1,
            min(q_tokens, 256),
        )
        for kv_dtype in _TUNED_KV_DTYPES
        for q_tokens in _QUERY_TOKEN_BUCKETS
    },
}

# The QK einsum flattens bq and the query heads sharing one KV head. The
# registered shapes target 2048 query rows per KV head. Preserve the default
# bkv_pages * bq area while rotating the tile toward Q; this avoids increasing
# per-iteration QK work.
#
# A profile supplies that row target rather than a final tile. This lets one
# measured model shape cover every query bucket while keeping the derivation
# reusable for future shapes. Same-chip 8k/1k sweeps validate the (2, 128)
# steady-state endpoint; smaller query buckets rotate back toward the default
# while preserving tile area: (8, 32), (4, 64), then (2, 128).
_HEURISTIC_QUERY_ROW_TARGETS = {
    # For 16 query heads per KV head at page size 512, BF16 and FP8 8k/1k and
    # 131k/1k full-attention sweeps both select (bkv_p=2, bq=128).
    **{
        ("TPU7x", 512, "bfloat16", kv_dtype, 32, 2, 128): 2048
        for kv_dtype in _TUNED_KV_DTYPES
    },
}


def _next_power_of_2(value):
    return 1 if value == 1 else 1 << (value - 1).bit_length()


def _previous_power_of_2(value):
    if value < 1:
        raise ValueError(f"Expected a positive value, got {value}")
    return 1 << (value.bit_length() - 1)


def _device_name():
    return tpu_info.get_device_kind()


def _default_block_sizes(page_size, device=None):
    device = _device_name() if device is None else device
    if device == "TPU7x":
        return 4096 // page_size, 32
    if device.startswith("TPU v4"):
        return 512 // page_size, 32
    if device.startswith("TPU"):
        return 2048 // page_size, 32
    raise NotImplementedError("RPA-seq tuning requires TPU v4 or newer")


def _shape_key(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
):
    return (
        _device_name(),
        page_size,
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
    )


def _heuristic_block_sizes(shape_key, query_bucket):
    """Rotate the default tile toward Q for registered high-GQA shapes."""
    target_query_rows = _HEURISTIC_QUERY_ROW_TARGETS.get(shape_key)
    if target_query_rows is None:
        return None

    device, page_size, _, _, num_q_heads, num_kv_heads, _ = shape_key
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    default_bkv_p, default_bq = _default_block_sizes(page_size, device=device)
    max_bq = _previous_power_of_2(target_query_rows // num_q_heads_per_kv_head)
    # Query buffers are power-of-two bucketed before this lookup. Increase BQ
    # with the bucket until the flattened query-row target is reached, and
    # decrease BKV by the same factor. This preserves the default QK tile area:
    #   8 pages * 32 queries == 4 * 64 == 2 * 128.
    bq = min(max(query_bucket, default_bq), max_bq)
    default_tile_area = default_bkv_p * default_bq
    bkv_p = max(1, default_tile_area // bq)
    return bkv_p, bq


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
):
    shape_key = _shape_key(
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size,
    )
    query_bucket = _next_power_of_2(max_num_tokens)
    key = (*shape_key, query_bucket)
    if key in TUNED_BLOCK_SIZES:
        bkv_p, bq = TUNED_BLOCK_SIZES[key]
    elif heuristic := _heuristic_block_sizes(shape_key, query_bucket):
        bkv_p, bq = heuristic
    else:
        buckets = [entry[-1] for entry in TUNED_BLOCK_SIZES if entry[:-1] == shape_key]
        if buckets:
            not_larger = [bucket for bucket in buckets if bucket <= query_bucket]
            fallback_bucket = max(not_larger) if not_larger else min(buckets)
            bkv_p, bq = TUNED_BLOCK_SIZES[(*shape_key, fallback_bucket)]
            logger.warning(
                "Couldn't find tuned RPA-seq prefill query bucket %s; "
                "falling back to %s",
                query_bucket,
                fallback_bucket,
            )
        else:
            bkv_p, bq = _default_block_sizes(page_size)
            logger.warning(
                "Couldn't find tuned RPA-seq prefill shape %s; using (%s, %s)",
                shape_key,
                bkv_p,
                bq,
            )
    return min(pages_per_seq, bkv_p), min(max_num_tokens, bq)


def resolve_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
    *,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
):
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        tuned_bkv_p, tuned_bq = get_tuned_block_sizes(
            q_dtype,
            kv_dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            head_dim,
            page_size,
            max_num_tokens,
            pages_per_seq,
        )
        if num_kv_pages_per_block is None:
            num_kv_pages_per_block = tuned_bkv_p
        if num_queries_per_block is None:
            num_queries_per_block = tuned_bq
    return num_kv_pages_per_block, num_queries_per_block
