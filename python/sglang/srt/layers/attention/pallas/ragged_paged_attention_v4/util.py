"""Utility functions for ragged paged attention."""

import enum
import logging

import jax
import jax.numpy as jnp

try:
    from tpu_inference.logger import init_logger

    logger = init_logger(__name__)
except:
    logger = logging.getLogger(__name__)


class RPAKind(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    """Get the bit width of a dtype, compatible with different JAX versions."""
    # Try the newer JAX API first
    try:
        from jax._src import dtypes

        if hasattr(dtypes, "bit_width"):
            return dtypes.bit_width(dtype)
    except (ImportError, AttributeError):
        pass

    # Fallback: compute bit width from itemsize
    dtype = jnp.dtype(dtype)
    return dtype.itemsize * 8


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if kind.endswith("p") or kind.endswith("e"):
        kind = kind[:-1]
    if kind == "TPU7x":
        return 7
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


# ============================================================================
# Tuned block sizes infrastructure
# ============================================================================

# Device name mapping: JAX device kind -> TUNED_BLOCK_SIZES key
# This maps device names reported by JAX to the canonical names used in TUNED_BLOCK_SIZES
DEVICE_NAME_MAPPING = {
    "TPU v6 lite": "TPU v6e",
}


def get_device_name() -> str:
    """Get the device name, mapped to the canonical name used in TUNED_BLOCK_SIZES."""
    device_kind = jax.devices()[0].device_kind
    return DEVICE_NAME_MAPPING.get(device_kind, device_kind)


BLOCK_SIZE_CACHE = {}


def get_default_block_sizes(tpu_version: int, page_size: int) -> tuple[int, int]:
    """Get default block sizes for a given TPU version."""
    if tpu_version < 4:
        raise NotImplementedError("TPU version must be 4 or higher.")
    match tpu_version:
        case 4:
            # TPUv4 has much smaller VMEM size so we pick fixed block sizes.
            return (512 // page_size, 32)
        case 7:
            return (4096 // page_size, 32)
        case _:
            return (2048 // page_size, 32)


def get_simplified_raw_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
    multiply_num_kv_heads=True,
    align_head_dim=True,
):
    """Get the simplified key."""
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)

    if multiply_num_kv_heads:
        num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
        num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
        assert num_kv_heads_x2 % 2 == 0
        num_kv_heads_result = next_power_of_2(num_kv_heads_x2) // 2
    else:
        num_kv_heads_result = next_power_of_2(align_to(actual_num_kv_heads, kv_packing))
        num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)

    if align_head_dim:
        head_dim_result = align_to(head_dim, 128)
    else:
        head_dim_result = head_dim

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        num_kv_heads_result,
        head_dim_result,
        next_power_of_2(max_model_len),
    )


def get_lookup_keys(
    page_size,
    q_dtype,
    kv_dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_model_len,
    multiply_num_kv_heads=True,
    align_head_dim=True,
):
    """Get the lookup keys for tuned block sizes."""
    (
        page_size,
        q_dtype_name,
        kv_dtype_name,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    ) = get_simplified_raw_key(
        page_size,
        q_dtype,
        kv_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
        multiply_num_kv_heads=multiply_num_kv_heads,
        align_head_dim=align_head_dim,
    )

    return (
        get_device_name(),
        next_power_of_2(page_size),
        f"q_{q_dtype_name}_kv_{kv_dtype_name}",
        f"q_head-{num_q_heads}_kv_head-{num_kv_heads}_head-{head_dim}",
        next_power_of_2(max_model_len),
    )


def get_block_sizes_with_max_model_len_key_fallback(
    d: dict,
    max_model_len: tuple,
    fallback_max_model_len: tuple | None = None,
):
    if max_model_len in d:
        return d[max_model_len]
    available_keys = tuple(d.keys())

    # Window-preserving fallback: when a sliding-window (local-attention) lookup
    # (key = (context_len, window) with window != None) has no exact entry -- e.g.
    # a context length longer than anything tuned -- prefer another entry tuned
    # for the SAME window over collapsing to an unwindowed (full-attention)
    # config. Without this, long-context local layers silently reuse the
    # full-attention block sizes and lose the sliding-window speedup. Only applies
    # to windowed lookups; unwindowed (full) lookups keep the original behavior.
    requested_window = max_model_len[1] if isinstance(max_model_len, tuple) else None
    if requested_window is not None:
        same_window_keys = [
            k
            for k in d
            if isinstance(k, tuple) and len(k) == 2 and k[1] == requested_window
        ]
        if same_window_keys:
            requested_len = max_model_len[0]
            not_longer = [k for k in same_window_keys if k[0] <= requested_len]
            chosen_key = (
                max(not_longer, key=lambda k: k[0])
                if not_longer
                else min(same_window_keys, key=lambda k: k[0])
            )
            logger.info(
                "Couldn't find tuned sizes for RPA kernel for "
                f"{max_model_len}; falling back to same-window key {chosen_key}. "
                f"Available keys: {available_keys}"
            )
            return d[chosen_key]

    if fallback_max_model_len is not None and fallback_max_model_len in d:
        logger.info(
            "Couldn't find tuned sizes for RPA kernel for "
            f"{max_model_len}; falling back to {fallback_max_model_len}. "
            f"Available keys: {available_keys}"
        )
        return d[fallback_max_model_len]

    fallback_key = sorted(d.keys(), key=lambda x: d[x], reverse=True)[0]
    logger.warning(
        "Couldn't find tuned sizes for RPA kernel for "
        f"{max_model_len}; falling back to {fallback_key}. "
        f"Available keys: {available_keys}"
    )
    return d[fallback_key]


def get_max_model_len_key(max_model_len: int, use_sliding_window: bool, sliding_window):
    # Unified tuple key: (max_model_len, sliding_window). When the kernel is not
    # sliding-window-aware, the window is recorded as None (the non-windowed entry).
    return (max_model_len, sliding_window if use_sliding_window else None)


def get_tuned_block_sizes_impl(
    tuned_block_sizes_data,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
    chunk_prefill_size,
    use_sliding_window=False,
    sliding_window=None,
    use_sliding_window_decode: bool | None = None,
    multiply_num_kv_heads=True,
    align_head_dim=True,
):
    """Implementation of get_tuned_block_sizes that works with different data formats."""
    tpu_version = get_tpu_version()
    bkv_p, bq = get_default_block_sizes(tpu_version, page_size)

    keys = get_lookup_keys(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size * pages_per_seq,
        multiply_num_kv_heads=multiply_num_kv_heads,
        align_head_dim=align_head_dim,
    )
    device, page_size, dtypes, head_dims, max_model_len = keys

    if use_sliding_window_decode is None:
        use_sliding_window_decode = use_sliding_window

    max_model_len_key = get_max_model_len_key(
        max_model_len, use_sliding_window, sliding_window
    )
    decode_max_model_len_key = get_max_model_len_key(
        max_model_len, use_sliding_window_decode, sliding_window
    )

    cache_key = (
        device,
        page_size,
        dtypes,
        head_dims,
        decode_max_model_len_key,
        max_model_len_key,
        chunk_prefill_size,
    )

    logger.info(f"RPA tuned block sizes lookup: {cache_key}")

    if cache_key in BLOCK_SIZE_CACHE:
        return BLOCK_SIZE_CACHE[cache_key]

    try:
        bkv_p_decode, bq_decode = get_block_sizes_with_max_model_len_key_fallback(
            tuned_block_sizes_data[RPAKind.DECODE][device][page_size][dtypes][
                head_dims
            ],
            decode_max_model_len_key,
            fallback_max_model_len=(max_model_len, None),
        )
        bkv_p_prefill, bq_prefill = get_block_sizes_with_max_model_len_key_fallback(
            tuned_block_sizes_data[RPAKind.PREFILL][chunk_prefill_size][device][
                page_size
            ][dtypes][head_dims],
            max_model_len_key,
        )
        bkv_p_mixed, bq_mixed = get_block_sizes_with_max_model_len_key_fallback(
            tuned_block_sizes_data[RPAKind.MIXED][device][page_size][dtypes][head_dims],
            max_model_len_key,
        )
    except KeyError:
        logger.warning(f"Couldn't find tuned sizes for RPA kernel with {cache_key}")
        bkv_p_decode = bkv_p_prefill = bkv_p_mixed = bkv_p
        bq_decode = bq_prefill = bq_mixed = bq

    BLOCK_SIZE_CACHE[cache_key] = (
        min(pages_per_seq, bkv_p_decode),
        min(max_num_tokens, bq_decode),
        min(pages_per_seq, bkv_p_prefill),
        min(max_num_tokens, bq_prefill),
        min(pages_per_seq, bkv_p_mixed),
        min(max_num_tokens, bq_mixed),
    )
    return BLOCK_SIZE_CACHE[cache_key]
