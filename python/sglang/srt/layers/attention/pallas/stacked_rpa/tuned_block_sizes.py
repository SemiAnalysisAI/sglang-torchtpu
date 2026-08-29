"""Auto-tuned block sizes for standalone stacked-RPA decode."""

import jax
import jax.numpy as jnp

from sglang.srt.layers.attention.pallas.stacked_rpa import configs


def next_power_of_2(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


_BATCHED_TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)


def bucket_batched_tokens(total_q_tokens: int) -> int:
    """Snap the static padded query-token count to a tuning bucket."""
    n = max(int(total_q_tokens), 1)
    for bucket in _BATCHED_TOKEN_BUCKETS:
        if n <= bucket:
            return bucket
    return _BATCHED_TOKEN_BUCKETS[-1]


def get_device_name() -> str:
    return jax.devices()[0].device_kind


def compute_head_key(num_q_heads: int, num_kv_heads: int, head_dim: int) -> str:
    return (
        f"q{next_power_of_2(num_q_heads)}"
        f"_kv{next_power_of_2(num_kv_heads)}"
        f"_d{head_dim}"
    )


def compute_dtype_key(q_dtype, kv_dtype) -> str:
    return f"q_{jnp.dtype(q_dtype).name}_kv_{jnp.dtype(kv_dtype).name}"


def make_lookup_key(
    *,
    device_name: str,
    page_size: int,
    q_dtype,
    kv_dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_model_len: int,
    mode_symbol: str,
    batched_tokens_bucket: int | None = None,
    sliding_window: int | None = None,
) -> tuple:
    key = (
        device_name,
        next_power_of_2(page_size),
        compute_dtype_key(q_dtype, kv_dtype),
        compute_head_key(num_q_heads, num_kv_heads, head_dim),
        next_power_of_2(max_model_len),
        mode_symbol,
    )
    if batched_tokens_bucket is None:
        return key
    window_key = "none" if sliding_window is None else int(sliding_window)
    return (*key, int(batched_tokens_bucket), window_key)


_TPU7X_FP8 = "q_bfloat16_kv_float8_e4m3fn"

TUNED_BLOCK_SIZES: dict[tuple, tuple[int, int, int, int, int]] = {
    **{
        ("TPU7x", 128, _TPU7X_FP8, head_key, max_model_len, "d"): (
            1,
            1,
            4096,
            8,
            2,
        )
        for head_key in ("q32_kv4_d64", "q16_kv2_d64", "q8_kv1_d64")
        for max_model_len in (32768, 131072)
    },
    **{
        ("TPU7x", 512, _TPU7X_FP8, "q8_kv1_d64", max_model_len, "d"): (
            1,
            1,
            min(max_model_len, 8192),
            8,
            2,
        )
        for max_model_len in (
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
        )
    },
}


def get_tuned_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    max_model_len: int,
    mode: configs.RpaCase,
) -> tuple[int, int, int, int, int] | None:
    """Return a tuned decode block-size tuple, if one matches."""
    if mode != configs.RpaCase.DECODE:
        return None
    base = dict(
        device_name=get_device_name(),
        page_size=serve_cfgs.page_size,
        q_dtype=serve_cfgs.dtype_q,
        kv_dtype=serve_cfgs.dtype_kv,
        num_q_heads=model_cfgs.num_q_heads,
        num_kv_heads=model_cfgs.num_kv_heads,
        head_dim=model_cfgs.head_dim,
        max_model_len=max_model_len,
        mode_symbol=mode.symbol,
    )
    bucket = bucket_batched_tokens(serve_cfgs.total_q_tokens)
    window = model_cfgs.sliding_window

    entry = TUNED_BLOCK_SIZES.get(
        make_lookup_key(
            **base,
            batched_tokens_bucket=bucket,
            sliding_window=window,
        )
    )
    if entry is None and window is not None:
        entry = TUNED_BLOCK_SIZES.get(
            make_lookup_key(
                **base,
                batched_tokens_bucket=bucket,
                sliding_window=None,
            )
        )
    if entry is None:
        entry = TUNED_BLOCK_SIZES.get(make_lookup_key(**base))
    return entry
