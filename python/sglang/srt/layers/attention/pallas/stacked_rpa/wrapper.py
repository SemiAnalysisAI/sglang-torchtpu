"""SGLang wrapper for the standalone stacked-RPA decode kernel."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.layers.attention.pallas.stacked_rpa import configs, utils
from sglang.srt.layers.attention.pallas.stacked_rpa.decode import (
    block_sizes as decode_blocks,
)
from sglang.srt.layers.attention.pallas.stacked_rpa.decode import (
    config as decode_config,
)
from sglang.srt.layers.attention.pallas.stacked_rpa.decode.non_sliding_window import (
    kernel as non_sliding_window_kernel,
)
from sglang.srt.layers.attention.pallas.stacked_rpa.decode.non_sliding_window import (
    schedule as non_sliding_window_schedule,
)
from sglang.srt.layers.attention.pallas.stacked_rpa.decode.sliding_window import (
    kernel as sliding_window_kernel,
)
from sglang.srt.layers.attention.pallas.stacked_rpa.decode.sliding_window import (
    schedule as sliding_window_schedule,
)


def prepare_queries(
    q: jax.Array,
    num_kv_heads: int,
    q_dtype: jnp.dtype,
) -> jax.Array:
    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    num_q_heads_per_kv_head = actual_num_q_heads // num_kv_heads
    q_packing = utils.get_dtype_packing(q_dtype)
    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head, q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_q_head_dim = utils.align_to(actual_head_dim, num_lanes)

    return (
        jnp.pad(
            q.reshape(
                total_q_tokens,
                num_kv_heads,
                num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
                (0, aligned_q_head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            total_q_tokens,
            num_kv_heads,
            aligned_num_q_heads_per_kv_head // q_packing,
            q_packing,
            aligned_q_head_dim,
        )
        .swapaxes(0, 1)
    )


def prepare_seq_along_lane_new_kv_hbm(
    k: jax.Array,
    v: jax.Array,
    *,
    kv_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Pack post-RoPE K/V into stacked decode's sequence-minor layout."""
    total_q_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    if v.shape != k.shape:
        raise ValueError(f"k/v shapes differ: {k.shape=} {v.shape=}")
    if kv_dtype is None:
        kv_dtype = k.dtype
    kv_packing = utils.get_dtype_packing(kv_dtype)
    tpu_info = pltpu.get_tpu_info()
    aligned_kv_head_dim = utils.align_to(
        actual_head_dim, tpu_info.num_sublanes * kv_packing
    )
    padded_total_tokens = utils.align_to(total_q_tokens, tpu_info.num_lanes)
    return jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(
            total_q_tokens, actual_num_kv_heads * 2, actual_head_dim
        ),
        (
            (0, padded_total_tokens - total_q_tokens),
            (0, 0),
            (0, aligned_kv_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).transpose(1, 2, 0)


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    return (
        prepare_queries(q, k.shape[1], q.dtype),
        prepare_seq_along_lane_new_kv_hbm(k, v, kv_dtype=kv_dtype),
    )


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, head_dim = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, head_dim)


def _resolve_decode_config(
    queries: jax.Array,
    keys: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    decode_block_sizes: decode_blocks.BlockSizes | None,
    decode_q_len: int,
    decode_bkv_override: int,
) -> decode_config.DecodeConfig:
    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    model = configs.ModelConfigs(
        num_q_heads=queries.shape[1],
        num_kv_heads=keys.shape[1],
        head_dim=queries.shape[2],
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )
    serve = configs.ServingConfigs(
        num_seqs=kv_lens.shape[0],
        num_page_indices=page_indices.shape[0],
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=kv_cache.shape[3],
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
    )
    if sliding_window is None:
        block = decode_blocks.choose_non_sliding_window_block_sizes(
            model,
            serve,
            vmem_limit_bytes,
            decode_q_len,
            override=decode_block_sizes,
            bkv_override=decode_bkv_override,
        )
    else:
        block = decode_blocks.choose_sliding_window_block_sizes(
            model,
            serve,
            vmem_limit_bytes,
            decode_q_len,
            override=decode_block_sizes,
        )
    return decode_config.make_config(
        model,
        serve,
        block,
        vmem_limit_bytes,
        update_kv_cache=True,
        decode_q_len=decode_q_len,
    )


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "decode_q_len",
        "decode_bkv_override",
        "decode_block_sizes",
        "vmem_limit_bytes",
        "out_dtype",
        "use_causal_mask",
        "update_kv_cache",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_q_len: int = 1,
    decode_bkv_override: int = 0,
    decode_block_sizes: decode_blocks.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Run stacked RPA decode against its native rank-4 KV cache."""
    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if not update_kv_cache:
        raise ValueError("Stacked decode requires update_kv_cache=True.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")
    if out_dtype is not None and jnp.dtype(out_dtype) != queries.dtype:
        raise ValueError("Stacked decode output dtype must match the query dtype.")

    if kv_cache.ndim != 4:
        raise ValueError(
            "stacked decode requires its native rank-4 KV cache, "
            f"got shape={kv_cache.shape}."
        )
    cfgs = _resolve_decode_config(
        queries,
        keys,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        decode_block_sizes=decode_block_sizes,
        decode_q_len=decode_q_len,
        decode_bkv_override=decode_bkv_override,
    )
    cfgs.validate_inputs(
        q=queries,
        k=keys,
        v=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )
    q_hbm, new_kv_hbm = prepare_inputs(queries, keys, values, kv_cache.dtype)

    def run_decode(args):
        output_hbm, current_cache = args
        if cfgs.model.sliding_window is None:
            schedule_hbm = non_sliding_window_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=cfgs,
            )
            return non_sliding_window_kernel.rpa_kernel(
                cu_q_lens,
                kv_lens,
                schedule_hbm,
                output_hbm,
                new_kv_hbm,
                current_cache,
                cfgs=cfgs,
            )
        schedule_hbm = sliding_window_schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices,
            cfgs=cfgs,
        )
        return sliding_window_kernel.rpa_kernel(
            schedule_hbm,
            output_hbm,
            new_kv_hbm,
            current_cache,
            cfgs=cfgs,
        )

    output_hbm, kv_cache = jax.lax.cond(
        distribution[0] > 0,
        run_decode,
        lambda args: args,
        (q_hbm, kv_cache),
    )
    output = prepare_outputs(output_hbm)
    q_heads_per_kv_head = queries.shape[1] // keys.shape[1]
    output = output[:, :, :q_heads_per_kv_head, : queries.shape[2]]
    output = output.swapaxes(1, 0).reshape(queries.shape)
    return output, kv_cache
