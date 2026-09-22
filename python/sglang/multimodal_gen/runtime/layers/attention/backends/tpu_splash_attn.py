# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas Custom Splash Attention, Block-Size Autotuner, and Cross-Attention kernels.

Ported directly from DiffusionServing:
- DiffusionServing/runtime/kernels/custom_splash_attention.py (_flash_attention_kernel, hybrid fixed-m / online softmax)
- DiffusionServing/runtime/kernels/block_sizes.py (V7_SPEC, tile_grid_cost, autotune_block_sizes)
- DiffusionServing/runtime/kernels/fixed_m_bound.py (LOG2_E, FIXED_M_RECENTER, FIXED_M_SAFE_BOUND, fixed_m_operands)
- DiffusionServing/runtime/kernels/splash_attention.py (FullMask short-KV splash cross-attention)
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

LOG2_E = 1.44269504
FIXED_M_RECENTER = 88.0
FIXED_M_SAFE_BOUND = 213.0
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))

_splash_kernel_cache: Dict[tuple, Tuple[object, int, int]] = {}
_SPLASH_AVAILABLE: bool | None = None

# Calibrated overrides from DiffusionServing/runtime/kernels/block_sizes.py
_MEASURED_OVERRIDES: dict[tuple[int, int], dict[str, int]] = {
    (27900, 27900): {"block_q": 9472, "block_kv": 1024, "block_kv_compute": 1024},
    (55800, 55800): {"block_q": 5632, "block_kv": 2048, "block_kv_compute": 2048},
    (37800, 37800): {"block_q": 6400, "block_kv": 2048, "block_kv_compute": 2048},
    (18900, 18900): {"block_q": 9472, "block_kv": 1024, "block_kv_compute": 1024},
}


def _check_splash_available() -> bool:
    global _SPLASH_AVAILABLE
    if _SPLASH_AVAILABLE is not None:
        return _SPLASH_AVAILABLE
    try:
        import jax  # noqa: F401
        from jax.experimental import pallas as pl  # noqa: F401
        from jax.experimental.pallas import tpu as pltpu  # noqa: F401
        from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: F401
            splash_attention_kernel as splash,
            splash_attention_mask as sam,
        )
        from torch_tpu._internal.pallas.pallas import custom_jax_kernel  # noqa: F401

        _SPLASH_AVAILABLE = True
    except Exception as exc:
        logger.warning("TPU Pallas Splash Attention unavailable (%s); falling back to SDPA.", exc)
        _SPLASH_AVAILABLE = False
    return _SPLASH_AVAILABLE


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _round_down(x: int, multiple: int) -> int:
    return max((x // multiple) * multiple, multiple)


def _score_tile_bytes(
    block_q: int,
    block_kv_compute: int,
    head_dim: int = 128,
    block_kv: int = 2048,
    bytes_per_element: int = 2,
) -> int:
    score_f32 = block_kv_compute * block_q * 4
    qkv_2x = 2 * (block_q + 2 * block_kv) * head_dim * bytes_per_element
    output_f32 = block_q * head_dim * 4
    max_sum_scratch = 2 * 8 * block_q * 4
    return score_f32 + qkv_2x + output_f32 + max_sum_scratch


def autotune_block_sizes(
    q_seq: int,
    kv_seq: int,
    head_dim: int = 128,
) -> Tuple[int, int, int]:
    """Port of DiffusionServing/runtime/kernels/block_sizes.py for TPU v7x."""
    measured = _MEASURED_OVERRIDES.get((q_seq, kv_seq))
    if measured is not None:
        return measured["block_q"], measured["block_kv"], measured["block_kv_compute"]

    mxu_dim = 256
    safe_kv_block = 2048
    vmem_budget = int(64 * (2**20) * 0.93)

    if q_seq < mxu_dim or kv_seq < mxu_dim:
        return 256, 256, 256

    block_kv = _round_down(min(safe_kv_block, kv_seq), mxu_dim)
    block_kv_compute = block_kv

    vmem_max = mxu_dim
    bq = mxu_dim
    while bq <= q_seq:
        if _score_tile_bytes(bq, block_kv_compute, head_dim, block_kv) <= vmem_budget:
            vmem_max = bq
            bq += mxu_dim
        else:
            break

    candidates = [
        c for c in range(mxu_dim, vmem_max + 1, mxu_dim) if math.ceil(q_seq / c) >= 2
    ]
    if not candidates:
        candidates = [min(vmem_max, _round_down(q_seq, mxu_dim))]

    cell_overhead = mxu_dim * safe_kv_block

    def _cost(cand_bq: int) -> Tuple[int, int]:
        cells = math.ceil(q_seq / cand_bq) * math.ceil(kv_seq / block_kv)
        return (cells * (cand_bq * block_kv + cell_overhead), -cand_bq)

    best_bq = min(candidates, key=_cost)
    return best_bq, block_kv, block_kv_compute


def _build_custom_splash_kernel(
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    use_fixed_m: bool = True,
):
    """Build a custom Pallas kernel from DiffusionServing's custom_splash_attention.py."""
    import jax
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    import jax.numpy as jnp
    from torch_tpu._internal.pallas.pallas import custom_jax_kernel

    try:
        from jax._src.pallas.mosaic import tpu_info as _tpu_info

        if "cpu" not in _tpu_info.registry:
            _tpu_info.registry["cpu"] = lambda: (
                _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
                if hasattr(_tpu_info, "get_tpu_info_for_chip")
                else _tpu_info.TpuInfo(
                    chip_version=_tpu_info.ChipVersion.TPU_7X,
                    generation=7,
                    num_cores=1,
                    num_lanes=128,
                    num_sublanes=8,
                    mxu_column_size=256,
                    vmem_capacity_bytes=64 * 1024 * 1024,
                    cmem_capacity_bytes=0,
                    smem_capacity_bytes=1024 * 1024,
                    hbm_capacity_bytes=206_000_000_000 // 2,
                    mem_bw_bytes_per_second=int(7.40e12 // 2),
                    bf16_ops_per_second=int(2.31e15 // 2),
                    int8_ops_per_second=0,
                    fp8_ops_per_second=int(4.60e15 // 2),
                    int4_ops_per_second=0,
                    sparse_core=None,
                )
            )
    except Exception:
        pass

    block_q, block_kv, block_kv_compute = autotune_block_sizes(q_seq, kv_seq, head_dim)
    # Under torch_tpu, each logical device is 1 TensorCore (64 MiB VMEM).
    # Keep (block_kv_compute x block_q) float32 intermediates well within 64 MiB VMEM
    # so Mosaic/XLA compiles in <2s without LP scheduler stalls.
    if block_q > 3072:
        block_q = 3072 if (q_seq % 3072 == 0 or q_seq > 16384) else 2048
    if block_kv_compute > 512:
        block_kv_compute = 512
    q_padded = _round_up(q_seq, block_q)
    kv_padded = _round_up(kv_seq, block_kv)

    mask_value = -0.7 * float(np.finfo(np.dtype("float32")).max)
    vmem_limit_bytes = 64 * (2**20)  # 64 MiB VMEM per TensorCore on TPU v7x

    def _flash_attention_kernel(
        mk_ref,
        q_ref,
        k_ref,
        v_ref,
        *refs,
        grid_width: int,
        bkv: int,
        bkv_compute: int,
        head_dim_v: int,
        kv_seq_len: int,
    ):
        float32 = jnp.float32
        m_scratch_ref, l_scratch_ref, o_scratch_ref, (o_ref,) = (
            refs[0],
            refs[1],
            refs[2],
            refs[3:],
        )
        head_dim_v_repeats = head_dim_v // NUM_SUBLANES
        sv_dims = (((0,), (0,)), ((), ()))

        h = pl.program_id(0)
        j = pl.program_id(2)
        is_fixed = mk_ref[1, h] > 0.5 if use_fixed_m else False

        @pl.when(j == 0)
        def init():
            o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
            if use_fixed_m:

                @pl.when(is_fixed)
                def init_fixed():
                    m_scratch_ref[...] = jnp.full_like(
                        m_scratch_ref, mk_ref[0, h] - FIXED_M_RECENTER
                    )

                @pl.when(jnp.logical_not(is_fixed))
                def init_online():
                    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
            else:
                m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

        def _update(slice_k, fixed: bool):
            q_tile = q_ref[...]
            k_chunk = k_ref[slice_k, :]
            v_chunk = v_ref[slice_k, :]
            qk = lax.dot_general(
                k_chunk, q_tile, NT_DIM_NUMBERS, preferred_element_type=float32
            )
            if fixed:
                s_curr = jnp.exp2(qk - (mk_ref[0, h] - FIXED_M_RECENTER))
                s_cast = s_curr.astype(jnp.bfloat16)
                l_curr = s_curr.sum(axis=0, keepdims=True).astype(float32)
                o_curr = lax.dot_general(
                    v_chunk, s_cast, sv_dims, preferred_element_type=float32
                )
                o_scratch_ref[:] = o_scratch_ref[:] + o_curr
                l_scratch_ref[:] = l_scratch_ref[:] + l_curr
                return

            m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
            m_curr = qk.max(axis=0)[None, :]
            m_next = jnp.maximum(m_prev, m_curr)
            s_curr = jnp.exp2(qk - m_next[0:1])
            s_cast = s_curr.astype(jnp.bfloat16)
            l_curr = s_curr.sum(axis=0, keepdims=True)
            alpha = jnp.exp2(m_prev - m_next)
            o_curr = lax.dot_general(
                v_chunk, s_cast, sv_dims, preferred_element_type=float32
            )
            o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
            m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

        def _compute_body(fixed: bool):
            def compute_body(kv_compute_index, _):
                _update(pl.ds(kv_compute_index * bkv_compute, bkv_compute), fixed)

            return compute_body

        compute_online = _compute_body(False)

        def last_compute_body(kv_compute_index):
            slice_len = kv_seq_len % bkv_compute
            _update(pl.ds(kv_compute_index * bkv_compute, slice_len), False)

        def _run_main(body_fn):
            lax.fori_loop(0, bkv // bkv_compute, body_fn, None, unroll=True)

        def _run_last():
            if kv_seq_len % bkv == 0:
                lax.fori_loop(0, bkv // bkv_compute, compute_online, None, unroll=True)
            else:
                remain = kv_seq_len % bkv
                iters = (remain + bkv_compute - 1) // bkv_compute
                if remain % bkv_compute == 0:
                    lax.fori_loop(0, iters, compute_online, None, unroll=True)
                else:
                    lax.fori_loop(0, iters - 1, compute_online, None, unroll=True)
                    last_compute_body(iters - 1)

        if use_fixed_m:
            compute_fixed = _compute_body(True)

            @pl.when((j != grid_width - 1) & is_fixed)
            def body_fixed():
                _run_main(compute_fixed)

            @pl.when((j != grid_width - 1) & jnp.logical_not(is_fixed))
            def body_online():
                _run_main(compute_online)
        else:

            @pl.when(j != grid_width - 1)
            def body():
                _run_main(compute_online)

        @pl.when(j == grid_width - 1)
        def last_body():
            _run_last()

        @pl.when(j == grid_width - 1)
        def end():
            l = l_scratch_ref[...]
            l_inv = (
                pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=0)
                if hasattr(pltpu, "repeat")
                else jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
            )
            o_ref[...] = (o_scratch_ref[...] * l_inv[: o_scratch_ref.shape[0], :]).astype(
                o_ref.dtype
            )

    def _jax_fn(query, key, value):
        # k-smoothing over unpadded keys: zero-mean K rows guarantee rowmax >= 0
        k_mean = jnp.mean(key[:, :, :kv_seq, :], axis=2, keepdims=True)
        key = key - k_mean
        # query arrives pre-scaled by softmax scale; multiply by LOG2_E for exp2 softmax
        q_log2 = (query.astype(jnp.float32) * LOG2_E).astype(jnp.bfloat16)

        qn_max = jnp.sqrt((q_log2[:, :, :q_seq, :].astype(jnp.float32) ** 2).sum(axis=-1)).max(
            axis=(0, 2)
        )
        mk_h = jnp.sqrt((key[:, :, :kv_seq, :].astype(jnp.float32) ** 2).sum(axis=-1)).max(
            axis=(0, 2)
        )
        bound = qn_max * mk_h
        mk_arr = jnp.stack(
            [jnp.ceil(bound), (bound <= FIXED_M_SAFE_BOUND).astype(jnp.float32)]
        )

        def _kernel_3d(q_3d, k_3d, v_3d):
            num_q_heads = q_3d.shape[0]
            num_kv_heads = k_3d.shape[0]
            q_heads_per_kv_head = num_q_heads // num_kv_heads
            grid_width = (kv_seq + block_kv - 1) // block_kv
            grid_height = q_padded // block_q

            in_specs = [
                pl.BlockSpec((None, block_q, head_dim), lambda h, i, j, *_: (h, i, 0)),
                pl.BlockSpec(
                    (None, block_kv, head_dim),
                    lambda h, i, j, *_: (h // q_heads_per_kv_head, j, 0),
                ),
                pl.BlockSpec(
                    (None, block_kv, head_dim),
                    lambda h, i, j, *_: (h // q_heads_per_kv_head, j, 0),
                ),
            ]
            out_shapes = [
                jax.ShapeDtypeStruct((NUM_SUBLANES, block_q), jnp.float32),
                jax.ShapeDtypeStruct((NUM_SUBLANES, block_q), jnp.float32),
                jax.ShapeDtypeStruct((head_dim, block_q), jnp.float32),
                jax.ShapeDtypeStruct((num_q_heads, head_dim, q_padded), q_3d.dtype),
            ]
            out_specs = [
                pl.BlockSpec((NUM_SUBLANES, block_q), lambda *_: (0, 0)),
                pl.BlockSpec((NUM_SUBLANES, block_q), lambda *_: (0, 0)),
                pl.BlockSpec((head_dim, block_q), lambda *_: (0, 0)),
                pl.BlockSpec((None, head_dim, block_q), lambda h, i, j, *_: (h, 0, i)),
            ]
            all_out = pl.pallas_call(
                functools.partial(
                    _flash_attention_kernel,
                    grid_width=grid_width,
                    bkv=block_kv,
                    bkv_compute=block_kv_compute,
                    head_dim_v=head_dim,
                    kv_seq_len=kv_seq,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=(num_q_heads, grid_height, grid_width),
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel", "arbitrary", "arbitrary"),
                    flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
                    vmem_limit_bytes=vmem_limit_bytes,
                ),
                out_shape=out_shapes,
            )(mk_arr, q_3d, k_3d, v_3d)
            return jnp.swapaxes(all_out[3], 1, 2)

        return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(q_log2, key, value)

    wrapped = custom_jax_kernel(_jax_fn, warn_deprecated=False)
    return wrapped, q_padded, kv_padded


def _build_fullmask_splash_kernel(
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    block_q: int,
    block_kv: int,
    is_causal: bool = False,
):
    """Build Pallas make_splash_mha kernel using FullMask whenever KV is unpadded."""
    import jax
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as sam,
    )
    from torch_tpu._internal.pallas.pallas import custom_jax_kernel

    q_padded = _round_up(q_seq, block_q)
    kv_padded = _round_up(kv_seq, block_kv)

    if is_causal:
        if q_padded == q_seq and kv_padded == kv_seq:
            mask = sam.CausalMask((q_padded, kv_padded))
        else:
            mask_arr = np.zeros((q_padded, kv_padded), dtype=np.bool_)
            mask_arr[:q_seq, :kv_seq] = np.tril(np.ones((q_seq, kv_seq), dtype=np.bool_))
            mask = sam.NumpyMask(mask_arr)
    elif not is_causal:
        # Always use FullMask((q_padded, kv_padded)) to avoid allocating multi-GB
        # dense NumpyMask arrays in CPU/HBM for long video sequences (e.g. S=75,600).
        mask = sam.FullMask((q_padded, kv_padded))

    block_sizes = splash.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=min(block_kv, 512),
    )

    def splash_fn(q, k, v):
        num_heads = q.shape[0]
        multi_head_mask = sam.MultiHeadMask([mask] * num_heads)
        kernel = splash.make_splash_mha(
            multi_head_mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_sizes,
        )
        return kernel(q, k, v)

    wrapped = custom_jax_kernel(jax.vmap(splash_fn), warn_deprecated=False)
    return wrapped, q_padded, kv_padded


def _get_or_create_splash_kernel(
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
):
    """Select and cache the optimal DiffusionServing Pallas kernel for (q_seq, kv_seq)."""
    cache_key = (q_seq, kv_seq, head_dim, dtype, is_causal)
    cached = _splash_kernel_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Long-sequence video/image self-attention (e.g. Wan2.2 14B S=75600, LTX-2 S=14080):
    # Use make_splash_mha with FullMask((q_padded, kv_padded)) and 2048x2048 (kv_compute=512)
    # blocks so VMEM stays under 16 MiB per TensorCore and XLA compiles in <2s with 0 NumpyMask HBM.
    if not is_causal and q_seq >= 2048 and kv_seq >= 2048:
        result = _build_splash_fn(
            q_seq, kv_seq, is_causal=False, block_q_override=2048, block_kv_override=2048
        )
        _splash_kernel_cache[cache_key] = result
        return result

    # 2. Asymmetric cross-attention with long Q and aligned short KV (e.g. Wan2.2 / LTX-2 text cross-attn):
    # Use DiffusionServing's splash_attention.py (FullMask with unpadded KV and block_q=2048).
    if not is_causal and q_seq >= 2048 and 128 <= kv_seq <= 2048 and kv_seq % 128 == 0:
        block_q = 2048 if q_seq >= 2048 else 1024
        block_kv = min(1024, kv_seq)
        result = _build_fullmask_splash_kernel(
            q_seq, kv_seq, head_dim, block_q=block_q, block_kv=block_kv, is_causal=False
        )
        _splash_kernel_cache[cache_key] = result
        return result

    # 3. Medium-sequence self-attention (e.g. FLUX.2 S=4608, Z-Image S=4128):
    if q_seq >= 1024 and kv_seq >= 1024 and q_seq % 1024 == 0 and kv_seq % 1024 == 0:
        block_q, block_kv = 1024, 1024
    elif q_seq >= 512 and kv_seq >= 512:
        block_q, block_kv = 512, 512
    else:
        block_q, block_kv = 256, 256

    result = _build_fullmask_splash_kernel(
        q_seq, kv_seq, head_dim, block_q=block_q, block_kv=block_kv, is_causal=is_causal
    )
    _splash_kernel_cache[cache_key] = result
    return result


@torch.library.custom_op(
    "sglang::tpu_splash_attention", mutates_args=(), device_types=["tpu"]
)
def _tpu_splash_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    q_seq, head_dim = q.shape[2], q.shape[3]
    kv_seq = k.shape[2]

    kernel, q_padded, kv_padded = _get_or_create_splash_kernel(
        q_seq, kv_seq, head_dim, q.dtype, is_causal
    )

    need_pad_q = q_padded > q_seq
    need_pad_kv = kv_padded > kv_seq
    if need_pad_q:
        q = F.pad(q, (0, 0, 0, q_padded - q_seq))
    if need_pad_kv:
        k = F.pad(k, (0, 0, 0, kv_padded - kv_seq))
        v = F.pad(v, (0, 0, 0, kv_padded - kv_seq))

    out = kernel(q.contiguous(), k.contiguous(), v.contiguous())

    if need_pad_q:
        out = out[:, :, :q_seq, :]
    return out


@_tpu_splash_attention_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    return torch.empty_like(q)


def tpu_fast_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fast XLA-fused dot-product attention for short Q and KV sequences."""
    key_scaled = key * scale
    scores = torch.matmul(query, key_scaled.transpose(-2, -1))
    attn_weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
    return torch.matmul(attn_weights, value)


def tpu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Dispatch [B, H, S, D] attention on TPU to DiffusionServing Pallas kernels."""
    head_dim = query.shape[-1]
    if scale is None:
        scale = head_dim**-0.5

    if query.shape[1] != key.shape[1]:
        repeat_factor = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    q_seq = query.shape[2]
    kv_seq = key.shape[2]

    # Short sequences or unaligned short KV cross-attention
    if (not is_causal) and (
        (q_seq <= 512 and kv_seq <= 512)
        or (q_seq != kv_seq and kv_seq <= 1024 and (q_seq < 2048 or kv_seq % 128 != 0))
    ):
        return tpu_fast_cross_attention(query, key, value, scale=scale)

    # Full-sequence self-attention & aligned long-Q cross-attention via DiffusionServing Pallas kernels
    if (
        head_dim == 128
        and query.dtype in (torch.bfloat16, torch.float32)
        and _check_splash_available()
    ):
        q_scaled = query * scale
        return _tpu_splash_attention_op(q_scaled, key, value, is_causal=is_causal)

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )
