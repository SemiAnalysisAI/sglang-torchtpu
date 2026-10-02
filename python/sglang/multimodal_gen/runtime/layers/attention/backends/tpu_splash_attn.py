# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas attention backend (``--attention-backend tpu_splash_attn``).

Kernels ported from DiffusionServing:
- runtime/kernels/custom_splash_attention.py: flash attention, online exp2 softmax
- runtime/kernels/block_sizes.py: v7x block-size autotuner
- runtime/kernels/splash_attention.py: make_splash_mha for short-KV cross-attention
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    SDPABackend,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export

logger = init_logger(__name__)

LOG2_E = 1.44269504
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))

_splash_kernel_cache: Dict[tuple, Tuple[object, int, int, bool]] = {}
_logged_sdpa_fallbacks: set[tuple] = set()
_SPLASH_AVAILABLE: bool | None = None

# Calibrated overrides from DiffusionServing/runtime/kernels/block_sizes.py
_MEASURED_OVERRIDES: dict[tuple[int, int], dict[str, int]] = {
    (27900, 27900): {"block_q": 9472, "block_kv": 1024, "block_kv_compute": 1024},
    (55800, 55800): {"block_q": 5632, "block_kv": 2048, "block_kv_compute": 2048},
    (37800, 37800): {"block_q": 6400, "block_kv": 2048, "block_kv_compute": 2048},
    (18900, 18900): {"block_q": 9472, "block_kv": 1024, "block_kv_compute": 1024},
}


def _splash_vmem_limit_bytes(pltpu) -> int:
    """Mosaic scoped-VMEM ceiling per pallas_call: 2x physical core VMEM.

    Called at trace time under ``tpu_info_for_cpu_export``, so ``get_tpu_info``
    reports the physical chip.
    """
    try:
        return 2 * int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # CPU-only tracing, or an older pltpu without get_tpu_info
        return 128 * (2**20)  # 2 x 64 MiB, the v7x value


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
        )
        from torch_tpu._internal.pallas.pallas import custom_jax_kernel  # noqa: F401

        _SPLASH_AVAILABLE = True
    except Exception as exc:
        logger.warning(
            "TPU Pallas Splash Attention unavailable (%s); falling back to SDPA.", exc
        )
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
    scale: float,
):
    """Flash attention with an online exp2 softmax.

    Takes the unscaled query; the softmax scale is applied to the float32
    scores. Returns ``(kernel, q_seq, kv_seq)``: inputs are not padded.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal.pallas.pallas import custom_jax_kernel

    block_q, block_kv, block_kv_compute = autotune_block_sizes(q_seq, kv_seq, head_dim)
    qk_scale = scale * LOG2_E

    mask_value = -0.7 * float(np.finfo(np.dtype("float32")).max)

    def _flash_attention_kernel(
        q_ref,
        k_ref,
        v_ref,
        o_ref,
        m_scratch_ref,
        l_scratch_ref,
        o_scratch_ref,
        *,
        grid_width: int,
        bkv: int,
        bkv_compute: int,
        head_dim_v: int,
        kv_seq_len: int,
    ):
        float32 = jnp.float32
        head_dim_v_repeats = head_dim_v // NUM_SUBLANES
        sv_dims = (((0,), (0,)), ((), ()))
        j = pl.program_id(2)

        @pl.when(j == 0)
        def init():
            o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

        def _update(slice_k):
            q_tile = q_ref[...]
            k_chunk = k_ref[slice_k, :]
            v_chunk = v_ref[slice_k, :]
            qk = lax.dot_general(
                k_chunk, q_tile, NT_DIM_NUMBERS, preferred_element_type=float32
            )
            m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
            # Max of raw scores (qk_scale > 0): qk * c - m is one FMA.
            m_curr = qk.max(axis=0)[None, :] * qk_scale
            m_next = jnp.maximum(m_prev, m_curr)
            s_curr = jnp.exp2(qk * qk_scale - m_next[0:1])
            s_cast = s_curr.astype(jnp.bfloat16)
            l_curr = s_curr.sum(axis=0, keepdims=True)
            alpha = jnp.exp2(m_prev - m_next)
            o_curr = lax.dot_general(
                v_chunk, s_cast, sv_dims, preferred_element_type=float32
            )
            o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
            m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

        def compute_body(kv_compute_index, _):
            _update(pl.ds(kv_compute_index * bkv_compute, bkv_compute))

        @pl.when(j != grid_width - 1)
        def body():
            lax.fori_loop(0, bkv // bkv_compute, compute_body, None, unroll=True)

        # The last KV block reads only real keys.
        @pl.when(j == grid_width - 1)
        def last_body():
            remain = kv_seq_len - (grid_width - 1) * bkv
            full, tail = divmod(remain, bkv_compute)
            lax.fori_loop(0, full, compute_body, None, unroll=True)
            if tail:
                _update(pl.ds(full * bkv_compute, tail))

        @pl.when(j == grid_width - 1)
        def end():
            l = l_scratch_ref[...]
            l_inv = (
                pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=0)
                if hasattr(pltpu, "repeat")
                else jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
            )
            # Scratch is [D, block_q]; write the [block_q, D] output tile.
            o_ref[...] = (
                o_scratch_ref[...] * l_inv[: o_scratch_ref.shape[0], :]
            ).T.astype(o_ref.dtype)

    def _jax_fn(query, key, value):
        def _kernel_3d(q_3d, k_3d, v_3d):
            num_q_heads = q_3d.shape[0]
            num_kv_heads = k_3d.shape[0]
            q_heads_per_kv_head = num_q_heads // num_kv_heads
            # Mosaic masks partial edge blocks; no physical padding.
            grid_width = pl.cdiv(kv_seq, block_kv)
            grid_height = pl.cdiv(q_seq, block_q)

            in_specs = [
                pl.BlockSpec((None, block_q, head_dim), lambda h, i, j: (h, i, 0)),
                pl.BlockSpec(
                    (None, block_kv, head_dim),
                    lambda h, i, j: (h // q_heads_per_kv_head, j, 0),
                ),
                pl.BlockSpec(
                    (None, block_kv, head_dim),
                    lambda h, i, j: (h // q_heads_per_kv_head, j, 0),
                ),
            ]
            return pl.pallas_call(
                functools.partial(
                    _flash_attention_kernel,
                    grid_width=grid_width,
                    bkv=block_kv,
                    bkv_compute=block_kv_compute,
                    head_dim_v=head_dim,
                    kv_seq_len=kv_seq,
                ),
                grid=(num_q_heads, grid_height, grid_width),
                in_specs=in_specs,
                out_specs=pl.BlockSpec(
                    (None, block_q, head_dim), lambda h, i, j: (h, i, 0)
                ),
                scratch_shapes=[
                    pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
                    pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
                    pltpu.VMEM((head_dim, block_q), jnp.float32),
                ],
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel", "parallel", "arbitrary"),
                    flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
                    # Autotuned score tiles exceed Mosaic's 32 MiB default.
                    vmem_limit_bytes=_splash_vmem_limit_bytes(pltpu),
                ),
                out_shape=jax.ShapeDtypeStruct(
                    (num_q_heads, q_seq, head_dim), q_3d.dtype
                ),
            )(q_3d, k_3d, v_3d)

        return jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(query, key, value)

    wrapped = custom_jax_kernel(_jax_fn, warn_deprecated=False)
    return wrapped, q_seq, kv_seq


def _build_fullmask_splash_kernel(
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    block_q: int,
    block_kv: int,
    is_causal: bool = False,
):
    """Build a stock ``make_splash_mha`` kernel; padded keys are masked by segment ids."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask as sam,
    )
    from torch_tpu._internal.pallas.pallas import custom_jax_kernel

    q_padded = _round_up(q_seq, block_q)
    kv_padded = _round_up(kv_seq, block_kv)

    # Causal attention only reaches here with q_seq == kv_seq, so real rows
    # never see padded keys. Padded query rows are sliced off after the kernel.
    if is_causal:
        mask = sam.CausalMask((q_padded, kv_padded))
    else:
        mask = sam.FullMask((q_padded, kv_padded))
    pad_kv = kv_padded != kv_seq

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
        if not pad_kv:
            return kernel(q, k, v)
        # Zero-padded keys score 0; segment id 0 excludes them.
        segment_ids = splash.SegmentIds(
            q=jnp.ones((q_padded,), jnp.int32),
            kv=(jnp.arange(kv_padded) < kv_seq).astype(jnp.int32),
        )
        return kernel(q, k, v, segment_ids=segment_ids)

    wrapped = custom_jax_kernel(jax.vmap(splash_fn), warn_deprecated=False)
    return wrapped, q_padded, kv_padded


def _use_custom_kernel(q_seq: int, kv_seq: int, is_causal: bool) -> bool:
    """Long-KV non-causal attention, including a short SP query shard."""
    return (not is_causal) and q_seq >= 256 and kv_seq >= 2048


def _cross_attn_block_kv(kv_seq: int) -> int:
    """Largest KV block <= 1024 dividing ``kv_seq`` (a multiple of 128)."""
    for block_kv in (1024, 512, 256):
        if kv_seq % block_kv == 0:
            return block_kv
    return 128


def _build_splash_kernel(
    q_seq: int, kv_seq: int, head_dim: int, is_causal: bool, scale: float
) -> Tuple[object, int, int, bool]:
    """Return ``(kernel, q_padded, kv_padded, takes_unscaled_q)``."""
    # Long-KV self-attention and SP query shards.
    if _use_custom_kernel(q_seq, kv_seq, is_causal):
        blocks = autotune_block_sizes(q_seq, kv_seq, head_dim)
        kind = f"custom blocks={blocks}"
        result = (*_build_custom_splash_kernel(q_seq, kv_seq, head_dim, scale), True)

    # Long Q against short 128-aligned KV; block_kv divides kv_seq.
    elif not is_causal and q_seq >= 2048 and 128 <= kv_seq and kv_seq % 128 == 0:
        block_kv = _cross_attn_block_kv(kv_seq)
        kind = f"make_splash_mha blocks=(2048, {block_kv})"
        result = (
            *_build_fullmask_splash_kernel(
                q_seq, kv_seq, head_dim, block_q=2048, block_kv=block_kv
            ),
            False,
        )

    # Medium and causal attention.
    else:
        if q_seq % 1024 == 0 and kv_seq % 1024 == 0:
            block_q = block_kv = 1024
        elif q_seq >= 512 and kv_seq >= 512:
            block_q = block_kv = 512
        else:
            block_q = block_kv = 256
        kind = f"make_splash_mha blocks=({block_q}, {block_kv})"
        result = (
            *_build_fullmask_splash_kernel(
                q_seq,
                kv_seq,
                head_dim,
                block_q=block_q,
                block_kv=block_kv,
                is_causal=is_causal,
            ),
            False,
        )

    logger.info(
        "TPU splash attention: q_seq=%d kv_seq=%d causal=%s -> %s",
        q_seq,
        kv_seq,
        is_causal,
        kind,
    )
    return result


def _get_or_create_splash_kernel(
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    scale: float,
):
    """Select and cache the Pallas kernel for (q_seq, kv_seq)."""
    cache_key = (q_seq, kv_seq, head_dim, dtype, is_causal, scale)
    cached = _splash_kernel_cache.get(cache_key)
    if cached is None:
        cached = _build_splash_kernel(q_seq, kv_seq, head_dim, is_causal, scale)
        _splash_kernel_cache[cache_key] = cached
    return cached


@torch.library.custom_op(
    "sglang::tpu_splash_attention", mutates_args=(), device_types=["tpu"]
)
def _tpu_splash_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    q_seq, head_dim = q.shape[2], q.shape[3]
    kv_seq = k.shape[2]

    kernel, q_padded, kv_padded, takes_unscaled_q = _get_or_create_splash_kernel(
        q_seq, kv_seq, head_dim, q.dtype, is_causal, scale
    )
    if not takes_unscaled_q:
        # make_splash_mha: pre-scaled query, one KV head per query head. The
        # custom kernel maps query heads to KV heads itself.
        q = q * scale
        if q.shape[1] != k.shape[1]:
            repeat_factor = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

    need_pad_q = q_padded > q_seq
    need_pad_kv = kv_padded > kv_seq
    if need_pad_q:
        q = F.pad(q, (0, 0, 0, q_padded - q_seq))
    if need_pad_kv:
        k = F.pad(k, (0, 0, 0, kv_padded - kv_seq))
        v = F.pad(v, (0, 0, 0, kv_padded - kv_seq))

    with tpu_info_for_cpu_export():
        out = kernel(q.contiguous(), k.contiguous(), v.contiguous())

    if need_pad_q:
        out = out[:, :, :q_seq, :].contiguous()
    return out


@_tpu_splash_attention_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    # The kernels write a contiguous output whatever the query's strides.
    return q.new_empty(q.shape)


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

    q_seq = query.shape[2]
    kv_seq = key.shape[2]

    # Short sequences and unaligned short-KV cross-attention stay on SDPA.
    short = (q_seq <= 512 and kv_seq <= 512) or (
        not is_causal
        and q_seq != kv_seq
        and kv_seq <= 1024
        and (q_seq < 2048 or kv_seq % 128 != 0)
    )

    # The Pallas kernels compute in bf16.
    if short:
        reason = "short"
    elif is_causal and q_seq != kv_seq:
        reason = "ragged causal"
    elif head_dim != 128:
        reason = f"head_dim={head_dim}"
    elif not query.dtype == key.dtype == value.dtype == torch.bfloat16:
        reason = f"dtype={query.dtype}/{key.dtype}/{value.dtype}"
    elif not _check_splash_available():
        reason = "pallas unavailable"
    else:
        reason = None
    if reason is None:
        return _tpu_splash_attention_op(query, key, value, scale, is_causal=is_causal)

    log_key = (q_seq, kv_seq, is_causal, reason)
    if log_key not in _logged_sdpa_fallbacks:
        _logged_sdpa_fallbacks.add(log_key)
        logger.info(
            "TPU splash attention: q_seq=%d kv_seq=%d causal=%s -> sdpa (%s)",
            q_seq,
            kv_seq,
            is_causal,
            reason,
        )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
    )


class TpuSplashAttentionBackend(SDPABackend):
    """TPU Pallas attention, tuned on v7x; uncovered shapes fall back to SDPA."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TPU_SPLASH_ATTN

    @staticmethod
    def get_impl_cls() -> type[TpuSplashAttentionImpl]:
        return TpuSplashAttentionImpl


class TpuSplashAttentionImpl(SDPAImpl):
    """Routes eligible attention to the Pallas kernels, else to SDPA."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        self.use_splash = self.dropout == 0.0 and _check_splash_available()
        self._sdpa = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        if not self.use_splash:
            logger.info(
                "%s: using SDPA instead of the TPU splash kernels (dropout=%s).",
                prefix or type(self).__name__,
                self.dropout,
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # Inputs are [b, s, h, d]. Ragged causal attention needs an explicit
        # mask, which the kernels do not take.
        if (
            not self.use_splash
            or query.device.type != "tpu"
            or (self.causal and query.shape[1] != key.shape[1])
        ):
            return super().forward(query, key, value, attn_metadata)

        output = tpu_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=self.softmax_scale,
            is_causal=self.causal,
        )
        return output.transpose(1, 2)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        # Packed segment lengths vary per request; a Pallas kernel per length
        # would recompile, so segments stay on SDPA.
        return self._sdpa.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )
