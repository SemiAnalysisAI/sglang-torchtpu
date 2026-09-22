# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas Splash Attention and Fast Cross-Attention kernels for multimodal_gen.

Adapted from DiffusionServing (runtime/attention/torch_tpu.py,
runtime/kernels/block_sizes.py, and models/wan/attention.py).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_splash_kernel_cache: Dict[tuple, Tuple[object, int, int]] = {}
_SPLASH_AVAILABLE: bool | None = None


def _check_splash_available() -> bool:
    global _SPLASH_AVAILABLE
    if _SPLASH_AVAILABLE is not None:
        return _SPLASH_AVAILABLE
    try:
        import jax  # noqa: F401
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


def get_splash_block_sizes(
    q_seq: int,
    kv_seq: int,
    head_dim: int = 128,
) -> Tuple[int, int]:
    """Select optimal (block_q, block_kv) for TPU Pallas Splash Attention.

    Prefers exact-divisor block sizes (1024 or 512) so FullMask is used with
    zero padding whenever possible (e.g. LTX-2 S=9216 -> 1024; FLUX.2 S=4608 -> 512).
    For non-multiples (e.g. Wan2.2 S=6160, Z-Image S=4128), uses 512 to minimize
    padding overhead to at most 511 tokens.
    """
    del head_dim
    if q_seq >= 1024 and kv_seq >= 1024 and q_seq % 1024 == 0 and kv_seq % 1024 == 0:
        return 1024, 1024
    if q_seq >= 512 and kv_seq >= 512:
        return 512, 512
    if q_seq >= 256 and kv_seq >= 256:
        return 256, 256
    return 128, 128


def _get_or_create_splash_kernel(
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
):
    """Build or retrieve a cached Pallas Splash MHA kernel."""
    block_q, block_kv = get_splash_block_sizes(q_seq, kv_seq, head_dim)
    q_padded = _round_up(q_seq, block_q)
    kv_padded = _round_up(kv_seq, block_kv)

    cache_key = (q_seq, kv_seq, head_dim, dtype, is_causal, block_q, block_kv)
    cached = _splash_kernel_cache.get(cache_key)
    if cached is not None:
        return cached

    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as sam,
    )
    from torch_tpu._internal.pallas.pallas import custom_jax_kernel

    if is_causal:
        if q_padded == q_seq and kv_padded == kv_seq:
            mask = sam.CausalMask((q_padded, kv_padded))
        else:
            mask_arr = np.zeros((q_padded, kv_padded), dtype=np.bool_)
            valid = np.tril(np.ones((q_seq, kv_seq), dtype=np.bool_))
            mask_arr[:q_seq, :kv_seq] = valid
            mask = sam.NumpyMask(mask_arr)
    elif q_padded == q_seq and kv_padded == kv_seq:
        mask = sam.FullMask((q_padded, kv_padded))
    else:
        mask_arr = np.zeros((q_padded, kv_padded), dtype=np.bool_)
        mask_arr[:q_seq, :kv_seq] = True
        mask = sam.NumpyMask(mask_arr)

    block_sizes = splash.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv,
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

    batched_splash = jax.vmap(splash_fn)

    wrapped = custom_jax_kernel(
        batched_splash,
        warn_deprecated=False,
    )
    result = (wrapped, q_padded, kv_padded)
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
    """Fast XLA-fused dot-product attention for short KV sequences (e.g. text cross-attention).

    Pre-scales key (which is much shorter than query in diffusion cross-attention)
    and computes matmul -> fp32 softmax -> matmul, which XLA fuses into a single
    MXU+VPU kernel without tile-padding overhead.
    """
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
    """Dispatch [B, H, S, D] attention on TPU to Pallas Splash Attention or Fast Cross-Attention."""
    head_dim = query.shape[-1]
    if scale is None:
        scale = head_dim**-0.5

    if query.shape[1] != key.shape[1]:
        repeat_factor = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    q_seq = query.shape[2]
    kv_seq = key.shape[2]

    # Short-sequence or asymmetric cross-attention with short KV (e.g. Wan2.2 / LTX-2 text cross-attn)
    if (not is_causal) and (
        (q_seq <= 512 and kv_seq <= 512) or (q_seq != kv_seq and kv_seq <= 1024)
    ):
        return tpu_fast_cross_attention(query, key, value, scale=scale)

    # Full-sequence self-attention / joint-attention via Pallas Splash Attention
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
