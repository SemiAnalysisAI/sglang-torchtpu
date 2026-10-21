# SPDX-License-Identifier: Apache-2.0

"""One-pass expert-TP fused-RS post-kernel combine for models with a post-expert norm.

Replaces three XLA fusions (ep-shard sum, per-slot RMS variance, normalize + top-k
weighted sum) with one Pallas pass over the kernel's partial rows. Every bf16 cast of the JAX reference is
kept (torch_tpu compiles without XLA excess precision), so results differ only in
f32 summation order.
"""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Tokens per grid step: 32 overflowed scoped VMEM at a 4096-wide hidden.
_TOKENS_PER_BLOCK = 16


def _combine_kernel(x_ref, gain_ref, w_ref, o_ref, *, tokens, topk, ep, eps):
    rows = tokens * topk
    hidden = x_ref.shape[1] * x_ref.shape[2]
    x = x_ref[...].reshape(rows, ep, *x_ref.shape[1:])
    acc = x[:, 0].astype(jnp.float32)
    for e in range(1, ep):
        acc = acc + x[:, e].astype(jnp.float32)
    y = acc.astype(x_ref.dtype).astype(jnp.float32) * gain_ref[...][None]
    sum_sq = jnp.sum(jnp.sum(y * y, axis=1), axis=-1, keepdims=True)
    z = (y * jax.lax.rsqrt(sum_sq / hidden + eps)[:, :, None]).astype(x_ref.dtype)
    z = z.astype(jnp.float32).reshape(tokens, topk, *x_ref.shape[1:])
    w = w_ref[...].astype(jnp.float32)
    out = None
    for k in range(topk):
        prod = (z[:, k] * w[:, k : k + 1, None]).astype(o_ref.dtype)
        out = (
            prod.astype(jnp.float32) if out is None else out + prod.astype(jnp.float32)
        )
    o_ref[...] = out.astype(o_ref.dtype)


def supports_fused_post_combine(out_buf, chunk_size: int, norm_size: int) -> bool:
    """Tiled expert-TP output, unpadded hidden, and a token count the blocks tile."""
    if out_buf.ndim != 3 or out_buf.shape[1] * out_buf.shape[2] != norm_size:
        return False
    return chunk_size % 8 == 0 or chunk_size < 8


def fused_post_combine(
    out_buf: jax.Array,  # [chunk * topk * ep, H1, H2] partial rows, token-major
    post_norm_weight: jax.Array,  # [H1 * H2], zero-centered gain
    topk_weights: jax.Array,  # [chunk, topk]
    *,
    chunk_size: int,
    topk: int,
    ep_size: int,
    eps: float,
) -> jax.Array:
    """[chunk, H1 * H2]: per-slot ep sum, post-expert RMS norm, top-k weighted sum."""
    h1, h2 = out_buf.shape[1:]
    tokens = min(_TOKENS_PER_BLOCK, chunk_size)
    while chunk_size % tokens:
        tokens //= 2
    if tokens % 8 and tokens != chunk_size:
        tokens = chunk_size
    gain = (post_norm_weight.astype(jnp.float32) + 1.0).reshape(h1, h2)
    out_dtype = jnp.result_type(out_buf.dtype, topk_weights.dtype)
    out = pl.pallas_call(
        functools.partial(
            _combine_kernel, tokens=tokens, topk=topk, ep=ep_size, eps=eps
        ),
        grid=(chunk_size // tokens,),
        in_specs=[
            pl.BlockSpec((tokens * topk * ep_size, h1, h2), lambda i: (i, 0, 0)),
            pl.BlockSpec((h1, h2), lambda i: (0, 0)),
            pl.BlockSpec((tokens, topk), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tokens, h1, h2), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((chunk_size, h1, h2), out_dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(out_buf, gain, topk_weights)
    return out.reshape(chunk_size, h1 * h2)
