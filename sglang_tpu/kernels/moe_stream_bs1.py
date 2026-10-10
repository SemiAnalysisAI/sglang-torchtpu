# SPDX-License-Identifier: Apache-2.0
"""One-token fused MoE that streams the selected experts' weights.

At batch size 1 with all experts local (tensor-parallel experts, EP = 1), the
routed rows are k copies of one token. A grid over the k slots with the expert
ids scalar-prefetched makes each step's weight blocks w1[ids[s]] / w2[ids[s]],
so the pipeline fetches the next expert's weights while the current one
computes, and the weighted sum accumulates on-chip. This replaces route-meta,
the row broadcast, both grouped GEMMs, the activation and the combine.

Per expert it reproduces gmm_v2's quantized path (per-row FP8 quantization of
the lhs in 512-wide K blocks, block partials rescaled by lhs and per-channel rhs
scales, bf16 accumulation) and fused_moe's activation (silu(min(gate, l)) * clip(up, l),
rounded once). Each expert output is rounded to the activation dtype before
the f32 weighted sum, as the fused-MoE combine does.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LHS_QUANT_BLOCK = 512


def supports(tokens: int, hidden: int, two_n: int) -> bool:
    return tokens == 1 and hidden % 128 == 0 and two_n % 256 == 0


def _block_quant_dot(x, w, w_scale, *, fp8_dtype):
    """gmm_v2's quantized matmul for one row: x [1, K] in the activation dtype,
    w [K, N]. The lhs scale, its reciprocal and the accumulator are in the
    activation dtype, as gmm_v2 runs a quantized lhs (bf16 accumulation); each
    block partial is rescaled in f32 and rounded back before accumulating."""
    fp8_max = float(jnp.finfo(fp8_dtype).max)
    dt = x.dtype
    k = x.shape[1]
    block = min(_LHS_QUANT_BLOCK, k)
    acc = None
    for start in range(0, k, block):
        xb = x[:, start : start + block]
        scale = jnp.max(jnp.abs(xb), axis=1, keepdims=True) / fp8_max
        scale_inv = jnp.where(scale == 0, 0, 1 / scale)
        partial = jax.lax.dot_general(
            (xb * scale_inv).astype(fp8_dtype),
            w[start : start + block, :],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(dt)
        partial = (
            partial.astype(jnp.float32) * scale.astype(jnp.float32) * w_scale
        ).astype(dt)
        acc = partial if acc is None else acc + partial
    return acc


def _kernel(
    ids_ref,
    wts_ref,
    x_ref,
    w1_ref,
    s1_ref,
    w2_ref,
    s2_ref,
    o_ref,
    acc_ref,
    *,
    n,
    limit,
    fp8_dtype,
):
    s = pl.program_id(0)

    @pl.when(s == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    dt = o_ref.dtype
    dot = functools.partial(_block_quant_dot, fp8_dtype=fp8_dtype)
    gate_up = dot(x_ref[...], w1_ref[0], s1_ref[0])
    gate, up = gate_up[:, :n], gate_up[:, n:]
    if limit is not None:
        gate = jnp.minimum(gate, limit)
        up = jnp.clip(up, -limit, limit)
    h = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(dt)
    y = dot(h, w2_ref[0], s2_ref[0])
    acc_ref[...] += y.astype(jnp.float32) * wts_ref[s]

    @pl.when(s == pl.num_programs(0) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(dt)


def moe_stream_bs1(
    x: jax.Array,
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array,
    w2: jax.Array,
    w2_scale: jax.Array,
    clamp_limit: float | None,
) -> jax.Array:
    """x [1, H]; ids / weights [1, k]; w1 [E, H, 2n], w2 [E, n, H] FP8 with
    per-channel scales [E, 1, 1, 2n] / [E, 1, 1, H]. Returns [1, H]."""
    _, hidden = x.shape
    k = topk_ids.shape[1]
    num_experts, _, two_n = w1.shape
    n = two_n // 2
    s1 = w1_scale.reshape(num_experts, 1, two_n).astype(jnp.float32)
    s2 = w2_scale.reshape(num_experts, 1, hidden).astype(jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(k,),
        in_specs=[
            pl.BlockSpec((1, hidden), lambda s, ids, wts: (0, 0)),
            pl.BlockSpec((1, hidden, two_n), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, 1, two_n), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, n, hidden), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, 1, hidden), lambda s, ids, wts: (ids[s], 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, hidden), lambda s, ids, wts: (0, 0)),
        scratch_shapes=[pltpu.VMEM((1, hidden), jnp.float32)],
    )
    return pl.pallas_call(
        functools.partial(_kernel, n=n, limit=clamp_limit, fp8_dtype=w1.dtype),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((1, hidden), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"moe_stream_bs1_{num_experts}x{hidden}x{two_n}",
    )(
        topk_ids.reshape(k).astype(jnp.int32),
        topk_weights.reshape(k).astype(jnp.float32),
        x,
        w1,
        s1,
        w2,
        s2,
    )
