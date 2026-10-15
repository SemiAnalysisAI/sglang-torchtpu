# SPDX-License-Identifier: Apache-2.0
"""Single-token (bs=1 decode) packed-MXFP4 fused MoE kernel.

The general path runs route metadata, a broadcast, gmm_v2 for GMM1, the
gated activation, gmm_v2 for GMM2 and the weighted combine. Each gmm_v2 call
walks every local expert group to build its metadata and zero-fills its
output, so at batch 1 the fixed cost dominates: on Kimi-K3 shapes (112 local
experts) it is about 11 us per MoE layer before any weight is read.

Here one Pallas call does it all:

1. a scalar SMEM loop over the k routed slots compacts the local hits (ids
   and weights) to the front;
2. an inner pipeline with a dynamic grid (hits, intermediate tiles) streams
   only the hit experts' weight tiles;
3. each step dequantizes the gate/up and down tiles in VMEM, runs the gate/up
   GEMV, the gated activation and the down GEMV, and accumulates the weighted
   result.

The intermediate never goes to HBM and experts nobody hit cost nothing.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# One token, padded to a full sublane tile.
_ROWS = 8
# Intermediate tile widths, largest first; whole 768-wide expert slices
# measured fastest on Kimi-K3 shapes.
_TILE_N = (768, 512, 384, 256, 128)


def _dequant(w_ref, s_ref, rows: int):
    """[1, rows / 2, tn] packed FP4 and [1, rows / 32, 1, tn] scales -> [rows, tn] bf16."""
    w = pltpu.bitcast(w_ref.bitcast(jnp.uint32)[0], jnp.float4_e2m1fn)
    tn = w.shape[-1]
    s = s_ref[0].astype(jnp.bfloat16)
    return (w.astype(jnp.bfloat16).reshape(rows // 32, 32, tn) * s).reshape(rows, tn)


def _kernel(
    ids_ref,
    w_ref,
    x_ref,
    w13_hbm,
    s13_hbm,
    w2_hbm,
    s2_hbm,
    o_ref,
    acc_ref,
    hit_ids_ref,
    num_hits_ref,
    hit_w_ref,
    *,
    k: int,
    offset: int,
    num_experts: int,
    k_dim: int,
    h_out: int,
    num_tiles: int,
    tn: int,
    act: str,
):
    from sglang_tpu.moe.pallas.gmm_v2 import apply_act_fn

    num_hits_ref[0] = 0

    def compact(i, carry):
        e = ids_ref[i] - offset

        # Non-local ids and TopK's -1 for padded tokens fall outside the range.
        @pl.when((e >= 0) & (e < num_experts))
        def _():
            c = num_hits_ref[0]
            hit_ids_ref[c] = e
            hit_w_ref[c] = jnp.full((8, 128), w_ref[i], jnp.float32)
            num_hits_ref[0] = c + 1

        return carry

    lax.fori_loop(0, k, compact, 0, unroll=True)
    acc_ref[...] = jnp.zeros_like(acc_ref)
    x = x_ref[...]

    def body(wt, w1_gate, w1_up, s1_gate, s1_up, w2, s2):
        gate = jnp.dot(
            x, _dequant(w1_gate, s1_gate, k_dim), preferred_element_type=jnp.float32
        )
        up = jnp.dot(
            x, _dequant(w1_up, s1_up, k_dim), preferred_element_type=jnp.float32
        )
        h = apply_act_fn(jnp.concatenate([gate, up], axis=-1), tn, act)
        y = jnp.dot(
            h.astype(jnp.bfloat16),
            _dequant(w2, s2, tn),
            preferred_element_type=jnp.float32,
        )
        acc_ref[...] += wt[0, :, :1] * y

    hit = lambda i: hit_ids_ref[i]
    pltpu.emit_pipeline(
        body,
        grid=(num_hits_ref[0], num_tiles),
        in_specs=[
            pl.BlockSpec((1, 8, 128), lambda i, j: (i, 0, 0)),
            # w13 is [E, K / 2, 2N] with the gate half first.
            pl.BlockSpec((1, k_dim // 2, tn), lambda i, j: (hit(i), 0, j)),
            pl.BlockSpec((1, k_dim // 2, tn), lambda i, j: (hit(i), 0, j + num_tiles)),
            pl.BlockSpec((1, k_dim // 32, 1, tn), lambda i, j: (hit(i), 0, 0, j)),
            pl.BlockSpec(
                (1, k_dim // 32, 1, tn), lambda i, j: (hit(i), 0, 0, j + num_tiles)
            ),
            pl.BlockSpec((1, tn // 2, h_out), lambda i, j: (hit(i), j, 0)),
            pl.BlockSpec((1, tn // 32, 1, h_out), lambda i, j: (hit(i), j, 0, 0)),
        ],
        out_specs=[],
    )(hit_w_ref, w13_hbm, w13_hbm, s13_hbm, s13_hbm, w2_hbm, s2_hbm)
    o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def eligible(
    w1, w2, w1_scale, w2_scale, w1_bias, w2_bias, w1_channel_scale, w2_channel_scale
) -> bool:
    """Packed MXFP4 (one scale per 32 along K, per-column scales), no biases,
    no VMEM-FP8 channel scales, and a 128-aligned intermediate width."""
    if w1.dtype != jnp.uint8 or w2.dtype != jnp.uint8:
        return False
    if w1_scale is None or w2_scale is None:
        return False
    if any(
        t is not None for t in (w1_bias, w2_bias, w1_channel_scale, w2_channel_scale)
    ):
        return False
    k_dim = w1.shape[1] * 2
    n = w1.shape[2] // 2
    return (
        w1_scale.shape[1] * 32 == k_dim
        and w2_scale.shape[1] * 32 == n
        and w1_scale.shape[-1] == w1.shape[2]
        and w2_scale.shape[-1] == w2.shape[2]
        and n % 128 == 0
        and k_dim % 256 == 0
    )


def single_token_moe(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_ids,
    topk_weights,
    *,
    offset: int,
    act: str,
    hidden_size: int,
):
    """hidden_states [1, K]; topk_ids / topk_weights [1, k] with global ids.

    Local experts are ids in [offset, offset + w1.shape[0]). Returns the
    weighted sum over the local hits, [1, hidden_size] in hidden_states' dtype.
    """
    num_experts = w1.shape[0]
    k_dim = w1.shape[1] * 2
    n = w1.shape[2] // 2
    h_out = w2.shape[2]
    tn = next(t for t in _TILE_N if n % t == 0)
    k = topk_ids.shape[1]
    x = jnp.pad(
        hidden_states[:1, :k_dim].astype(jnp.bfloat16), ((0, _ROWS - 1), (0, 0))
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            k=k,
            offset=offset,
            num_experts=num_experts,
            k_dim=k_dim,
            h_out=h_out,
            num_tiles=n // tn,
            tn=tn,
            act=act,
        ),
        out_shape=jax.ShapeDtypeStruct((_ROWS, h_out), hidden_states.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 4,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((_ROWS, h_out), jnp.float32),
                pltpu.SMEM((k,), jnp.int32),
                pltpu.SMEM((1,), jnp.int32),
                pltpu.VMEM((k, 8, 128), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=48 * 1024 * 1024),
        name=f"single_token_moe_e{num_experts}_n{n}",
    )(
        topk_ids[0].astype(jnp.int32),
        topk_weights[0].astype(jnp.float32),
        x,
        w1,
        w1_scale,
        w2,
        w2_scale,
    )
    return out[:1, :hidden_size]
