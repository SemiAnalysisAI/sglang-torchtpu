# SPDX-License-Identifier: Apache-2.0
"""Fused mHC (manifold-constrained hyper-connection) site kernel for GLM-5.3-Flash.

One Pallas call per mHC site replaces the torch reference's ~80 small XLA ops:
the previous sublayer's hc_post (stream remix) and the next sublayer's hc_pre
(RMS-scaled 4*d -> 24 projection, sigmoid gates, softmax plus Sinkhorn on the
4x4 mix, stream collapse). Gate and Sinkhorn math follows sglang-jax
sgl_jax/srt/kernels/mhc/mhc.py (63584e276f4b); the post/pre seam follows
tpu-inference kernels/experimental/deepseek_v4/mhc/fused_post_pre_kernel.py.

Layouts: streams are [T, hc * d] bf16 (stream-major), post is [T, hc] f32 and
comb is [T, hc * hc] f32 row-major in (i, j), matching
y[t, j] = post[t, j] * x[t] + sum_i comb[t, i, j] * res[t, i].
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# [1, VEC_WIDTH] f32 parameter row: hc_base in [0, mix), hc_scale in [mix, mix + 3).
VEC_WIDTH = 128
# Prefill token block; decode buckets below this run as one block.
_MAX_BLOCK_TOKENS = 32
_PREFILL_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def pack_hc_vec(hc_scale: jax.Array, hc_base: jax.Array) -> jax.Array:
    mix = hc_base.shape[0]
    vec = jnp.zeros((1, VEC_WIDTH), jnp.float32)
    vec = vec.at[0, :mix].set(hc_base.astype(jnp.float32))
    return vec.at[0, mix : mix + 3].set(hc_scale.astype(jnp.float32))


def _remix(x_ref, res_ref, post_ref, comb_ref, new_res_ref, *, hc, d):
    """hc_post: returns the new streams as f32 values of their bf16 rounding."""
    x = x_ref[...].astype(jnp.float32)
    post = post_ref[...]
    comb = comb_ref[...]
    old = [res_ref[:, i * d : (i + 1) * d].astype(jnp.float32) for i in range(hc)]
    streams = []
    for j in range(hc):
        acc = post[:, j : j + 1] * x
        for i in range(hc):
            acc = acc + comb[:, i * hc + j : i * hc + j + 1] * old[i]
        rounded = acc.astype(new_res_ref.dtype)
        new_res_ref[:, j * d : (j + 1) * d] = rounded
        streams.append(rounded.astype(jnp.float32))
    return streams


def _gates_and_collapse(
    streams,
    fn_ref,
    vec_ref,
    layer_ref,
    post_ref,
    comb_ref,
    *,
    hc,
    d,
    iters,
    rms_eps,
    hc_eps,
):
    """hc_pre on f32 streams: writes the collapsed layer input, post and comb."""
    mix = (2 + hc) * hc
    sq = functools.reduce(
        jnp.add, [jnp.sum(s * s, axis=-1, keepdims=True) for s in streams]
    )
    rms = jax.lax.rsqrt(sq / (hc * d) + rms_eps)
    dims = (((1,), (1,)), ((), ()))
    mixes = (
        functools.reduce(
            jnp.add,
            [
                jax.lax.dot_general(
                    s,
                    fn_ref[:, i * d : (i + 1) * d],
                    dims,
                    precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32,
                )
                for i, s in enumerate(streams)
            ],
        )
        * rms
    )
    vec = vec_ref[...]
    base = vec[:, :mix]
    s0, s1, s2 = (
        vec[:, mix : mix + 1],
        vec[:, mix + 1 : mix + 2],
        vec[:, mix + 2 : mix + 3],
    )

    pre = jax.nn.sigmoid(mixes[:, :hc] * s0 + base[:, :hc]) + hc_eps
    post_ref[...] = 2.0 * jax.nn.sigmoid(
        mixes[:, hc : 2 * hc] * s1 + base[:, hc : 2 * hc]
    )

    # comb row i is a [bt, hc] value over columns j; rows normalize over lanes,
    # columns across the hc row values.
    rows = []
    for i in range(hc):
        lo = 2 * hc + i * hc
        r = mixes[:, lo : lo + hc] * s2 + base[:, lo : lo + hc]
        r = jnp.exp(r - jnp.max(r, axis=-1, keepdims=True))
        rows.append(r / jnp.sum(r, axis=-1, keepdims=True) + hc_eps)

    def col_norm(rs):
        col = functools.reduce(jnp.add, rs) + hc_eps
        return [r / col for r in rs]

    rows = col_norm(rows)
    for _ in range(iters - 1):
        rows = [r / (jnp.sum(r, axis=-1, keepdims=True) + hc_eps) for r in rows]
        rows = col_norm(rows)
    comb_ref[...] = jnp.concatenate(rows, axis=-1)

    layer = functools.reduce(
        jnp.add, [pre[:, i : i + 1] * s for i, s in enumerate(streams)]
    )
    layer_ref[...] = layer.astype(layer_ref.dtype)


def _site_kernel(*refs, hc, d, has_post, has_pre, iters, rms_eps, hc_eps):
    refs = list(refs)
    if has_post:
        x_ref, res_ref, post_in_ref, comb_in_ref = refs[:4]
        refs = refs[4:]
    else:
        res_ref = refs.pop(0)
    if has_pre:
        fn_ref, vec_ref = refs[:2]
        refs = refs[2:]
    outs = refs
    if has_post:
        new_res_ref = outs.pop(0)
        streams = _remix(
            x_ref, res_ref, post_in_ref, comb_in_ref, new_res_ref, hc=hc, d=d
        )
    else:
        streams = [
            res_ref[:, i * d : (i + 1) * d].astype(jnp.float32) for i in range(hc)
        ]
    if has_pre:
        layer_ref, post_ref, comb_ref = outs
        _gates_and_collapse(
            streams,
            fn_ref,
            vec_ref,
            layer_ref,
            post_ref,
            comb_ref,
            hc=hc,
            d=d,
            iters=iters,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
        )


def mhc_site(
    x,
    res,
    post,
    comb,
    fn,
    vec,
    *,
    hc: int,
    iters: int,
    rms_eps: float,
    hc_eps: float,
):
    """One mHC site. x/post/comb are None for the first hc_pre; fn/vec are None
    for the final hc_post. Returns the outputs present among
    (new_res [T, hc*d], layer_input [T, d], post [T, hc], comb [T, hc*hc])."""
    has_post = x is not None
    has_pre = fn is not None
    tokens, width = res.shape
    d = width // hc
    mix = (2 + hc) * hc
    bt = tokens if tokens <= _MAX_BLOCK_TOKENS else _MAX_BLOCK_TOKENS
    padded = pl.cdiv(tokens, bt) * bt

    def pad(a):
        return (
            a
            if a is None or padded == tokens
            else jnp.pad(a, ((0, padded - tokens), (0, 0)))
        )

    def rows_spec(cols):
        return pl.BlockSpec((bt, cols), lambda t: (t, 0))

    def full_spec(shape):
        return pl.BlockSpec(shape, lambda t: (0, 0))

    operands, in_specs, out_shape, out_specs = [], [], [], []
    if has_post:
        operands += [
            pad(x),
            pad(res),
            pad(post.astype(jnp.float32)),
            pad(comb.astype(jnp.float32)),
        ]
        in_specs += [rows_spec(d), rows_spec(width), rows_spec(hc), rows_spec(hc * hc)]
        out_shape.append(jax.ShapeDtypeStruct((padded, width), res.dtype))
        out_specs.append(rows_spec(width))
    else:
        operands.append(pad(res))
        in_specs.append(rows_spec(width))
    if has_pre:
        operands += [fn.astype(jnp.float32), vec]
        in_specs += [full_spec((mix, width)), full_spec((1, VEC_WIDTH))]
        out_shape += [
            jax.ShapeDtypeStruct((padded, d), res.dtype),
            jax.ShapeDtypeStruct((padded, hc), jnp.float32),
            jax.ShapeDtypeStruct((padded, hc * hc), jnp.float32),
        ]
        out_specs += [rows_spec(d), rows_spec(hc), rows_spec(hc * hc)]

    params = {"dimension_semantics": ("parallel",)}
    if bt > 16:
        params["vmem_limit_bytes"] = _PREFILL_VMEM_LIMIT_BYTES
    outs = pl.pallas_call(
        functools.partial(
            _site_kernel,
            hc=hc,
            d=d,
            has_post=has_post,
            has_pre=has_pre,
            iters=iters,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
        ),
        grid=(padded // bt,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(**params),
        name=f"glm_mhc_{'post' if has_post else ''}{'pre' if has_pre else ''}",
    )(*operands)
    return tuple(o[:tokens] for o in outs)
