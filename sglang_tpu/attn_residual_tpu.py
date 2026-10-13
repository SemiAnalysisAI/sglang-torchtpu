# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas decode kernel for Kimi-K3 attention residuals on TPU.

One aggregation point is: optional pending residual add, a score per banked
row (RMSNorm folded into the score weight), softmax over the rows, the
weighted mix, and the output RMSNorm. The generic torch path lowers that to
about a dozen tiny XLA ops per call, and Kimi-K3 has two calls per layer. In
decode the rows are a few KB, so one kernel that keeps everything in VMEM
replaces them.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
from threading import Lock
from typing import Callable, Optional

import torch

# Decode-sized token counts only: the kernel holds the whole [T, NB, H] bank in
# VMEM. Prefill keeps the generic path.
MAX_DECODE_TOKENS = 64

_OPS: dict[bool, Callable] = {}
_OPS_LOCK = Lock()
# Plain bool, not dict truthiness: Dynamo traces the eligibility check.
_OPS_READY = False


def _decode_kernel(*refs, nvb: int, has_addend: bool, eps: float, out_eps: float):
    import jax
    import jax.numpy as jnp

    if has_addend:
        a_ref, b_ref, bank_ref, cw_ref, w_ref, out_ref, prefix_ref = refs
        # bf16 add, rounded like the torch `prefix_a + prefix_b` it replaces.
        prefix = (
            a_ref[...].astype(jnp.float32) + b_ref[...].astype(jnp.float32)
        ).astype(a_ref.dtype)
        prefix_ref[...] = prefix
    else:
        a_ref, bank_ref, cw_ref, w_ref, out_ref = refs
        prefix = a_ref[...]
    cw = cw_ref[...]
    values = [bank_ref[:, j, :].astype(jnp.float32) for j in range(nvb)]
    values.append(prefix.astype(jnp.float32))
    # score_proj(score_norm(v)) == sum(v * cw) * rsqrt(mean(v^2) + eps)
    scores = [
        jnp.sum(v * cw, axis=-1, keepdims=True)
        * jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + eps)
        for v in values
    ]
    m = functools.reduce(jnp.maximum, scores)
    exps = [jnp.exp(s - m) for s in scores]
    total = functools.reduce(jnp.add, exps)
    mixed = functools.reduce(jnp.add, [e * v for e, v in zip(exps, values)]) / total
    # Output RMSNorm on the bf16 mixture (RMSNorm.forward_native semantics).
    x = mixed.astype(out_ref.dtype).astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + out_eps)
    y = x * w_ref[...].astype(jnp.float32)
    # A [1, H] row tiles as (2, 128) and makes the consuming GEMV relayout it;
    # [H / 128, 128] tiles as (8, 128), the same bytes as XLA's 1-D [H].
    out_ref[...] = y.reshape(out_ref.shape).astype(out_ref.dtype)


def _normed_shape(tokens: int, hidden: int) -> tuple[int, int]:
    """One token: [H / 128, 128] (see the end of _decode_kernel); else [T, H]."""
    return (hidden // 128, 128) if tokens == 1 else (tokens, hidden)


def _vmem_limit_bytes(tokens: int, hidden: int, nvb: int, bank_elems: int) -> int:
    """Only what the kernel needs. A near-full limit stops XLA from keeping the
    next layers' weights prefetched in VMEM across the call (measured: the
    following router-gate GEMV went from 4.3 to 7.4 us)."""
    # Double-buffered bf16 inputs/outputs, plus f32 copies of the nvb + 1 rows
    # and a few f32 row temporaries.
    io = 2 * 2 * (bank_elems + 4 * tokens * hidden)
    work = 4 * tokens * hidden * (2 * (nvb + 1) + 6)
    return max(2 * 1024 * 1024, 2 * (io + work))


def attention_residual_decode_jax(
    prefix_a,
    prefix_b,
    bank,
    folded_weight,
    norm_weight,
    nvb: int,
    eps: float,
    out_eps: float,
    interpret: bool = False,
):
    """Returns (normed, prefix); prefix = prefix_a + prefix_b, or prefix_a.

    prefix_a / prefix_b: [T, H]; bank: [T, NB, H] with rows [:nvb] live;
    folded_weight: [H] f32 (score norm weight * score proj weight);
    norm_weight: [H] output RMSNorm weight. normed comes back in
    _normed_shape(T, H).
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, hidden = prefix_a.shape
    has_addend = prefix_b is not None
    normed_shape = _normed_shape(tokens, hidden)
    row_spec = pl.BlockSpec((tokens, hidden), lambda i: (0, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    in_specs = [row_spec] * (1 + has_addend)
    in_specs += [pl.BlockSpec(bank.shape, lambda i: (0, 0, 0)), vec_spec, vec_spec]
    out_shape = [jax.ShapeDtypeStruct(normed_shape, prefix_a.dtype)]
    out_specs = [pl.BlockSpec(normed_shape, lambda i: (0, 0))]
    if has_addend:
        out_shape.append(jax.ShapeDtypeStruct((tokens, hidden), prefix_a.dtype))
        out_specs.append(row_spec)
    outs = pl.pallas_call(
        functools.partial(
            _decode_kernel, nvb=nvb, has_addend=has_addend, eps=eps, out_eps=out_eps
        ),
        out_shape=out_shape,
        grid=(1,),
        in_specs=in_specs,
        out_specs=out_specs,
        # Let XLA fold the activations' producers (all-reduce relayouts, the
        # residual add) into the kernel's input copies.
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_limit_bytes(tokens, hidden, nvb, bank.size),
            allow_input_fusion=[True] * (1 + has_addend) + [False, False, False],
        ),
        interpret=interpret,
        name=f"kimi_attnres_decode_nvb{nvb}",
    )(
        *([prefix_a] + [prefix_b] * has_addend),
        bank,
        folded_weight.reshape(1, hidden).astype(jnp.float32),
        norm_weight.reshape(1, hidden),
    )
    return outs[0], (outs[1] if has_addend else prefix_a)


def _build_op(has_addend: bool) -> Callable:
    import jax  # noqa: F401  (force loud failure if JAX is missing)

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def _normed_like(prefix_a):
        return prefix_a.new_empty(_normed_shape(*prefix_a.shape))

    if has_addend:

        def _wrapped(
            prefix_a: jax.Array,
            prefix_b: jax.Array,
            bank: jax.Array,
            folded_weight: jax.Array,
            norm_weight: jax.Array,
            nvb: int,
            eps: float,
            out_eps: float,
        ) -> tuple[jax.Array, jax.Array]:
            return attention_residual_decode_jax(
                prefix_a, prefix_b, bank, folded_weight, norm_weight, nvb, eps, out_eps
            )

        def _fake(
            prefix_a, prefix_b, bank, folded_weight, norm_weight, nvb, eps, out_eps
        ):
            return _normed_like(prefix_a), torch.empty_like(prefix_a)

    else:

        def _wrapped(
            prefix_a: jax.Array,
            bank: jax.Array,
            folded_weight: jax.Array,
            norm_weight: jax.Array,
            nvb: int,
            eps: float,
            out_eps: float,
        ) -> jax.Array:
            return attention_residual_decode_jax(
                prefix_a, None, bank, folded_weight, norm_weight, nvb, eps, out_eps
            )[0]

        def _fake(prefix_a, bank, folded_weight, norm_weight, nvb, eps, out_eps):
            return _normed_like(prefix_a)

    name = "attn_res_decode_add" if has_addend else "attn_res_decode"
    op = cpu_export_jax_op(f"sglang_tpu_pallas::{name}", _wrapped)
    op.register_fake(_fake)
    return op


def register_attn_residual_decode_ops() -> None:
    """Register both ops; must run before torch.compile traces the model."""
    global _OPS_READY
    with _OPS_LOCK:
        for has_addend in (False, True):
            if has_addend not in _OPS:
                _OPS[has_addend] = _build_op(has_addend)
        _OPS_READY = True


def _decode_eligible(prefix: torch.Tensor, bank, nvb, out_norm, write_bank_row) -> bool:
    return (
        _OPS_READY
        and nvb > 0
        and not write_bank_row
        and 0 < prefix.shape[0] <= MAX_DECODE_TOKENS
        and prefix.shape[1] % 128 == 0
        and prefix.dtype == torch.bfloat16
        and bank.dtype == prefix.dtype
        and not getattr(out_norm, "cast_x_before_out_mul", False)
        and getattr(out_norm, "variance_size_override", None) is None
    )


def _run(
    prefix_a: torch.Tensor,
    prefix_b: Optional[torch.Tensor],
    bank: torch.Tensor,
    nvb: int,
    score_proj,
    score_norm,
    out_norm,
):
    from sglang.srt.layers.attn_residual import get_cw

    cw = get_cw(score_proj, score_norm)
    args = (
        cw,
        out_norm.weight,
        nvb,
        score_norm.variance_epsilon,
        out_norm.variance_epsilon,
    )
    if prefix_b is None:
        normed, prefix = _OPS[False](prefix_a, bank, *args), prefix_a
    else:
        normed, prefix = _OPS[True](prefix_a, prefix_b, bank, *args)
    return normed.reshape(prefix_a.shape), prefix


def aggregate_around(
    original,
    prefix_sum,
    bank,
    nvb,
    score_proj,
    score_norm,
    out_norm,
    write_bank_row=False,
):
    """AROUND hook for ``attn_residual._aggregate``."""
    if not _decode_eligible(prefix_sum, bank, nvb, out_norm, write_bank_row):
        return original(
            prefix_sum, bank, nvb, score_proj, score_norm, out_norm, write_bank_row
        )
    return _run(prefix_sum, None, bank, nvb, score_proj, score_norm, out_norm)[0]


def aggregate_fused_add_around(
    original,
    prefix_a,
    prefix_b,
    bank,
    nvb,
    score_proj,
    score_norm,
    out_norm,
    write_bank_row=False,
):
    """AROUND hook for ``attn_residual._aggregate_fused_add``."""
    if not (
        _decode_eligible(prefix_a, bank, nvb, out_norm, write_bank_row)
        and prefix_b.shape == prefix_a.shape
        and prefix_b.dtype == prefix_a.dtype
    ):
        return original(
            prefix_a,
            prefix_b,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row,
        )
    return _run(prefix_a, prefix_b, bank, nvb, score_proj, score_norm, out_norm)
