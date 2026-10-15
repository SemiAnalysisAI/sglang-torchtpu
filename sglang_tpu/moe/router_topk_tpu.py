# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas router top-k for small batches on TPU.

``biased_topk_impl`` (sigmoid scores + correction bias, then top-k) lowers on
TPU to an XLA sort over all experts plus a gather and reductions: 7 us per
MoE layer for Kimi-K3's top-16 of 896 at batch 1. This kernel does sigmoid,
the biased choice, k rounds of argmax-and-mask, the weight gather and the
renormalization in VMEM. Core's select_experts calls ``router_topk`` on TPU
and keeps the sort when it returns None.

Ties pick the lowest expert id, which ``torch.topk`` does not promise; the
selected set differs from the sort only on exactly tied scores.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
from threading import Lock
from typing import Callable

import torch

# Batch sizes the kernel serves; larger batches keep the sort.
MAX_TOKENS = 64

_OP: list = []
_OP_LOCK = Lock()
# Plain bool: Dynamo traces the eligibility check.
_OP_READY = False


def _topk_kernel(
    logits_ref,
    bias_ref,
    w_ref,
    ids_ref,
    *,
    k: int,
    renormalize: bool,
    eps: float,
    scale: float,
):
    import jax
    import jax.numpy as jnp

    logits = logits_ref[...]
    # sigmoid in f32, rounded to the logits dtype like torch's bf16 sigmoid.
    scores = (
        (1.0 / (1.0 + jnp.exp(-logits.astype(jnp.float32))))
        .astype(logits.dtype)
        .astype(jnp.float32)
    )
    choice = scores + bias_ref[...].astype(jnp.float32)
    tokens, experts = choice.shape
    lane = jnp.broadcast_to(
        jnp.arange(experts, dtype=jnp.int32)[None, :], (tokens, experts)
    )
    col = jnp.broadcast_to(jnp.arange(k, dtype=jnp.int32)[None, :], (tokens, k))
    ids = jnp.zeros((tokens, k), jnp.int32)
    weights = jnp.zeros((tokens, k), jnp.float32)
    neg = jnp.float32(-jnp.inf)
    packed = logits.dtype == jnp.bfloat16
    if packed:
        # bf16 scores fit in 16 bits: key = expert * 2^16 + score bits, so the
        # min over the tied keys gives the lowest tied expert and its weight
        # in one reduction.
        bits = jax.lax.bitcast_convert_type(scores, jnp.int32) >> 16
        key = lane * 65536 + (bits & 0xFFFF)
        no_key = jnp.int32(experts * 65536)
    for r in range(k):
        m = jnp.max(choice, axis=-1, keepdims=True)
        if packed:
            best = jnp.min(jnp.where(choice == m, key, no_key), axis=-1, keepdims=True)
            idx = best >> 16
            w = jax.lax.bitcast_convert_type((best & 0xFFFF) << 16, jnp.float32)
        else:
            idx = jnp.min(jnp.where(choice == m, lane, experts), axis=-1, keepdims=True)
            w = jnp.sum(jnp.where(lane == idx, scores, 0.0), axis=-1, keepdims=True)
        ids = jnp.where(col == r, idx, ids)
        weights = jnp.where(col == r, w, weights)
        choice = jnp.where(lane == idx, neg, choice)
    if renormalize:
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + eps)
        weights = weights * scale
    w_ref[...] = weights
    ids_ref[...] = ids


def router_topk_jax(
    logits,
    bias,
    k: int,
    renormalize: bool,
    eps: float,
    scale: float,
    interpret: bool = False,
):
    """logits: [T, E]; bias: [E] f32. Returns (f32 weights [T, k], i32 ids [T, k])."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, experts = logits.shape
    full = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))
    return pl.pallas_call(
        functools.partial(
            _topk_kernel, k=k, renormalize=renormalize, eps=eps, scale=scale
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, k), jnp.float32),
            jax.ShapeDtypeStruct((tokens, k), jnp.int32),
        ),
        grid=(1,),
        in_specs=[full((tokens, experts)), full((1, experts))],
        out_specs=(full((tokens, k)), full((tokens, k))),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=4 * 1024 * 1024,
            allow_input_fusion=[True, False],
        ),
        interpret=interpret,
        name=f"router_topk_{k}_of_{experts}",
    )(logits, bias.reshape(1, experts).astype(jnp.float32))


def _build_op() -> Callable:
    import jax  # noqa: F401  (force loud failure if JAX is missing)

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        logits: jax.Array,
        bias: jax.Array,
        k: int,
        renormalize: bool,
        eps: float,
        scale: float,
    ) -> tuple[jax.Array, jax.Array]:
        return router_topk_jax(logits, bias, k, renormalize, eps, scale)

    def _fake(logits, bias, k, renormalize, eps, scale):
        t = logits.shape[0]
        return (
            logits.new_empty((t, k), dtype=torch.float32),
            logits.new_empty((t, k), dtype=torch.int32),
        )

    op = cpu_export_jax_op("sglang_tpu_pallas::router_topk", _wrapped)
    op.register_fake(_fake)
    return op


def register_router_topk_op() -> None:
    """Register the op; TpuSRTPlatform.init_backend calls this before any model
    is built, so torch.compile traces a registered op."""
    global _OP_READY
    with _OP_LOCK:
        if not _OP:
            _OP.append(_build_op())
        _OP_READY = True


def router_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output: bool,
):
    """(topk_weights, topk_ids) of sigmoid scores plus a shared correction bias,
    or None when the kernel does not apply and the caller keeps its top-k.

    Callers pass sigmoid routing over a single expert group only.
    """
    if not (
        _OP_READY
        and num_fused_shared_experts == 0
        and correction_bias.dim() == 1
        and gating_output.dim() == 2
        and 0 < gating_output.shape[0] <= MAX_TOKENS
        and gating_output.shape[1] % 128 == 0
        and topk <= 128
    ):
        return None
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

    scale = (
        float(routed_scaling_factor)
        if renormalize
        and apply_routed_scaling_factor_on_output
        and routed_scaling_factor is not None
        else 1.0
    )
    return _OP[0](
        gating_output,
        correction_bias,
        topk,
        bool(renormalize),
        float(_RENORMALIZE_SUM_EPSILON),
        scale,
    )
