# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Fused Llama4x attention-to-MoE prelude for small decode batches.

The kernel owns the local sequence:

    post_attn_rmsnorm(attn_out) -> norm-preserving residual add -> pre_ffn scale

It returns the residual stream ``h`` plus either the per-token pre-FFN RMSNorm
scale or the fully prepared ``h * scale * weight`` FFN input. The prepared form
keeps all elementwise work from the post-attention norm through the pooler/router
input in one Pallas call.

The conservative sharded wrapper uses the equivalent XLA expression. The
opt-in Pallas wrapper preserves the same BF16 rounding boundaries and emits the
prepared FFN input directly.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import align_to
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_router import (
    route_hidden_block,
    router_block_tokens,
)

_ROUTER_OUTPUT_TILE = 128


def ref_fused_attention_moe_prelude(
    attn_out,
    residual_stream,
    post_weight,
    gate,
    *,
    post_eps: float,
    pre_eps: float,
    temperature: float = 0.3,
    residual_eps: float = 1e-3,
):
    """Reference implementation matching ``Llama4xDecoderLayer`` pre-FFN math."""
    dtype = residual_stream.dtype
    attn_f32 = attn_out.astype(jnp.float32)
    post_var = jnp.mean(attn_f32 * attn_f32, axis=-1, keepdims=True)
    branch = attn_f32 * jax.lax.rsqrt(post_var + post_eps)
    branch = branch.astype(dtype)
    if post_weight is not None:
        branch = (branch * post_weight.astype(jnp.float32)[jnp.newaxis, :]).astype(
            dtype
        )

    w = gate.astype(jnp.float32) / temperature
    beta = jax.nn.sigmoid(w)
    alpha_sq = jax.nn.sigmoid(-w) * (1.0 + beta)
    alpha = jnp.sqrt(jnp.maximum(alpha_sq, residual_eps))
    h = (
        alpha[jnp.newaxis, :] * residual_stream.astype(jnp.float32)
        + beta[jnp.newaxis, :] * branch.astype(jnp.float32)
    ).astype(dtype)

    h_f32 = h.astype(jnp.float32)
    pre_var = jnp.mean(h_f32 * h_f32, axis=-1)
    pre_scale = jax.lax.rsqrt(pre_var + pre_eps)
    return h, pre_scale


def _prelude_values(
    attn,
    stream,
    post_weight,
    gate,
    pre_weight,
    *,
    has_post_weight: bool,
    post_eps: float,
    pre_eps: float,
    temperature: float,
    residual_eps: float,
    has_pre_weight: bool,
):
    attn_f32 = attn.astype(jnp.float32)
    stream_f32 = stream.astype(jnp.float32)

    post_var = jnp.mean(attn_f32 * attn_f32, axis=1, keepdims=True)
    branch = attn_f32 * jax.lax.rsqrt(post_var + post_eps)
    branch = branch.astype(stream.dtype)
    if has_post_weight:
        branch = (branch * post_weight.astype(jnp.float32)).astype(stream.dtype)

    w = gate.astype(jnp.float32) / temperature
    beta = jax.nn.sigmoid(w)
    alpha_sq = jax.nn.sigmoid(-w) * (1.0 + beta)
    alpha = jnp.sqrt(jnp.maximum(alpha_sq, residual_eps))
    h = (alpha * stream_f32 + beta * branch.astype(jnp.float32)).astype(stream.dtype)

    h_f32 = h.astype(jnp.float32)
    pre_var = jnp.mean(h_f32 * h_f32, axis=1)
    pre_scale = jax.lax.rsqrt(pre_var + pre_eps)

    prepared = (h.astype(jnp.float32) * pre_scale[:, None]).astype(stream.dtype)
    if has_pre_weight:
        prepared = (prepared * pre_weight).astype(stream.dtype)
    return h, pre_scale, prepared


def _prelude_kernel(
    attn_ref,
    stream_ref,
    post_weight_ref,
    gate_ref,
    pre_weight_ref,
    h_ref,
    pre_ffn_ref,
    *,
    prepare_ffn: bool,
    **kwargs,
):
    h, pre_scale, prepared = _prelude_values(
        attn_ref[...],
        stream_ref[...],
        post_weight_ref[...],
        gate_ref[...],
        pre_weight_ref[...],
        **kwargs,
    )

    h_ref[...] = h
    if prepare_ffn:
        pre_ffn_ref[...] = prepared
    elif len(pre_ffn_ref.shape) == 1:
        pre_ffn_ref[...] = pre_scale
    else:
        pre_ffn_ref[...] = pre_scale[:, None]


def _prelude_router_kernel(
    attn_ref,
    stream_ref,
    post_weight_ref,
    gate_ref,
    pre_weight_ref,
    router_weight_ref,
    expert_bias_ref,
    h_ref,
    pre_ffn_ref,
    routing_weights_ref,
    routing_indices_ref,
    *,
    router_kwargs,
    **prelude_kwargs,
):
    h, _, prepared = _prelude_values(
        attn_ref[...],
        stream_ref[...],
        post_weight_ref[...],
        gate_ref[...],
        pre_weight_ref[...],
        **prelude_kwargs,
    )
    routing_weights, routing_indices = route_hidden_block(
        prepared,
        router_weight_ref[...],
        expert_bias_ref[...],
        **router_kwargs,
    )
    h_ref[...] = h
    pre_ffn_ref[...] = prepared
    topk = router_kwargs["topk"]
    routing_weights_ref[...] = routing_weights[:, :topk]
    routing_indices_ref[...] = routing_indices[:, :topk].astype(jnp.uint8)


@functools.partial(
    jax.jit,
    static_argnames=(
        "has_post_weight",
        "post_eps",
        "pre_eps",
        "temperature",
        "residual_eps",
        "prepare_ffn",
    ),
)
def fused_attention_moe_prelude(
    attn_out,
    residual_stream,
    post_weight,
    gate,
    *,
    has_post_weight: bool,
    post_eps: float,
    pre_eps: float,
    temperature: float = 0.3,
    residual_eps: float = 1e-3,
    pre_weight=None,
    prepare_ffn: bool = False,
):
    """Return the residual stream and scale or prepared rank-2 FFN input."""
    if attn_out.ndim != 2 or residual_stream.ndim != 2:
        raise ValueError(
            "fused_attention_moe_prelude supports rank-2 activations, got "
            f"{attn_out.shape} and {residual_stream.shape}"
        )
    if attn_out.shape != residual_stream.shape:
        raise ValueError(
            "attn_out and residual_stream must have the same shape, got "
            f"{attn_out.shape} and {residual_stream.shape}"
        )
    tokens, hidden = attn_out.shape
    if gate.shape != (hidden,):
        raise ValueError(f"gate must have shape {(hidden,)}, got {gate.shape}")
    if post_weight.shape != (hidden,):
        raise ValueError(
            f"post_weight must have shape {(hidden,)}, got {post_weight.shape}"
        )
    has_pre_weight = pre_weight is not None
    if has_pre_weight and pre_weight.shape != (hidden,):
        raise ValueError(
            f"pre_weight must have shape {(hidden,)}, got {pre_weight.shape}"
        )
    if pre_weight is None:
        pre_weight = jnp.zeros((hidden,), dtype=residual_stream.dtype)

    # A single 128/256-token tile amortizes the 8K-wide weight vectors.
    # The 512-token production tile needs 64.25 MiB including input/output
    # windows, exceeding TPU7x's hard 64 MiB cap; the sharded wrapper selects
    # the scattered FFN prepare boundary above 256 local tokens.
    if prepare_ffn and 128 <= tokens <= 256:
        block_tokens = tokens
    elif tokens < 16:
        block_tokens = tokens
    elif tokens < 64:
        block_tokens = align_to(tokens, 16)
    else:
        block_tokens = 64
    padded_tokens = align_to(tokens, block_tokens)
    if padded_tokens != tokens:
        attn_out = jnp.pad(attn_out, ((0, padded_tokens - tokens), (0, 0)))
        residual_stream = jnp.pad(
            residual_stream, ((0, padded_tokens - tokens), (0, 0))
        )

    activation_spec = pl.BlockSpec(
        (block_tokens, hidden), lambda token_block: (token_block, 0)
    )
    vector_spec = pl.BlockSpec((hidden,), lambda token_block: (0,))
    use_vector_scale = padded_tokens == block_tokens
    if prepare_ffn:
        pre_ffn_spec = activation_spec
        pre_ffn_shape = jax.ShapeDtypeStruct(
            (padded_tokens, hidden), residual_stream.dtype
        )
    elif use_vector_scale:
        pre_ffn_spec = pl.BlockSpec((block_tokens,), lambda token_block: (0,))
        pre_ffn_shape = jax.ShapeDtypeStruct((padded_tokens,), jnp.float32)
    else:
        pre_ffn_spec = pl.BlockSpec(
            (block_tokens, 1), lambda token_block: (token_block, 0)
        )
        pre_ffn_shape = jax.ShapeDtypeStruct((padded_tokens, 1), jnp.float32)

    h, scale = pl.pallas_call(
        functools.partial(
            _prelude_kernel,
            has_post_weight=has_post_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
            temperature=temperature,
            residual_eps=residual_eps,
            prepare_ffn=prepare_ffn,
            has_pre_weight=has_pre_weight,
        ),
        grid=(padded_tokens // block_tokens,),
        in_specs=(
            activation_spec,
            activation_spec,
            vector_spec,
            vector_spec,
            vector_spec,
        ),
        out_specs=(activation_spec, pre_ffn_spec),
        out_shape=(
            jax.ShapeDtypeStruct((padded_tokens, hidden), residual_stream.dtype),
            pre_ffn_shape,
        ),
        input_output_aliases={1: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="fused_attention_moe_prelude",
    )(attn_out, residual_stream, post_weight, gate, pre_weight)
    return h[:tokens], scale[:tokens]


@functools.partial(
    jax.jit,
    static_argnames=(
        "has_post_weight",
        "post_eps",
        "pre_eps",
        "temperature",
        "residual_eps",
        "topk",
        "router_output_multiplier",
        "route_scale",
        "division_eps",
        "score_normalization",
    ),
)
def fused_attention_moe_prelude_router(
    attn_out,
    residual_stream,
    post_weight,
    gate,
    pre_weight,
    router_weight,
    expert_bias,
    *,
    has_post_weight: bool,
    post_eps: float,
    pre_eps: float,
    topk: int,
    temperature: float = 0.3,
    residual_eps: float = 1e-3,
    router_output_multiplier: float = 1.0,
    route_scale: float = 1.0,
    division_eps: float = 0.0,
    score_normalization: str = "sigmoid",
):
    """Fuse the prelude and router, returning compact flat uint8 expert IDs."""
    if attn_out.ndim != 2 or residual_stream.ndim != 2:
        raise ValueError("fused prelude-router requires rank-2 activations")
    if attn_out.shape != residual_stream.shape:
        raise ValueError("attn_out and residual_stream must have the same shape")
    tokens, hidden = attn_out.shape
    num_experts, router_hidden = router_weight.shape
    if router_hidden != hidden:
        raise ValueError(
            f"router contraction mismatch: hidden={hidden}, weight={router_weight.shape}"
        )
    if hidden % 256:
        raise ValueError(
            f"fused prelude-router requires 256-aligned hidden, got {hidden}"
        )
    if post_weight.shape != (hidden,) or gate.shape != (hidden,):
        raise ValueError("post_weight and gate must match the hidden dimension")
    if pre_weight.shape != (hidden,):
        raise ValueError("pre_weight must match the hidden dimension")
    if expert_bias.shape != (num_experts,):
        raise ValueError("expert_bias must match the router expert dimension")
    if not 1 <= topk <= min(8, num_experts):
        raise ValueError(f"topk must be in [1, {min(8, num_experts)}], got {topk}")
    if score_normalization not in ("sigmoid", "softmax"):
        raise ValueError(f"unsupported score normalization {score_normalization!r}")

    # Match the standalone router's adaptive tile for small batches. Production
    # local token counts above 128 remain capped at 128: a 256-row tile exceeds
    # the full graph's scoped-VMEM limit with FP32 router weights.
    block_tokens = router_block_tokens(tokens)
    padded_tokens = align_to(tokens, block_tokens)
    padded_experts = align_to(num_experts, _ROUTER_OUTPUT_TILE)
    if padded_tokens != tokens:
        pad = ((0, padded_tokens - tokens), (0, 0))
        attn_out = jnp.pad(attn_out, pad)
        residual_stream = jnp.pad(residual_stream, pad)
    if padded_experts != num_experts:
        router_weight = jnp.pad(
            router_weight, ((0, padded_experts - num_experts), (0, 0))
        )
        expert_bias = jnp.pad(expert_bias, (0, padded_experts - num_experts))

    activation_spec = pl.BlockSpec(
        (block_tokens, hidden), lambda token_block: (token_block, 0)
    )
    vector_spec = pl.BlockSpec((hidden,), lambda token_block: (0,))
    router_weight_spec = pl.BlockSpec(
        (padded_experts, hidden), lambda token_block: (0, 0)
    )
    expert_spec = pl.BlockSpec((padded_experts,), lambda token_block: (0,))
    routing_weight_spec = pl.BlockSpec(
        (block_tokens, topk), lambda token_block: (token_block, 0)
    )
    routing_index_spec = pl.BlockSpec(
        (block_tokens, topk), lambda token_block: (token_block, 0)
    )
    outputs = pl.pallas_call(
        functools.partial(
            _prelude_router_kernel,
            has_post_weight=has_post_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
            temperature=temperature,
            residual_eps=residual_eps,
            has_pre_weight=True,
            router_kwargs={
                "num_experts": num_experts,
                "topk": topk,
                "router_output_multiplier": router_output_multiplier,
                "route_scale": route_scale,
                "division_eps": division_eps,
                "score_normalization": score_normalization,
            },
        ),
        grid=(padded_tokens // block_tokens,),
        in_specs=(
            activation_spec,
            activation_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            router_weight_spec,
            expert_spec,
        ),
        out_specs=(
            activation_spec,
            activation_spec,
            routing_weight_spec,
            routing_index_spec,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((padded_tokens, hidden), residual_stream.dtype),
            jax.ShapeDtypeStruct((padded_tokens, hidden), residual_stream.dtype),
            jax.ShapeDtypeStruct((padded_tokens, topk), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, topk), jnp.uint8),
        ),
        input_output_aliases={1: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="fused_attention_moe_prelude_router",
    )(
        attn_out,
        residual_stream,
        post_weight,
        gate,
        pre_weight,
        router_weight,
        expert_bias,
    )
    h, prepared, routing_weights, routing_indices = outputs
    return (
        h[:tokens],
        prepared[:tokens],
        routing_weights[:tokens],
        routing_indices[:tokens].reshape(tokens * topk),
    )


def build_fused_attention_moe_prelude_fn(
    mesh,
    *,
    token_spec=P(None, None),
    use_pallas: bool = False,
    prepare_ffn: bool = False,
):
    """Run the prelude with replicated or sequence-parallel token sharding."""

    token_axis = token_spec[0]
    token_axes = (
        ()
        if token_axis is None
        else token_axis if isinstance(token_axis, tuple) else (token_axis,)
    )
    token_shards = math.prod(mesh.shape[axis] for axis in token_axes)

    def _run(attn_out, residual_stream, post_weight, gate, pre_weight=None, **kwargs):
        if pre_weight is None:
            pre_weight = jnp.ones((attn_out.shape[-1],), dtype=residual_stream.dtype)
        local_tokens = attn_out.shape[0] // token_shards
        prepare_in_kernel = prepare_ffn and local_tokens <= 256

        def local_fn(
            local_attn, local_stream, local_post_weight, local_gate, local_pre_weight
        ):
            if use_pallas:
                h, pre_ffn = fused_attention_moe_prelude(
                    local_attn,
                    local_stream,
                    local_post_weight,
                    local_gate,
                    pre_weight=local_pre_weight if prepare_in_kernel else None,
                    prepare_ffn=prepare_in_kernel,
                    **kwargs,
                )
                if prepare_in_kernel:
                    return h, pre_ffn
                return h, pre_ffn.reshape(local_attn.shape[0], 1)
            has_post_weight = kwargs["has_post_weight"]
            h, scale = ref_fused_attention_moe_prelude(
                local_attn,
                local_stream,
                local_post_weight if has_post_weight else None,
                local_gate,
                post_eps=kwargs["post_eps"],
                pre_eps=kwargs["pre_eps"],
                temperature=kwargs.get("temperature", 0.3),
                residual_eps=kwargs.get("residual_eps", 1e-3),
            )
            if prepare_in_kernel:
                prepared = (h.astype(jnp.float32) * scale[:, None]).astype(h.dtype)
                prepared = (prepared * local_pre_weight).astype(h.dtype)
                return h, prepared
            return h, scale[:, None]

        h, pre_ffn = jax.shard_map(
            local_fn,
            mesh=mesh,
            in_specs=(token_spec, token_spec, P(), P(), P()),
            out_specs=(
                token_spec,
                token_spec if prepare_in_kernel else P(token_axis, None),
            ),
            check_vma=False,
        )(attn_out, residual_stream, post_weight, gate, pre_weight)
        if prepare_ffn and not prepare_in_kernel:
            prepared = (h.astype(jnp.float32) * pre_ffn).astype(h.dtype)
            prepared = (prepared * pre_weight).astype(h.dtype)
            return h, prepared
        return h, pre_ffn

    return _run


def build_fused_attention_moe_prelude_router_fn(mesh, *, token_spec):
    """Build the production SP wrapper for the combined prelude and router."""

    token_axis = token_spec[0]

    def _run(
        attn_out,
        residual_stream,
        post_weight,
        gate,
        pre_weight,
        router_weight,
        expert_bias,
        **kwargs,
    ):
        def local_fn(
            local_attn,
            local_stream,
            local_post_weight,
            local_gate,
            local_pre_weight,
            local_router_weight,
            local_expert_bias,
        ):
            return fused_attention_moe_prelude_router(
                local_attn,
                local_stream,
                local_post_weight,
                local_gate,
                local_pre_weight,
                local_router_weight,
                local_expert_bias,
                **kwargs,
            )

        return jax.shard_map(
            local_fn,
            mesh=mesh,
            in_specs=(token_spec, token_spec, P(), P(), P(), P(), P()),
            out_specs=(token_spec, token_spec, token_spec, P(token_axis)),
            check_vma=False,
        )(
            attn_out,
            residual_stream,
            post_weight,
            gate,
            pre_weight,
            router_weight,
            expert_bias,
        )

    return _run


__all__ = [
    "build_fused_attention_moe_prelude_fn",
    "build_fused_attention_moe_prelude_router_fn",
    "fused_attention_moe_prelude",
    "fused_attention_moe_prelude_router",
    "ref_fused_attention_moe_prelude",
]
