# SPDX-License-Identifier: Apache-2.0
"""Fused Llama4x router matmul, scoring, and top-k selection for TPU."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    align_to,
    cost_estimate_metadata,
    estimate_activation_compute_cost,
)

_BLOCK_K = 256
_OUTPUT_TILE = 128
_MAX_TOPK = 8


def estimate_moe_routing_compute_cost(
    *, tokens: int, num_experts: int, topk: int, score_normalization: str
) -> tuple[int, int]:
    """Return floating-point operations and transcendentals for routing.

    Comparisons and integer index operations in top-k selection are excluded.
    Reductions count one floating-point addition per combined element.
    """
    score_values = tokens * num_experts
    # Logit scaling and expert-bias addition are common to both modes.
    flops = 2 * score_values
    if score_normalization == "sigmoid":
        sigmoid_flops, transcendentals = estimate_activation_compute_cost(
            "sigmoid", score_values
        )
        flops += sigmoid_flops
    elif score_normalization == "softmax":
        # Subtract max, sum the exponentials, and divide by the row sum.
        flops += 3 * score_values - tokens
        transcendentals = score_values
    else:
        raise ValueError(
            f"unsupported router score normalization {score_normalization!r}"
        )
    # Each selected score is reduced across experts, then the selected weights
    # are summed, divided by the sum, and multiplied by route_scale.
    flops += tokens * topk * (num_experts - 1)
    flops += 3 * tokens * topk
    return flops, transcendentals


def get_fused_moe_router_cost_estimate(
    *,
    tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    block_tokens: int,
    hidden_dtype,
    router_weight_dtype,
    expert_bias_dtype,
    score_normalization: str,
) -> pl.CostEstimate:
    """Estimate the padded static work of ``fused_moe_router``."""
    routing_flops, transcendentals = estimate_moe_routing_compute_cost(
        tokens=tokens,
        num_experts=num_experts,
        topk=topk,
        score_normalization=score_normalization,
    )
    flops = 2 * tokens * hidden_size * num_experts + routing_flops
    num_token_blocks = tokens // block_tokens
    bytes_accessed = (
        tokens * hidden_size * jnp.dtype(hidden_dtype).itemsize
        + num_token_blocks
        * num_experts
        * hidden_size
        * jnp.dtype(router_weight_dtype).itemsize
        + num_token_blocks * num_experts * jnp.dtype(expert_bias_dtype).itemsize
        + tokens
        * _OUTPUT_TILE
        * (jnp.dtype(jnp.float32).itemsize + jnp.dtype(jnp.int32).itemsize)
    )
    return pl.CostEstimate(
        flops=int(flops),
        bytes_accessed=int(bytes_accessed),
        transcendentals=int(transcendentals),
    )


def ref_fused_moe_router(
    hidden_states,
    router_weight,
    expert_bias,
    *,
    topk: int,
    router_output_multiplier: float = 1.0,
    route_scale: float = 1.0,
    division_eps: float = 0.0,
    score_normalization: str = "sigmoid",
):
    """Reference implementation matching Llama4x routing semantics."""
    logits = hidden_states.astype(jnp.float32) @ router_weight.astype(jnp.float32).T
    logits = logits * router_output_multiplier
    if score_normalization == "sigmoid":
        scores = jax.nn.sigmoid(logits)
    elif score_normalization == "softmax":
        scores = jax.nn.softmax(logits, axis=-1)
    else:
        raise ValueError(
            f"unsupported router score normalization {score_normalization!r}"
        )
    _, indices = jax.lax.top_k(scores + expert_bias.astype(jnp.float32), topk)
    weights = jnp.take_along_axis(scores, indices, axis=1)
    weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + division_eps)
    return weights * route_scale, indices.astype(jnp.int32)


def route_hidden_block(
    hidden,
    weight,
    bias,
    *,
    num_experts: int,
    topk: int,
    router_output_multiplier: float,
    route_scale: float,
    division_eps: float,
    score_normalization: str,
):
    """Route one TPU-resident token block and return padded top-k outputs."""
    logits = jax.lax.dot_general(
        hidden.astype(jnp.float32),
        weight.astype(jnp.float32),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    expert_ids = jnp.arange(logits.shape[1], dtype=jnp.int32)[jnp.newaxis, :]
    valid_experts = expert_ids < num_experts
    logits = logits * router_output_multiplier
    logits = jnp.where(valid_experts, logits, -jnp.inf)
    if score_normalization == "sigmoid":
        scores = jax.nn.sigmoid(logits)
    else:
        scores = jax.nn.softmax(logits, axis=1)

    candidates = jnp.where(
        valid_experts,
        scores + bias.astype(jnp.float32)[jnp.newaxis, :],
        -jnp.inf,
    )
    selected_weights = []
    selected_indices = []
    for _ in range(topk):
        index = jnp.argmax(candidates, axis=1).astype(jnp.int32)
        mask = expert_ids == index[:, jnp.newaxis]
        selected_indices.append(index)
        selected_weights.append(jnp.sum(jnp.where(mask, scores, 0.0), axis=1))
        candidates = jnp.where(mask, -jnp.inf, candidates)

    weights = jnp.stack(selected_weights, axis=1)
    indices = jnp.stack(selected_indices, axis=1)

    weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + division_eps)
    weights = weights * route_scale
    return (
        jnp.pad(weights, ((0, 0), (0, _OUTPUT_TILE - topk))),
        jnp.pad(indices, ((0, 0), (0, _OUTPUT_TILE - topk))),
    )


def _router_kernel(
    hidden_ref,
    weight_ref,
    bias_ref,
    weights_ref,
    indices_ref,
    **kwargs,
):
    weights, indices = route_hidden_block(
        hidden_ref[...], weight_ref[...], bias_ref[...], **kwargs
    )
    weights_ref[...] = weights
    indices_ref[...] = indices


def router_block_tokens(tokens: int) -> int:
    for block in (8, 16, 32, 64, 128):
        if tokens <= block:
            return block
    return 128


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "router_output_multiplier",
        "route_scale",
        "division_eps",
        "score_normalization",
    ),
)
def fused_moe_router(
    hidden_states,
    router_weight,
    expert_bias,
    *,
    topk: int,
    router_output_multiplier: float = 1.0,
    route_scale: float = 1.0,
    division_eps: float = 0.0,
    score_normalization: str = "sigmoid",
):
    """Return routing weights and IDs without materializing router logits."""
    if hidden_states.ndim != 2 or router_weight.ndim != 2:
        raise ValueError(
            "hidden_states and router_weight must be rank 2, got "
            f"{hidden_states.shape} and {router_weight.shape}"
        )
    tokens, hidden_size = hidden_states.shape
    num_experts, weight_hidden_size = router_weight.shape
    if hidden_size != weight_hidden_size:
        raise ValueError(
            f"router contraction mismatch: {hidden_size} and {weight_hidden_size}"
        )
    if expert_bias.shape != (num_experts,):
        raise ValueError(
            f"expert_bias must have shape {(num_experts,)}, got {expert_bias.shape}"
        )
    if not 1 <= topk <= min(_MAX_TOPK, num_experts):
        raise ValueError(
            f"topk must be in [1, {min(_MAX_TOPK, num_experts)}], got {topk}"
        )
    if score_normalization not in ("sigmoid", "softmax"):
        raise ValueError(
            f"unsupported router score normalization {score_normalization!r}"
        )

    block_m = router_block_tokens(tokens)
    padded_tokens = align_to(tokens, block_m)
    padded_hidden = align_to(hidden_size, _BLOCK_K)
    padded_experts = align_to(num_experts, 128)
    hidden_states = jnp.pad(
        hidden_states,
        ((0, padded_tokens - tokens), (0, padded_hidden - hidden_size)),
    )
    router_weight = jnp.pad(
        router_weight,
        ((0, padded_experts - num_experts), (0, padded_hidden - hidden_size)),
    )
    expert_bias = jnp.pad(expert_bias, (0, padded_experts - num_experts))
    cost_estimate = get_fused_moe_router_cost_estimate(
        tokens=padded_tokens,
        hidden_size=padded_hidden,
        num_experts=padded_experts,
        topk=topk,
        block_tokens=block_m,
        hidden_dtype=hidden_states.dtype,
        router_weight_dtype=router_weight.dtype,
        expert_bias_dtype=expert_bias.dtype,
        score_normalization=score_normalization,
    )
    output_spec = pl.BlockSpec(
        (block_m, _OUTPUT_TILE), lambda token_block: (token_block, 0)
    )
    weights, indices = pl.pallas_call(
        functools.partial(
            _router_kernel,
            num_experts=num_experts,
            topk=topk,
            router_output_multiplier=router_output_multiplier,
            route_scale=route_scale,
            division_eps=division_eps,
            score_normalization=score_normalization,
        ),
        grid=(padded_tokens // block_m,),
        in_specs=(
            pl.BlockSpec(
                (block_m, padded_hidden),
                lambda token_block: (token_block, 0),
            ),
            pl.BlockSpec(
                (padded_experts, padded_hidden),
                lambda token_block: (0, 0),
            ),
            pl.BlockSpec((padded_experts,), lambda token_block: (0,)),
        ),
        out_specs=(output_spec, output_spec),
        out_shape=(
            jax.ShapeDtypeStruct((padded_tokens, _OUTPUT_TILE), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, _OUTPUT_TILE), jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        cost_estimate=cost_estimate,
        metadata=cost_estimate_metadata(
            flops_basis="padded static router matmul and routing arithmetic",
            bytes_accessed_basis=(
                "modeled HBM reads and writes, including repeated weight and bias reads"
            ),
        ),
        name="fused_moe_router",
    )(hidden_states, router_weight, expert_bias)
    return weights[:tokens, :topk], indices[:tokens, :topk]


def build_fused_moe_router_fn(mesh, token_spec):
    """Build a router preserving the caller's replicated or sharded token axis."""

    def _run(hidden_states, router_weight, expert_bias, **kwargs):
        def _local(local_hidden, local_weight, local_bias):
            return fused_moe_router(local_hidden, local_weight, local_bias, **kwargs)

        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(token_spec, P(), P()),
            out_specs=(token_spec, token_spec),
            check_vma=False,
        )(hidden_states, router_weight, expert_bias)

    return _run


__all__ = [
    "build_fused_moe_router_fn",
    "estimate_moe_routing_compute_cost",
    "fused_moe_router",
    "get_fused_moe_router_cost_estimate",
    "ref_fused_moe_router",
    "router_block_tokens",
]
