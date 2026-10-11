# SPDX-License-Identifier: Apache-2.0
"""Single-kernel sigmoid top-k router for a few tokens (decode or a verify window).

Replaces biased_grouped_topk_impl's XLA chain for one expert group (sigmoid,
bias add, TopK, gather, renormalize, optional routed scaling; ~12 ops per MoE
layer) with one Pallas call that walks the rows in a static loop. Top-k is rank based: one [E, E] comparison block
gives every expert's rank, so the kernel needs a handful of reductions instead
of k dependent arg-max rounds. Equal scores rank the lower index first, the
order torch.topk / XLA TopK select in.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_RENORMALIZE_SUM_EPSILON = 1e-20
_LANES = 128
# Rows are unrolled; a speculative verify window is at most this many tokens.
_MAX_TOKENS = 16


def supports(tokens: int, num_experts: int, topk: int) -> bool:
    return tokens <= _MAX_TOKENS and topk <= min(num_experts, _LANES)


def _kernel(logits_ref, bias_ref, w_ref, ids_ref, *, topk, scale):
    for row in range(logits_ref.shape[0]):
        w, ids = _route_row(
            logits_ref[row : row + 1, :], bias_ref[...], topk=topk, scale=scale
        )
        w_ref[row : row + 1, :] = w
        ids_ref[row : row + 1, :] = ids


def _route_row(logits, bias, *, topk, scale):
    width = logits.shape[1]
    scores = jax.nn.sigmoid(logits.astype(jnp.float32))
    choice = scores + bias.astype(jnp.float32)

    # Column copies of the [1, E] rows via a diagonal reduction (no transpose).
    rows = jax.lax.broadcasted_iota(jnp.int32, (width, width), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (width, width), 1)
    diag = rows == cols

    def column(row):
        return jnp.sum(
            jnp.where(diag, jnp.broadcast_to(row, (width, width)), 0.0),
            axis=1,
            keepdims=True,
        )

    choice_sq = jnp.broadcast_to(choice, (width, width))
    choice_col = column(choice)
    beats = (choice_sq > choice_col) | ((choice_sq == choice_col) & (cols < rows))
    rank = jnp.sum(beats.astype(jnp.int32), axis=1, keepdims=True)  # [E, 1]

    # Slot s holds the expert of rank s.
    slots = jax.lax.broadcasted_iota(jnp.int32, (width, _LANES), 1)
    pick = (rank == slots) & (slots < topk)
    expert = jax.lax.broadcasted_iota(jnp.int32, (width, _LANES), 0)
    ids = jnp.sum(jnp.where(pick, expert, 0), axis=0, keepdims=True)
    w = jnp.sum(jnp.where(pick, column(scores), 0.0), axis=0, keepdims=True)
    w = w / (jnp.sum(w, axis=-1, keepdims=True) + _RENORMALIZE_SUM_EPSILON)
    if scale != 1.0:
        w = w * scale
    return w[:, :topk], ids[:, :topk]


def router_topk_small_m(
    logits: jax.Array, bias: jax.Array, topk: int, scale: float
) -> tuple[jax.Array, jax.Array]:
    """logits [M, E], bias [E] -> (weights f32 [M, k], ids i32 [M, k]).

    Blocks span the full (unpadded) expert dim, so the call adds no pad or
    slice ops around the kernel.
    """
    tokens, num_experts = logits.shape
    return pl.pallas_call(
        functools.partial(_kernel, topk=topk, scale=scale),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, topk), jnp.float32),
            jax.ShapeDtypeStruct((tokens, topk), jnp.int32),
        ),
        name=f"router_topk_rank_{num_experts}_{topk}",
    )(logits, bias.reshape(1, num_experts))
