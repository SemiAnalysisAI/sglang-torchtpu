"""Pallas kernel for bs=1 decode MoE routing metadata.

With one real token every routed row is the same activation, so the permute is a
broadcast and only the expert-major metadata is needed. XLA emits that (plus the
pad / mask / local-id glue) as ~10 small ops; this is one kernel.

It runs on the vector unit: a slot's sorted position is the number of local
slots that sort before it (smaller expert id, or the same expert and an
earlier slot, which keeps the sort stable), and a group's size is a masked
count. A scalar SMEM version walked every local expert twice and took ~4 us at
224 local experts.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _route_meta_kernel(
    ids_row_ref,
    ids_col_ref,
    w_col_ref,
    ws_ref,
    gs_ref,
    *,
    k: int,
    n: int,
    offset: int,
    num_experts: int,
) -> None:
    e_row = ids_row_ref[...] - offset  # [1, k]: slot j along lanes
    e_col = ids_col_ref[...] - offset  # [k, 1]: slot i along sublanes
    ok_row = (e_row >= 0) & (e_row < num_experts)
    ok_col = (e_col >= 0) & (e_col < num_experts)
    j = jax.lax.broadcasted_iota(jnp.int32, (k, k), 1)
    i = jax.lax.broadcasted_iota(jnp.int32, (k, k), 0)
    ej = jnp.broadcast_to(e_row, (k, k))
    ei = jnp.broadcast_to(e_col, (k, k))
    before = jnp.broadcast_to(ok_row, (k, k)) & ((ej < ei) | ((ej == ei) & (j < i)))
    pos = jnp.sum(before.astype(jnp.int32), axis=1, keepdims=True)  # [k, 1]
    # ws[p] = w_i for the local slot i at sorted position p; zeros past them.
    p = jax.lax.broadcasted_iota(jnp.int32, (k, n), 1)
    place = (jnp.broadcast_to(pos, (k, n)) == p) & jnp.broadcast_to(ok_col, (k, n))
    w = jnp.broadcast_to(w_col_ref[...], (k, n))
    ws_ref[...] = jnp.sum(jnp.where(place, w, 0.0), axis=0, keepdims=True)
    e = jax.lax.broadcasted_iota(jnp.int32, (k, num_experts), 1)
    member = (jnp.broadcast_to(e_col, (k, num_experts)) == e) & jnp.broadcast_to(
        ok_col, (k, num_experts)
    )
    gs_ref[...] = jnp.sum(member.astype(jnp.int32), axis=0, keepdims=True)


@functools.partial(jax.jit, static_argnames=("n", "offset", "num_experts"))
def single_token_route_meta(
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    *,
    n: int,
    offset: int,
    num_experts: int,
) -> tuple[jax.Array, jax.Array]:
    """Expert-major metadata for the routed rows of one token.

    ``topk_ids`` / ``topk_weights`` are the token's [1, k] global ids and weights;
    ids in ``[offset, offset + num_experts)`` are local, every other id
    (non-local, or TopK's -1 for a padded token) is dropped. Returns f32[n]
    weights with the local rows as a sorted prefix (zeros after) and
    i32[num_experts] group sizes.
    """
    k = topk_ids.shape[1]
    ids = topk_ids.astype(jnp.int32)
    full = lambda shape: pl.BlockSpec(shape, lambda: (0,) * len(shape))
    ws, gs = pl.pallas_call(
        functools.partial(
            _route_meta_kernel, k=k, n=n, offset=offset, num_experts=num_experts
        ),
        out_shape=(
            jax.ShapeDtypeStruct((1, n), jnp.float32),
            jax.ShapeDtypeStruct((1, num_experts), jnp.int32),
        ),
        in_specs=[full((1, k)), full((k, 1)), full((k, 1))],
        out_specs=(full((1, n)), full((1, num_experts))),
        name=f"moe_route_meta_{k}_{n}_{num_experts}",
    )(ids, ids.reshape(k, 1), topk_weights.astype(jnp.float32).reshape(k, 1))
    return ws.reshape(n), gs.reshape(num_experts)
