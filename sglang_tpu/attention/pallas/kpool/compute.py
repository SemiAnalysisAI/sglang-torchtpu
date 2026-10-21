# SPDX-License-Identifier: Apache-2.0
"""Tile arithmetic shared by the short and long query schedules."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def _hadamard128(value):
    """Sylvester-order butterfly in f32; rounding is checked at the accuracy gate."""
    lane = jnp.arange(128, dtype=jnp.int32)[None, :]
    for step in (1, 2, 4, 8, 16, 32, 64):
        low = (lane & step) == 0
        partner = jnp.where(
            low,
            pltpu.roll(value, 128 - step, axis=1),
            pltpu.roll(value, step, axis=1),
        )
        value = partner + jnp.where(low, value, -value)
    return value


def pool_rows(raw, ape):
    """Pool a small aligned raw tile without expanding the live set."""
    pools = raw.shape[0] // 4
    raw = raw.astype(jnp.float32)
    keys = raw[:, :128].reshape((pools, 4, 128))
    scores = raw[:, 128:].reshape((pools, 4, 128)) + ape[None, :, :]
    scores -= jnp.max(scores, axis=1, keepdims=True)
    weight = jnp.exp(scores)
    weight /= jnp.sum(weight, axis=1, keepdims=True)
    return jnp.sum(weight * keys, axis=1).astype(jnp.bfloat16)


def quantize_pools(pooled):
    """Pool BF16 -> rotate F32 -> FP8 keys and scale bytes."""
    rotated = _hadamard128(pooled.astype(jnp.float32)) * (128**-0.5)
    # Match the pinned reference executable, which quantizes F32 rotation.
    # An explicit BF16 round trip here changes FP8 rounding at ties.
    amax = jnp.maximum(jnp.max(jnp.abs(rotated), axis=1, keepdims=True), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0))
    q = jnp.clip(rotated / jnp.exp2(exponent), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    scale_bytes = jnp.where(
        jnp.arange(128)[None, :] == 0,
        (exponent + 127).astype(jnp.uint8),
        jnp.uint8(0),
    )
    return q, scale_bytes


def pack_pools(pooled):
    """Pack four adjacent pool rows into each physical destination quad."""
    q, scale_bytes = quantize_pools(pooled)
    return jnp.concatenate(
        (pltpu.bitcast(q, jnp.uint32), pltpu.bitcast(scale_bytes, jnp.uint32)),
        axis=1,
    )


def pack_selected_pools(raw_tiles, ape, mask, pools_per_tile):
    """Compute at most ceil(Q/4) completed pools in each aligned raw tile.

    Select whole four-row pools before softmax. Packing keeps each tile's
    records in its own destination quad, even when tiles belong to different
    requests. The caller's byte mask excludes incomplete and absent pools.
    """
    tiles = raw_tiles.shape[0]
    raw = raw_tiles.astype(jnp.float32).reshape((tiles, 4, 4, 256))
    first = jnp.where(
        mask & jnp.uint32(0xFF) != 0,
        0,
        jnp.where(
            mask & jnp.uint32(0xFF00) != 0,
            1,
            jnp.where(mask & jnp.uint32(0xFF0000) != 0, 2, 3),
        ),
    )
    packed = jnp.zeros((tiles, 256), jnp.uint32)
    for offset in range(pools_per_tile):
        pool = first + offset
        members = jnp.sum(
            jnp.where(
                jnp.arange(4)[None, :, None, None] == pool[:, :, None, None],
                raw,
                jnp.float32(0),
            ),
            axis=1,
        )
        pooled = pool_rows(members.reshape((tiles * 4, 256)), ape)
        q, scale_bytes = quantize_pools(pooled)
        # Expanding each byte prevents adjacent requests from being packed
        # together. The masked merge remains the pipeline's responsibility.
        words = jnp.concatenate(
            (
                jax.lax.bitcast_convert_type(q, jnp.uint8).astype(jnp.uint32),
                scale_bytes.astype(jnp.uint32),
            ),
            axis=1,
        )
        shift = (jnp.minimum(pool, 3) * 8).astype(jnp.uint32)
        packed |= jnp.where(pool < 4, words << shift, jnp.uint32(0))
    return packed
