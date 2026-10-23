# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v4/rope.py
# Revision: bdb38700dd25959c029c7d77f00cee91ccd3a0bc
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(pltpu.BufferedRef):
    """Per-token gather of ``cos_sin_cache`` rows into a ``(tile_n, rotary_dim)``"""

    tile_n: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls, *, spec: pl.BlockSpec, dtype: jnp.dtype, buffer_count: int, tile_n: int
    ) -> "CosSinRef":
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=False,
        )
        return cls(
            tile_n=tile_n,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ) -> None:
        cos_sin_cache_ref, positions_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        pid = grid_indices[0]

        dest_ref = self.window_ref.at[slot]  # (tile_n, rotary_dim)
        width = self.window_ref.shape[-1]

        @pl.loop(0, self.tile_n, unroll=True)
        def _(i):
            position = positions_ref[pid * self.tile_n + i]
            src = cos_sin_cache_ref.at[position, pl.ds(0, width)]
            pltpu.make_async_copy(src, dest_ref.at[i, :], sem).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ) -> None:
        del src_ref, grid_indices
        slot = self.current_wait_in_slot
        vmem_ref = self.window_ref.at[slot]
        pltpu.make_async_copy(vmem_ref, vmem_ref, self.sem_recvs.at[slot]).wait()


def _lane_gather(operand: jax.Array, indices: jax.Array) -> jax.Array:
    """``out[b..., i] = operand[b..., indices[i]]`` over the last axis.

    The leading axes come along as batch dimensions. ``indices`` is a constant
    1-D array.
    """
    rank = operand.ndim
    lanes = indices.shape[0]
    coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
    batching = tuple(range(rank - 1))
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(rank - 1,),
        start_index_map=(rank - 1,),
        operand_batching_dims=batching,
        start_indices_batching_dims=batching,
    )
    return jax.lax.gather(
        operand,
        coords,
        dimension_numbers=dimension_numbers,
        slice_sizes=(1,) * rank,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _cos_sin_lanes(
    cos_sin: jax.Array, *, rotary_dim: int, inverse: bool
) -> tuple[jax.Array, jax.Array]:
    """The reference's ``np.split`` + ``np.repeat(..., 2)``, over a whole block.

    ``cos_sin`` is one packed ``[cos | sin]`` row per token. The reference splits
    it in half, repeats each half so both channels of a pair share a frequency,
    rotates ``x[..., -rotary_dim:]`` and concatenates the NoPE part back on.

    A kernel block is ``LANE`` lanes wide and the roped channels are its tail, so
    rather than slicing them out we widen cos/sin to the full block and hand the
    NoPE lanes the identity rotation -- multiplying by ``cos = 1``, ``sin = 0``
    is exactly the reference's pass-through concatenation::

        lane      0 ... 63 | 64  65  66  67 ...   (LANE 128, rotary_dim 64)
        channel   -- NoPE -|  0   1   2   3
        cos       1 ...  1 | c0  c0  c1  c1       <- np.repeat(cos, 2)
        sin       0 ...  0 | s0  s0  s1  s1
    """
    half = rotary_dim // 2
    channel = jnp.arange(LANE) - (LANE - rotary_dim)  # < 0 on the NoPE lanes
    is_rot = channel >= 0
    # ``np.repeat(..., 2)``: channels 2k and 2k+1 both read frequency k. Clamped
    # to keep the gather in bounds on the NoPE lanes, replaced just below.
    freq = jnp.maximum(channel, 0) // 2

    cos = _lane_gather(cos_sin, freq)  # np.split(...)[0], repeated
    sin = _lane_gather(cos_sin, half + freq)  # np.split(...)[1], repeated
    if inverse:
        sin = -sin
    return jnp.where(is_rot, cos, 1.0), jnp.where(is_rot, sin, 0.0)


def _rotate_gptj(x: jax.Array) -> jax.Array:
    """The reference's ``rotated``: ``[-x1, x0, -x3, x2, ...]`` along the lanes.

    ``np.stack([-x[..., 1::2], x[..., 0::2]], -1).reshape(...)`` says, lane by
    lane, "take the partner lane ``i ^ 1``, negated on the even ones" -- a
    constant-index lane shuffle and a sign, no reshape. Lane parity may be read
    for channel parity because ``LANE - rotary_dim`` is even.
    """
    lane = jnp.arange(LANE)
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
    return _lane_gather(x, lane ^ 1) * sign
