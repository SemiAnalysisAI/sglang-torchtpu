# SPDX-License-Identifier: Apache-2.0
"""Destination-major SparseCore gather-reduce (``sc_ragged_gather_reduce_v3``).

Drop-in for ``ragged_gather_reduce_v2.ragged_gather_reduce``. Each SparseCore
subcore owns contiguous 64-token destination blocks: valid routes of a block
are compacted to the front, gathered from HBM 16 at a time, weighted and
scatter-added into an f32 VMEM accumulator, and the block is written to HBM
once as packed bf16. Invalid routes are never scattered (masked lanes), so
rows the GMMs never wrote cannot reach the output.

Per-block metadata (``route_counts``, 16 lanes per block): lane 0 is the valid
route count, lane 1 selects the unmasked two-routes-per-token fast path.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from sglang.srt.layers.moe.pallas.sparse_core import core_map_helper

_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS
_ROUTE_INDEX_BITS = 20
_ROUTE_INDEX_MASK = (1 << _ROUTE_INDEX_BITS) - 1


def _align_to(value: int, alignment: int) -> int:
    return pl.cdiv(value, alignment) * alignment


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(x.dtype)


def use_tensorcore_fallback(x: jax.Array) -> bool:
    """Small inputs: a TensorCore gather-reduce beats the SparseCore round trip.

    Same cutoff as ``ragged_gather_reduce_v2``.
    """
    tpu_info = pltpu.get_tpu_info()
    if (
        tpu_info.sparse_core is None
        or x.dtype != jnp.bfloat16
        or x.shape[-1] % tpu_info.num_lanes != 0
    ):
        return True
    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    return jnp.size(x) * dtype_bytes * 2 < tpu_info.vmem_capacity_bytes * 0.6


def _calculate_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * 2) == 0
        and hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and hidden_size // (num_column_partitions * 2 * num_lanes)
        >= preferred_num_stages
    ):
        num_column_partitions *= 2
    if num_column_partitions >= 2 and hidden_size <= 1024 * (
        num_column_partitions // 2
    ):
        num_column_partitions //= 2
    return num_column_partitions


def partition_counts(hidden_size: int, tpu_info) -> tuple[int, int]:
    """Column and row partition counts of SparseCore subcores for `hidden_size`."""
    sc_info = tpu_info.sparse_core
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_column_partitions = _calculate_num_column_partitions(
        hidden_size, num_cores, tpu_info.num_lanes
    )
    return num_column_partitions, num_cores // num_column_partitions


def token_block_alignment(hidden_size: int, tpu_info) -> int:
    """Destination-token granularity: one 64-token block per row partition."""
    return partition_counts(hidden_size, tpu_info)[1] * _TOKEN_BLOCK


def _main_kernel(
    x_hbm_ref: jax.Ref,
    route_metadata_hbm_ref: jax.Ref,
    route_weights_hbm_ref: jax.Ref,
    route_counts_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    accum_vmem_ref: jax.Ref,
    out_tile_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
    blocks_per_row_partition: int,
    topk: int,
):
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    assert num_simd_lanes == _TOKEN_SUBCHUNK

    core_id = jax.lax.axis_index((core_axis_name, subcore_axis_name))
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions
    col_size = accum_vmem_ref.shape[-1]
    col_start = col_partition_id * col_size
    metadata_block_size = topk * _TOKEN_BLOCK
    x_32b_hbm_ref = x_hbm_ref.bitcast(jnp.int32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

    def metadata_block_index(block_id):
        return (row_partition_id * blocks_per_row_partition + block_id,)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(blocks_per_row_partition,),
        in_specs=(
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((_TOKEN_SUBCHUNK,), metadata_block_index),
        ),
        out_specs=(),
    )
    def token_block_pipeline(
        route_metadata_ref, route_weights_ref, route_counts_ref, output_sem_ref
    ):
        block_id = pl.program_id(0)
        global_block_id = row_partition_id * blocks_per_row_partition + block_id
        block_meta = route_counts_ref[...]
        valid_route_count = block_meta[0]
        fixed_two_routes = block_meta[1] != 0
        num_route_chunks = pl.cdiv(valid_route_count, _TOKEN_SUBCHUNK)

        def zero_column(col_offset):
            col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
            zero = jnp.zeros((_TOKEN_SUBCHUNK,), jnp.float32)
            for token_row in range(_TOKEN_BLOCK):
                accum_vmem_ref[token_row, col_slice] = zero

        plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(zero_column)

        def route_indices_slice(route_chunk):
            start = route_chunk * _TOKEN_SUBCHUNK
            metadata = route_metadata_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
            return jnp.bitwise_and(metadata, _ROUTE_INDEX_MASK)

        @functools.partial(
            pltpu.emit_pipeline,
            grid=(num_route_chunks,),
            in_specs=pl.BlockSpec(
                (pl.Indirect(_TOKEN_SUBCHUNK), col_size),
                lambda route_chunk: (
                    jnp.bitwise_right_shift(route_indices_slice(route_chunk), 1),
                    col_partition_id,
                ),
            ),
            out_specs=(),
        )
        def route_pipeline(gather_ref):
            route_chunk = pl.program_id(0)
            metadata_start = route_chunk * _TOKEN_SUBCHUNK
            metadata_slice = pl.ds(metadata_start, _TOKEN_SUBCHUNK)
            route_metadata = route_metadata_ref[metadata_slice]
            source_indices = jnp.bitwise_and(route_metadata, _ROUTE_INDEX_MASK)
            route_destinations = jnp.bitwise_right_shift(
                route_metadata, _ROUTE_INDEX_BITS
            )
            route_weights = route_weights_ref[metadata_slice]

            def weighted_row(route_lane, col_slice):
                # x is gathered as int32 row pairs; take this route's bf16 half
                # (even row = high half) as f32 and apply its weight.
                value_i32 = gather_ref[route_lane, col_slice]
                shift = jnp.where(
                    jnp.bitwise_and(source_indices[route_lane], 1) == 0, 16, 0
                )
                shifted = jnp.bitwise_and(
                    jnp.left_shift(value_i32, shift), jnp.int32(-65536)
                )
                return plsc.bitcast(shifted, jnp.float32) * route_weights[route_lane]

            def scatter_add(route_lane, feature_indices, value, mask=None):
                destination_indices = jnp.full_like(
                    feature_indices, route_destinations[route_lane]
                )
                kwargs = {} if mask is None else {"mask": mask}
                plsc.addupdate_scatter(
                    accum_vmem_ref,
                    (destination_indices, feature_indices),
                    value,
                    **kwargs,
                )

            def fixed_two_route_column_loop(col_offset):
                # Every lane valid and lanes (2i, 2i+1) share a destination:
                # sum each pair, then one unmasked scatter per pair.
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(
                    _TOKEN_SUBCHUNK, dtype=jnp.int32
                )
                values = [weighted_row(r, col_slice) for r in range(_TOKEN_SUBCHUNK)]
                for r in range(0, _TOKEN_SUBCHUNK, 2):
                    scatter_add(r, feature_indices, values[r] + values[r + 1])

            def generic_column_loop(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(
                    _TOKEN_SUBCHUNK, dtype=jnp.int32
                )
                route_batch_size = 4
                for batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
                    lanes = range(batch_start, batch_start + route_batch_size)
                    values = [weighted_row(r, col_slice) for r in lanes]
                    for r, value in zip(lanes, values):
                        # Lanes past the block's valid count are never added.
                        route_valid = metadata_start + r < valid_route_count
                        mask = jnp.broadcast_to(route_valid, (_TOKEN_SUBCHUNK,))
                        scatter_add(r, feature_indices, value, mask=mask)

            @pl.when(fixed_two_routes)
            def run_fixed_two_route_path():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(
                    fixed_two_route_column_loop
                )

            @pl.when(jnp.logical_not(fixed_two_routes))
            def run_generic_path():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(
                    generic_column_loop
                )

        route_pipeline(x_32b_hbm_ref)

        for token_subchunk_pair in range(_NUM_TOKEN_SUBCHUNKS // 2):
            copies = []
            for output_buffer in range(2):
                token_subchunk = token_subchunk_pair * 2 + output_buffer
                accum_row_start = token_subchunk * _TOKEN_SUBCHUNK

                def cast_column(col_offset):
                    col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                    for token_lane in range(0, _TOKEN_SUBCHUNK, 2):
                        accum_row = accum_row_start + token_lane
                        packed_bf16 = plsc.pack(
                            accum_vmem_ref[accum_row, col_slice],
                            accum_vmem_ref[accum_row + 1, col_slice],
                            format=plsc.PackFormat.INTERLEAVED,
                        )
                        out_tile_vmem_ref[output_buffer, token_lane // 2, col_slice] = (
                            plsc.bitcast(packed_bf16, jnp.uint32)
                        )

                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(cast_column)

                output_row = (
                    global_block_id * _TOKEN_BLOCK + token_subchunk * _TOKEN_SUBCHUNK
                )
                output_row_packed = pl.multiple_of(
                    output_row // 2, _TOKEN_SUBCHUNK // 2
                )
                copy = pltpu.make_async_copy(
                    out_tile_vmem_ref.at[output_buffer],
                    out_32b_hbm_ref.at[
                        pl.ds(output_row_packed, _TOKEN_SUBCHUNK // 2),
                        pl.ds(col_start, col_size),
                    ],
                    output_sem_ref.at[output_buffer],
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

    token_block_pipeline(
        route_metadata_hbm_ref,
        route_weights_hbm_ref,
        route_counts_hbm_ref,
        scratches=(sem_ref,),
    )


@functools.partial(jax.jit, static_argnames=("reduce_group_size",))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Destination-major SparseCore gather-reduce for MoE output combine."""
    if use_tensorcore_fallback(x):
        return _fallback_implementation(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )
    sc_info = pltpu.get_tpu_info().sparse_core

    input_size = indices.size
    if input_size % reduce_group_size != 0:
        raise ValueError(f"{input_size=} must be divisible by {reduce_group_size=}")
    if x.shape[0] % 2:
        x = jnp.pad(x, ((0, 1), (0, 0)))
    num_tokens = input_size // reduce_group_size
    hidden_size = x.shape[-1]
    if x.shape[0] > 1 << _ROUTE_INDEX_BITS:
        raise ValueError(
            f"destination-major gather-reduce supports at most "
            f"{1 << _ROUTE_INDEX_BITS} source rows, got {x.shape[0]}"
        )
    num_column_partitions, num_row_partitions = partition_counts(
        hidden_size, pltpu.get_tpu_info()
    )

    token_alignment = num_row_partitions * _TOKEN_BLOCK
    padded_num_tokens = _align_to(num_tokens, token_alignment)
    token_padding = padded_num_tokens - num_tokens

    indices_2d = indices.reshape(num_tokens, reduce_group_size)
    weights_2d = topk_weights.reshape(num_tokens, reduce_group_size)
    valid_2d = valid_rows_mask.reshape(num_tokens, reduce_group_size)
    if token_padding:
        indices_2d = jnp.pad(indices_2d, ((0, token_padding), (0, 0)))
        weights_2d = jnp.pad(weights_2d, ((0, token_padding), (0, 0)))
        valid_2d = jnp.pad(valid_2d, ((0, token_padding), (0, 0)))

    num_blocks = padded_num_tokens // _TOKEN_BLOCK
    metadata_block_size = _TOKEN_BLOCK * reduce_group_size
    indices_blocks = indices_2d.reshape(num_blocks, metadata_block_size)
    valid_blocks = valid_2d.reshape(num_blocks, metadata_block_size)
    destination_template = jnp.repeat(
        jnp.arange(_TOKEN_BLOCK, dtype=jnp.int32), reduce_group_size
    )
    destination_blocks = jnp.broadcast_to(destination_template, indices_blocks.shape)

    route_order = jnp.argsort(~valid_blocks, axis=1, stable=True)
    compact_valid = jnp.take_along_axis(valid_blocks, route_order, axis=1)
    route_indices = jnp.take_along_axis(indices_blocks, route_order, axis=1)
    route_destinations = jnp.take_along_axis(destination_blocks, route_order, axis=1)
    weights_blocks = weights_2d.reshape(num_blocks, metadata_block_size)
    route_weights = jnp.take_along_axis(weights_blocks, route_order, axis=1)
    route_indices = jnp.where(compact_valid, route_indices, 0).astype(jnp.int32)
    route_destinations = jnp.where(compact_valid, route_destinations, 0).astype(
        jnp.int32
    )
    route_metadata = jnp.bitwise_or(
        route_indices,
        jnp.left_shift(route_destinations, _ROUTE_INDEX_BITS),
    ).reshape(-1)
    route_weights = jnp.where(compact_valid, route_weights, 0).reshape(-1)
    route_weights = route_weights.astype(jnp.float32)
    route_counts = jnp.sum(valid_blocks, axis=1, dtype=jnp.int32)
    paired_destinations = route_destinations[:, : 2 * _TOKEN_BLOCK].reshape(
        num_blocks, _TOKEN_BLOCK, 2
    )
    # Fast path: exactly two valid routes per token, adjacent after compaction,
    # so route pairs can be summed before one unmasked scatter.
    fixed_two_route_blocks = jnp.logical_and(
        route_counts == 2 * _TOKEN_BLOCK,
        jnp.all(paired_destinations[:, :, 0] == paired_destinations[:, :, 1], axis=1),
    )
    route_counts = (
        jnp.zeros((num_blocks, _TOKEN_SUBCHUNK), jnp.int32)
        .at[:, 0]
        .set(route_counts)
        .at[:, 1]
        .set(fixed_two_route_blocks.astype(jnp.int32))
        .reshape(-1)
    )

    col_size = hidden_size // num_column_partitions
    blocks_per_row_partition = num_blocks // num_row_partitions
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = core_map_helper.kernel(
        functools.partial(
            _main_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_partitions=num_row_partitions,
            num_column_partitions=num_column_partitions,
            blocks_per_row_partition=blocks_per_row_partition,
            topk=reduce_group_size,
        ),
        out_type=jax.ShapeDtypeStruct((padded_num_tokens, hidden_size), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(
            pltpu.VMEM((_TOKEN_BLOCK, col_size), jnp.float32),
            pltpu.VMEM((2, _TOKEN_SUBCHUNK // 2, col_size), jnp.uint32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v3",
    )(x, route_metadata, route_weights, route_counts)
    return out[:num_tokens]
