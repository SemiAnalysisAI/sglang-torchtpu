# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Collective SP routing metadata for the fused-RS MoE path.

The fused-RS kernels consume one globally stable, expert-major assignment list.
With sequence parallelism the router IDs start token-sharded, so the generic
path first all-gathers all IDs and then sorts all ``tokens * topk`` rows on
every rank. This kernel keeps the sort local:

1. Sort each rank's contiguous assignment shard by ``(expert, global_row)``.
2. Hypercube-gather the sorted runs and their per-expert counts in ``log2(tp)`` rounds.
3. Merge the runs by computing every row's final expert-major destination.

Because token shards are contiguous, concatenating the rank-local expert
segments in rank order is exactly the same stable order as a global argsort.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import local_barrier
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    cdiv,
    cost_estimate_metadata,
    logical_comm_bytes,
)

_LANES = 128

# With recursive-doubling exchange, TP16 device-time sweeps beat the XLA
# all-gather + argsort path from the smallest supported shape: 2,048 global
# assignments (T=256, top-k=8).
_MIN_PROFITABLE_GLOBAL_ASSIGNMENTS = 2048


def get_fused_replicated_moe_dispatch_cost_estimate(
    *, assignments: int, num_experts: int, indices_dtype
) -> pl.CostEstimate:
    """Estimate HBM traffic for replicated integer routing metadata."""
    input_bytes = assignments * jnp.dtype(indices_dtype).itemsize
    output_int32_values = 3 * assignments + num_experts + 3 * _LANES
    return pl.CostEstimate(
        flops=0,
        bytes_accessed=int(
            input_bytes + output_int32_values * jnp.dtype(jnp.int32).itemsize
        ),
        transcendentals=0,
    )


def get_fused_sp_moe_dispatch_cost_estimate(
    *, local_assignments: int, num_experts: int, ep_size: int, indices_dtype
) -> pl.CostEstimate:
    """Estimate HBM and logical all-gather payloads for SP routing metadata."""
    run_values = local_assignments + num_experts
    global_assignments = local_assignments * ep_size
    input_bytes = local_assignments * jnp.dtype(indices_dtype).itemsize
    gathered_run_bytes = ep_size * run_values * jnp.dtype(jnp.int32).itemsize
    output_int32_values = (
        ep_size * run_values + 3 * global_assignments + num_experts + 3 * _LANES
    )
    run_bytes = run_values * jnp.dtype(jnp.int32).itemsize
    # Recursive doubling forwards 1, 2, 4, ... packed-run blocks from the HBM
    # result buffer. Across all stages that is ep_size - 1 extra reads of one
    # rank's packed-run payload, in addition to the final full reread.
    forwarding_read_bytes = (ep_size - 1) * run_bytes
    remote_bytes_transferred = (
        logical_comm_bytes(run_bytes, ep_size * run_bytes) if ep_size > 1 else 0
    )
    return pl.CostEstimate(
        flops=0,
        bytes_accessed=int(
            input_bytes
            + output_int32_values * jnp.dtype(jnp.int32).itemsize
            + gathered_run_bytes
            + forwarding_read_bytes
        ),
        transcendentals=0,
        remote_bytes_transferred=int(remote_bytes_transferred),
    )


def _fused_moe_dispatch_cost_metadata(
    cost_estimate: pl.CostEstimate, *, uses_sp_all_gather: bool
) -> dict[str, str]:
    """Describe either dispatch kernel without duplicating shared metadata."""
    return cost_estimate_metadata(
        flops_basis="integer routing metadata; floating-point FLOPs are zero",
        bytes_accessed_basis=(
            "modeled HBM reads and writes, including recursive forwarding and "
            "the final gathered-run reread"
            if uses_sp_all_gather
            else "modeled HBM input reads and output writes"
        ),
        remote_bytes_transferred=cost_estimate.remote_bytes_transferred,
        remote_bytes_transferred_basis=(
            "logical max of the communicated packed-run input and gathered output; "
            "not physical TPU-link traffic"
            if uses_sp_all_gather
            else "0; routing IDs are already replicated and no TPU collective runs"
        ),
    )


def should_use_fused_sp_moe_dispatch_metadata(
    local_assignments: int, *, num_experts: int, ep_size: int
) -> bool:
    """Return whether the fused collective supports and wins for this shape."""
    return (
        num_experts == 256
        and local_assignments >= 128
        and local_assignments & (local_assignments - 1) == 0
        and ep_size >= 1
        and ep_size & (ep_size - 1) == 0
        and local_assignments * ep_size >= _MIN_PROFITABLE_GLOBAL_ASSIGNMENTS
    )


def should_use_fused_replicated_moe_dispatch_metadata(
    assignments: int, *, num_experts: int, ep_size: int
) -> bool:
    """Return whether replicated SP-off routing fits the local VMEM sorter."""
    return (
        num_experts == 256
        and assignments >= 128
        and assignments & (assignments - 1) == 0
        and ep_size >= 1
        and ep_size & (ep_size - 1) == 0
    )


def _xor_partner(groups, stride):
    """Exchange XOR partners in ``[groups, tiles, 128]`` layout."""
    if stride < _LANES:
        lanes = jnp.arange(_LANES, dtype=jnp.int32)
        first = ((lanes & stride) == 0)[None, None, :]
        higher = pltpu.roll(groups, _LANES - stride, axis=2)
        lower = pltpu.roll(groups, stride, axis=2)
        return jnp.where(first, higher, lower)

    tile_stride = stride // _LANES
    blocks = groups.reshape(groups.shape[0], -1, tile_stride * 2, _LANES)
    return jnp.concatenate(
        (blocks[:, :, tile_stride:], blocks[:, :, :tile_stride]), axis=2
    ).reshape(groups.shape)


def _bitonic_sort_groups(groups):
    """Sort each power-of-two group while preserving 128-lane TPU layout."""
    size = groups.shape[1] * _LANES
    result = groups
    positions = (
        jnp.arange(groups.shape[1], dtype=jnp.int32)[None, :, None] * _LANES
        + jnp.arange(_LANES, dtype=jnp.int32)[None, None, :]
    )
    merge_size = 2
    while merge_size <= size:
        stride = merge_size // 2
        while stride:
            partner = _xor_partner(result, stride)
            lower = jnp.minimum(result, partner)
            upper = jnp.maximum(result, partner)
            ascending = (positions & merge_size) == 0
            first_in_pair = (positions & stride) == 0
            result = jnp.where(first_in_pair == ascending, lower, upper)
            stride //= 2
        merge_size *= 2
    return result


def _reverse_groups(groups):
    """Reverse each flattened group with tile-safe static permutations."""
    values = groups
    result = values
    block_size = 2
    while block_size <= values.shape[1]:
        half = block_size // 2
        blocks = result.reshape(result.shape[0], -1, block_size, _LANES)
        result = jnp.concatenate(
            (blocks[:, :, half:], blocks[:, :, :half]), axis=2
        ).reshape(values.shape)
        block_size *= 2
    for stride in (1, 2, 4, 8, 16, 32, 64):
        result = _xor_partner(result, stride)
    return result


def _merge_sorted_runs(sorted_runs):
    """Merge adjacent sorted power-of-two runs without indirect indexing."""
    num_runs, run_tiles, _ = sorted_runs.shape
    result = sorted_runs
    runs_per_group = 2
    while runs_per_group <= num_runs:
        group_tiles = runs_per_group * run_tiles
        groups = result.reshape(-1, group_tiles, _LANES)
        half_tiles = group_tiles // 2
        groups = jnp.concatenate(
            (groups[:, :half_tiles], _reverse_groups(groups[:, half_tiles:])), axis=1
        )
        stride = group_tiles * _LANES // 2
        while stride:
            partner = _xor_partner(groups, stride)
            lanes = jnp.arange(_LANES, dtype=jnp.int32)[None, None, :]
            positions = (
                jnp.arange(group_tiles, dtype=jnp.int32)[None, :, None] * _LANES + lanes
            )
            first = (positions & stride) == 0
            groups = jnp.where(
                first,
                jnp.minimum(groups, partner),
                jnp.maximum(groups, partner),
            )
            stride //= 2
        result = groups.reshape(num_runs, run_tiles, _LANES)
        runs_per_group *= 2
    return result


def _exclusive_prefix_256(group_sizes):
    """Exclusive prefix for the two TPU lane tiles holding 256 expert counts."""
    inclusive = group_sizes
    lanes = jnp.arange(_LANES, dtype=jnp.int32)[None, :]
    for stride in (1, 2, 4, 8, 16, 32, 64):
        shifted = pltpu.roll(inclusive, stride, axis=1)
        inclusive = inclusive + jnp.where(lanes >= stride, shifted, 0)
    first_tile_total = inclusive[0, -1]
    tile_offsets = jnp.arange(group_sizes.shape[0], dtype=jnp.int32)[:, None]
    inclusive = inclusive + jnp.where(tile_offsets == 0, 0, first_tile_total)
    return inclusive - group_sizes


def write_sp_moe_dispatch_metadata(
    local_ids,
    gathered_runs_ref,
    lhs_indices_ref,
    group_sizes_ref,
    output_indices_ref,
    topk_slot_indices_ref,
    group_start_ref,
    total_recv_count_ref,
    max_num_gm_ref,
    local_dma_sem,
    send_sems,
    recv_sems,
    packed_run_vmem_ref,
    gathered_runs_vmem_ref,
    lhs_indices_vmem_ref,
    group_sizes_vmem_ref,
    output_indices_vmem_ref,
    topk_slot_indices_vmem_ref,
    group_start_vmem_ref,
    total_recv_count_vmem_ref,
    max_num_gm_vmem_ref,
    *,
    my_rank,
    right_device,
    left_device,
    num_experts,
    topk,
    ep_size,
    ep_axis_name,
    tile_m,
    size_lhs_sublane,
):
    """Write global fused-RS metadata from one rank's tiled router IDs."""
    local_ids = local_ids.astype(jnp.int32)
    local_assignments = local_ids.size
    global_assignments = local_assignments * ep_size
    global_tiles = global_assignments // _LANES

    local_rows = (
        jnp.arange(local_ids.shape[0], dtype=jnp.int32)[:, None] * _LANES
        + jnp.arange(_LANES, dtype=jnp.int32)[None, :]
    )
    global_flat_rows = my_rank * local_assignments + local_rows
    local_sort_keys = local_ids * global_assignments + global_flat_rows
    sorted_keys = _bitonic_sort_groups(local_sort_keys[None, ...])[0]
    experts = jnp.arange(num_experts, dtype=jnp.int32)
    local_counts = jnp.sum(
        (local_ids[..., None] == experts[None, None, :]).astype(jnp.int32),
        axis=(0, 1),
    )
    packed_run = jnp.concatenate(
        (sorted_keys, local_counts.reshape(-1, _LANES)), axis=0
    )
    packed_run_vmem_ref[...] = packed_run
    local_store = pltpu.make_async_copy(
        src_ref=packed_run_vmem_ref,
        dst_ref=gathered_runs_ref.at[my_rank],
        sem=local_dma_sem,
    )
    local_store.start()
    local_store.wait()

    if ep_size > 1:
        # Recursive doubling gathers all rank slots in log2(ep_size) rounds.
        # After round ``step``, every rank owns the contiguous ``2**step``
        # block containing itself and sends that whole block to its XOR peer.
        # This preserves rank order in gathered_runs_ref while reducing TP16
        # synchronization rounds from the ring's 15 to 4.
        for step in range(ep_size.bit_length() - 1):
            span = 1 << step
            block_start = (my_rank // span) * span
            partner = my_rank ^ span
            rank_block = pl.ds(block_start, span)
            copy = pltpu.make_async_remote_copy(
                src_ref=gathered_runs_ref.at[rank_block],
                dst_ref=gathered_runs_ref.at[rank_block],
                send_sem=send_sems.at[step],
                recv_sem=recv_sems.at[step],
                device_id={ep_axis_name: partner},
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            copy.start()
            copy.wait()

    gathered_load = pltpu.make_async_copy(
        src_ref=gathered_runs_ref,
        dst_ref=gathered_runs_vmem_ref,
        sem=local_dma_sem,
    )
    gathered_load.start()
    gathered_load.wait()
    gathered = gathered_runs_vmem_ref[...]
    local_tiles = local_assignments // _LANES
    gathered_rows = gathered[:, :local_tiles, :]
    all_counts = gathered[:, local_tiles:, :]
    sorted_global_keys = _merge_sorted_runs(gathered_rows).reshape(-1, _LANES)
    global_counts = jnp.sum(all_counts, axis=0)
    global_flat_rows = sorted_global_keys % global_assignments
    lhs_indices = global_flat_rows // topk
    topk_slots = global_flat_rows % topk

    expert_ids = jnp.arange(num_experts, dtype=jnp.int32).reshape(-1, _LANES)
    experts_per_rank = num_experts // ep_size
    first_local_expert = my_rank * experts_per_rank
    after_local_expert = first_local_expert + experts_per_rank
    group_start = jnp.sum(jnp.where(expert_ids < first_local_expert, global_counts, 0))
    group_end = group_start + jnp.sum(
        jnp.where(
            (expert_ids >= first_local_expert) & (expert_ids < after_local_expert),
            global_counts,
            0,
        )
    )
    assignment_rows = (
        jnp.arange(global_tiles, dtype=jnp.int32)[:, None] * _LANES
        + jnp.arange(_LANES, dtype=jnp.int32)[None, :]
    )
    local_tokens = local_assignments // topk
    destination_ranks = lhs_indices // local_tokens
    row_is_local = (assignment_rows >= group_start) & (assignment_rows < group_end)
    total_recv_count = jnp.sum(
        ((destination_ranks == my_rank) & ~row_is_local).astype(jnp.int32)
    )

    if tile_m is None:
        max_num_gm = jnp.array(0, dtype=jnp.int32)
    else:
        # The fused GMM runs a common static loop bound on every EP rank. Derive
        # that bound while global_counts is already resident in VMEM, rather than
        # materializing group_sizes to HBM and rebuilding its prefix outside this
        # kernel. Experts are assigned as contiguous groups of experts_per_rank.
        group_starts = _exclusive_prefix_256(global_counts)
        local_offsets = group_starts % size_lhs_sublane
        expert_tiles = jnp.where(
            global_counts > 0,
            cdiv(global_counts + local_offsets, tile_m),
            0,
        )
        # Each EP rank owns 16 contiguous experts. Sum those groups in the native
        # [2, 128] lane layout: Mosaic cannot shape-cast this vector to [16, 16].
        rank_tiles = expert_tiles
        for stride in (1, 2, 4, 8):
            rank_tiles = rank_tiles + pltpu.roll(rank_tiles, stride, axis=1)
        rank_end = (jnp.arange(_LANES, dtype=jnp.int32)[None, :] + 1) % (
            num_experts // ep_size
        ) == 0
        max_num_gm = jnp.max(jnp.where(rank_end, rank_tiles, 0)).astype(jnp.int32)

    lhs_indices_vmem_ref[...] = lhs_indices
    output_indices_vmem_ref[...] = lhs_indices
    topk_slot_indices_vmem_ref[...] = topk_slots
    group_sizes_vmem_ref[...] = global_counts
    group_start_vmem_ref[...] = jnp.zeros((1, _LANES), jnp.int32) + group_start
    total_recv_count_vmem_ref[...] = (
        jnp.zeros((1, _LANES), jnp.int32) + total_recv_count
    )
    max_num_gm_vmem_ref[...] = jnp.zeros((1, _LANES), jnp.int32) + max_num_gm

    for source_ref, destination_ref in (
        (lhs_indices_vmem_ref, lhs_indices_ref),
        (group_sizes_vmem_ref, group_sizes_ref),
        (output_indices_vmem_ref, output_indices_ref),
        (topk_slot_indices_vmem_ref, topk_slot_indices_ref),
        (group_start_vmem_ref, group_start_ref),
        (total_recv_count_vmem_ref, total_recv_count_ref),
        (max_num_gm_vmem_ref, max_num_gm_ref),
    ):
        output_store = pltpu.make_async_copy(
            src_ref=source_ref,
            dst_ref=destination_ref,
            sem=local_dma_sem,
        )
        output_store.start()
        output_store.wait()

    if ep_size > 1:
        # Prevent a faster rank from reusing collective scratch while its
        # neighbor is still consuming the current invocation.
        local_barrier(left_device, right_device)


def _sp_dispatch_metadata_kernel(
    local_topk_ref,
    gathered_runs_ref,
    lhs_indices_ref,
    group_sizes_ref,
    output_indices_ref,
    topk_slot_indices_ref,
    group_start_ref,
    total_recv_count_ref,
    max_num_gm_ref,
    local_dma_sem,
    send_sems,
    recv_sems,
    local_topk_vmem_ref,
    packed_run_vmem_ref,
    gathered_runs_vmem_ref,
    lhs_indices_vmem_ref,
    group_sizes_vmem_ref,
    output_indices_vmem_ref,
    topk_slot_indices_vmem_ref,
    group_start_vmem_ref,
    total_recv_count_vmem_ref,
    max_num_gm_vmem_ref,
    *,
    num_experts,
    topk,
    ep_size,
    ep_axis_name,
    tile_m,
    size_lhs_sublane,
):
    local_load = pltpu.make_async_copy(
        src_ref=local_topk_ref,
        dst_ref=local_topk_vmem_ref,
        sem=local_dma_sem,
    )
    local_load.start()
    local_load.wait()
    my_rank = lax.axis_index(ep_axis_name)
    write_sp_moe_dispatch_metadata(
        local_topk_vmem_ref[...],
        gathered_runs_ref,
        lhs_indices_ref,
        group_sizes_ref,
        output_indices_ref,
        topk_slot_indices_ref,
        group_start_ref,
        total_recv_count_ref,
        max_num_gm_ref,
        local_dma_sem,
        send_sems,
        recv_sems,
        packed_run_vmem_ref,
        gathered_runs_vmem_ref,
        lhs_indices_vmem_ref,
        group_sizes_vmem_ref,
        output_indices_vmem_ref,
        topk_slot_indices_vmem_ref,
        group_start_vmem_ref,
        total_recv_count_vmem_ref,
        max_num_gm_vmem_ref,
        my_rank=my_rank,
        right_device={ep_axis_name: (my_rank + 1) % ep_size},
        left_device={ep_axis_name: (my_rank - 1) % ep_size},
        num_experts=num_experts,
        topk=topk,
        ep_size=ep_size,
        ep_axis_name=ep_axis_name,
        tile_m=tile_m,
        size_lhs_sublane=size_lhs_sublane,
    )


def _replicated_dispatch_metadata_kernel(
    topk_ref,
    lhs_indices_ref,
    group_sizes_ref,
    output_indices_ref,
    topk_slot_indices_ref,
    group_start_ref,
    total_recv_count_ref,
    max_num_gm_ref,
    dma_sem,
    topk_vmem_ref,
    lhs_indices_vmem_ref,
    group_sizes_vmem_ref,
    output_indices_vmem_ref,
    topk_slot_indices_vmem_ref,
    group_start_vmem_ref,
    total_recv_count_vmem_ref,
    max_num_gm_vmem_ref,
    *,
    num_experts,
    topk,
    ep_size,
    ep_axis_name,
    tile_m,
    size_lhs_sublane,
):
    """Build fused-RS metadata from IDs already replicated on every EP rank."""
    topk_load = pltpu.make_async_copy(
        src_ref=topk_ref,
        dst_ref=topk_vmem_ref,
        sem=dma_sem,
    )
    topk_load.start()
    topk_load.wait()

    ids = topk_vmem_ref[...].astype(jnp.int32)
    assignments = ids.size
    assignment_rows = (
        jnp.arange(ids.shape[0], dtype=jnp.int32)[:, None] * _LANES
        + jnp.arange(_LANES, dtype=jnp.int32)[None, :]
    )
    sort_keys = ids * assignments + assignment_rows
    sorted_keys = _bitonic_sort_groups(sort_keys[None, ...])[0]
    sorted_rows = sorted_keys % assignments
    lhs_indices = sorted_rows // topk
    topk_slots = sorted_rows % topk

    expert_ids = jnp.arange(num_experts, dtype=jnp.int32).reshape(-1, _LANES)
    group_sizes = jnp.sum(
        (ids[..., None] == expert_ids.reshape(-1)[None, None, :]).astype(jnp.int32),
        axis=(0, 1),
    ).reshape(-1, _LANES)
    my_rank = lax.axis_index(ep_axis_name)
    experts_per_rank = num_experts // ep_size
    first_local_expert = my_rank * experts_per_rank
    after_local_expert = first_local_expert + experts_per_rank
    group_start = jnp.sum(jnp.where(expert_ids < first_local_expert, group_sizes, 0))
    group_end = group_start + jnp.sum(
        jnp.where(
            (expert_ids >= first_local_expert) & (expert_ids < after_local_expert),
            group_sizes,
            0,
        )
    )

    tokens = assignments // topk
    tokens_per_rank = tokens // ep_size
    destination_ranks = lhs_indices // tokens_per_rank
    row_is_local = (assignment_rows >= group_start) & (assignment_rows < group_end)
    total_recv_count = jnp.sum(
        ((destination_ranks == my_rank) & ~row_is_local).astype(jnp.int32)
    )

    if tile_m is None:
        max_num_gm = jnp.array(0, dtype=jnp.int32)
    else:
        group_starts = _exclusive_prefix_256(group_sizes)
        local_offsets = group_starts % size_lhs_sublane
        expert_tiles = jnp.where(
            group_sizes > 0,
            cdiv(group_sizes + local_offsets, tile_m),
            0,
        )
        rank_tiles = expert_tiles
        for stride in (1, 2, 4, 8):
            rank_tiles = rank_tiles + pltpu.roll(rank_tiles, stride, axis=1)
        rank_end = (jnp.arange(_LANES, dtype=jnp.int32)[None, :] + 1) % (
            num_experts // ep_size
        ) == 0
        max_num_gm = jnp.max(jnp.where(rank_end, rank_tiles, 0)).astype(jnp.int32)

    lhs_indices_vmem_ref[...] = lhs_indices
    group_sizes_vmem_ref[...] = group_sizes
    output_indices_vmem_ref[...] = lhs_indices
    topk_slot_indices_vmem_ref[...] = topk_slots
    group_start_vmem_ref[...] = jnp.zeros((1, _LANES), jnp.int32) + group_start
    total_recv_count_vmem_ref[...] = (
        jnp.zeros((1, _LANES), jnp.int32) + total_recv_count
    )
    max_num_gm_vmem_ref[...] = jnp.zeros((1, _LANES), jnp.int32) + max_num_gm

    for source_ref, destination_ref in (
        (lhs_indices_vmem_ref, lhs_indices_ref),
        (group_sizes_vmem_ref, group_sizes_ref),
        (output_indices_vmem_ref, output_indices_ref),
        (topk_slot_indices_vmem_ref, topk_slot_indices_ref),
        (group_start_vmem_ref, group_start_ref),
        (total_recv_count_vmem_ref, total_recv_count_ref),
        (max_num_gm_vmem_ref, max_num_gm_ref),
    ):
        output_store = pltpu.make_async_copy(
            src_ref=source_ref,
            dst_ref=destination_ref,
            sem=dma_sem,
        )
        output_store.start()
        output_store.wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_experts",
        "topk",
        "ep_size",
        "ep_axis_name",
        "tile_m",
        "size_lhs_sublane",
    ),
)
def fused_replicated_moe_dispatch_metadata(
    topk_indices,
    *,
    num_experts,
    topk,
    ep_size,
    ep_axis_name,
    tile_m=None,
    size_lhs_sublane=1,
):
    """Return fused-RS metadata without gathering already-replicated IDs."""
    if topk_indices.ndim not in (1, 2):
        raise ValueError("topk_indices must be flat or rank-2")
    if topk_indices.ndim == 2 and topk_indices.shape[1] != topk:
        raise ValueError(f"topk={topk} does not match input shape {topk_indices.shape}")
    assignments = topk_indices.size
    if not should_use_fused_replicated_moe_dispatch_metadata(
        assignments,
        num_experts=num_experts,
        ep_size=ep_size,
    ):
        raise ValueError(
            "replicated fused routing requires power-of-two assignments, 256 experts, "
            f"and power-of-two EP; got assignments={assignments}, experts={num_experts}, "
            f"ep={ep_size}"
        )
    tokens = assignments // topk
    if assignments % topk or tokens % ep_size:
        raise ValueError(
            f"assignments={assignments} must divide topk={topk} and ep={ep_size}"
        )
    if tile_m is not None and tile_m <= 0:
        raise ValueError(f"tile_m must be positive when supplied, got {tile_m}")
    if size_lhs_sublane <= 0:
        raise ValueError(f"size_lhs_sublane must be positive, got {size_lhs_sublane}")

    assignment_tiles = assignments // _LANES
    expert_tiles = num_experts // _LANES
    topk_indices = topk_indices.reshape(assignment_tiles, _LANES)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    cost_estimate = get_fused_replicated_moe_dispatch_cost_estimate(
        assignments=assignments,
        num_experts=num_experts,
        indices_dtype=topk_indices.dtype,
    )
    outputs = pl.pallas_call(
        functools.partial(
            _replicated_dispatch_metadata_kernel,
            num_experts=num_experts,
            topk=topk,
            ep_size=ep_size,
            ep_axis_name=ep_axis_name,
            tile_m=tile_m,
            size_lhs_sublane=size_lhs_sublane,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((assignment_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((expert_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((assignment_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((assignment_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((1, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((1, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((1, _LANES), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm_spec,),
            out_specs=(hbm_spec,) * 7,
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((assignment_tiles, _LANES), topk_indices.dtype),
                pltpu.VMEM((assignment_tiles, _LANES), jnp.int32),
                pltpu.VMEM((expert_tiles, _LANES), jnp.int32),
                pltpu.VMEM((assignment_tiles, _LANES), jnp.int32),
                pltpu.VMEM((assignment_tiles, _LANES), jnp.int32),
                pltpu.VMEM((1, _LANES), jnp.int32),
                pltpu.VMEM((1, _LANES), jnp.int32),
                pltpu.VMEM((1, _LANES), jnp.int32),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        cost_estimate=cost_estimate,
        metadata=_fused_moe_dispatch_cost_metadata(
            cost_estimate, uses_sp_all_gather=False
        ),
        name="fused_replicated_moe_dispatch_metadata",
    )(topk_indices)
    return tuple(output.reshape(-1) for output in outputs)


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_experts",
        "topk",
        "ep_size",
        "ep_axis_name",
        "collective_id",
        "tile_m",
        "size_lhs_sublane",
    ),
)
def fused_sp_moe_dispatch_metadata(
    local_topk_indices,
    *,
    num_experts,
    topk,
    ep_size,
    ep_axis_name,
    collective_id=1,
    tile_m=None,
    size_lhs_sublane=1,
):
    """Return global fused-RS metadata from token-sharded router IDs.

    When ``tile_m`` is supplied, the final tuple element is the per-rank-broadcast
    ``max_num_gm`` required by the fused GMM. Computing it here keeps the expert
    counts in VMEM and removes the post-kernel prefix/gather/reduce sequence.
    """
    if local_topk_indices.ndim not in (1, 2):
        raise ValueError(
            "local_topk_indices must be flat or have shape [local_tokens, topk], got "
            f"{local_topk_indices.shape}"
        )
    if local_topk_indices.ndim == 2 and local_topk_indices.shape[1] != topk:
        raise ValueError(
            f"topk={topk} does not match input shape {local_topk_indices.shape}"
        )
    local_assignments = local_topk_indices.size
    if local_assignments % topk:
        raise ValueError(
            f"local assignment count {local_assignments} is not divisible by topk={topk}"
        )
    if local_assignments < 128 or local_assignments & (local_assignments - 1):
        raise ValueError(
            "local assignment count must be a power of two and at least 128, got "
            f"{local_assignments}"
        )
    if num_experts != 256:
        raise ValueError(f"SP fused routing requires 256 experts, got {num_experts}")
    if ep_size < 1 or ep_size & (ep_size - 1):
        raise ValueError(f"ep_size must be a positive power of two, got {ep_size}")
    if tile_m is not None and tile_m <= 0:
        raise ValueError(f"tile_m must be positive when supplied, got {tile_m}")
    if size_lhs_sublane <= 0:
        raise ValueError(f"size_lhs_sublane must be positive, got {size_lhs_sublane}")

    local_tiles = local_assignments // _LANES
    expert_tiles = num_experts // _LANES
    global_tiles = local_tiles * ep_size
    run_tiles = local_tiles + expert_tiles
    local_topk_indices = local_topk_indices.reshape(local_tiles, _LANES)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    cost_estimate = get_fused_sp_moe_dispatch_cost_estimate(
        local_assignments=local_assignments,
        num_experts=num_experts,
        ep_size=ep_size,
        indices_dtype=local_topk_indices.dtype,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=(hbm_spec,),
        out_specs=(
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
        ),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((max(ep_size.bit_length() - 1, 1),)),
            pltpu.SemaphoreType.DMA((max(ep_size.bit_length() - 1, 1),)),
            pltpu.VMEM((local_tiles, _LANES), local_topk_indices.dtype),
            pltpu.VMEM((run_tiles, _LANES), jnp.int32),
            pltpu.VMEM((ep_size, run_tiles, _LANES), jnp.int32),
            pltpu.VMEM((global_tiles, _LANES), jnp.int32),
            pltpu.VMEM((expert_tiles, _LANES), jnp.int32),
            pltpu.VMEM((global_tiles, _LANES), jnp.int32),
            pltpu.VMEM((global_tiles, _LANES), jnp.int32),
            pltpu.VMEM((1, _LANES), jnp.int32),
            pltpu.VMEM((1, _LANES), jnp.int32),
            pltpu.VMEM((1, _LANES), jnp.int32),
        ),
        grid=(),
    )
    outputs = pl.pallas_call(
        functools.partial(
            _sp_dispatch_metadata_kernel,
            num_experts=num_experts,
            topk=topk,
            ep_size=ep_size,
            ep_axis_name=ep_axis_name,
            tile_m=tile_m,
            size_lhs_sublane=size_lhs_sublane,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((ep_size, run_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((global_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((expert_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((global_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((global_tiles, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((1, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((1, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((1, _LANES), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        cost_estimate=cost_estimate,
        metadata=_fused_moe_dispatch_cost_metadata(
            cost_estimate, uses_sp_all_gather=True
        ),
        name="fused_sp_moe_dispatch_metadata",
    )(local_topk_indices)
    metadata = tuple(output.reshape(-1) for output in outputs[1:])
    return metadata if tile_m is not None else metadata[:-1]


__all__ = [
    "fused_replicated_moe_dispatch_metadata",
    "fused_sp_moe_dispatch_metadata",
    "get_fused_replicated_moe_dispatch_cost_estimate",
    "get_fused_sp_moe_dispatch_cost_estimate",
    "should_use_fused_replicated_moe_dispatch_metadata",
    "should_use_fused_sp_moe_dispatch_metadata",
    "write_sp_moe_dispatch_metadata",
]
