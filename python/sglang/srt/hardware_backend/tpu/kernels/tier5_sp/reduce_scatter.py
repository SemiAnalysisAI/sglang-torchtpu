# SPDX-License-Identifier: Apache-2.0
"""TP16 reverse-cube reduce-scatter for a 2x2x2 TPU topology."""

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    get_vmem_size_bytes,
    hypercube_device_order,
    is_2x2x2_topology,
)

_TP_SIZE = 16

_HIDDEN_BLOCK_ALIGNMENT = 512
_VMEM_RESERVE_BYTES = 4 * 1024**2

# The reverse-cube algorithm partitions hidden over three physical ICI dimensions.
_CUBE_MAX_BLOCK_TOKENS = 256
_CUBE_OWNER_SLOT = 0
_CUBE_ANTIPODE_SLOT = 7
_CUBE_ICI_DIMENSION_MASKS = (2, 4, 8)
_CUBE_NUM_LANES = 2 * len(_CUBE_ICI_DIMENSION_MASKS)
# Give the first-started, usually widest stream more time to complete.
_CUBE_STREAM_ORDER = (1, 2, 0)
_CUBE_PHASE1_DESTINATION_SLOTS = tuple(
    _CUBE_ANTIPODE_SLOT ^ (physical_mask >> 1)
    for physical_mask in _CUBE_ICI_DIMENSION_MASKS
)
# phase-1 destination slot -> (phase-2 dimension, destination slot)
_CUBE_PHASE2_STEPS = {
    3: (0, 2),
    6: (1, 4),
    5: (2, 1),
}


def _get_usable_vmem_bytes() -> int:
    return get_vmem_size_bytes() - _VMEM_RESERVE_BYTES


def _get_cube_stream_shape(hidden: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split hidden into three nearly equal 128-column ICI streams."""
    hidden_tiles = hidden // 128
    base_tiles, extra_tiles = divmod(hidden_tiles, len(_CUBE_ICI_DIMENSION_MASKS))
    widths = tuple(
        (base_tiles + (stream < extra_tiles)) * 128
        for stream in range(len(_CUBE_ICI_DIMENSION_MASKS))
    )
    offsets = (0, widths[0], widths[0] + widths[1])
    return offsets, widths


def _get_cube_block_tokens(
    output_tokens: int, hidden: int, dtype_itemsize: int
) -> int | None:
    """Choose a power-of-two output tile supported by BF16 DMA layouts."""
    if output_tokens < 2:
        return None
    available_scratch_bytes = _get_usable_vmem_bytes()
    # Seven local slots, six receive lanes, and one output tile are reused.
    scratch_buffers = 7 + _CUBE_NUM_LANES + 1
    max_vmem_tokens = available_scratch_bytes // (
        scratch_buffers * hidden * dtype_itemsize
    )
    max_block_tokens = min(output_tokens, _CUBE_MAX_BLOCK_TOKENS, max_vmem_tokens)
    if max_block_tokens < 2:
        return None
    block_tokens = 1 << (max_block_tokens.bit_length() - 1)
    while block_tokens >= 2:
        if output_tokens % block_tokens == 0:
            return block_tokens
        block_tokens //= 2
    return None


def _physical_rank_maps(mesh: Mesh):
    devices = list(mesh.devices.flat)
    physical_devices = hypercube_device_order(devices)
    logical_to_physical = tuple(physical_devices.index(device) for device in devices)
    physical_to_logical = tuple(devices.index(device) for device in physical_devices)
    return devices, logical_to_physical, physical_to_logical


def _cube_rank_maps(mesh: Mesh) -> tuple[tuple[int, ...], ...]:
    """Map physical cube relationships to logical mesh ranks."""
    devices, logical_to_physical, physical_to_logical = _physical_rank_maps(mesh)

    def rank_map(physical_mask):
        return tuple(
            physical_to_logical[logical_to_physical[rank] ^ physical_mask]
            for rank in range(len(devices))
        )

    sibling_ranks = rank_map(1)
    neighbor_ranks = tuple(
        rank_map(physical_mask) for physical_mask in _CUBE_ICI_DIMENSION_MASKS
    )
    output_rank_ids = []
    for physical in logical_to_physical:
        core = physical & 1
        chip = physical >> 1
        output_rank_ids.append(
            tuple(
                physical_to_logical[((chip ^ offset) << 1) | core]
                for offset in range(8)
            )
        )
    return sibling_ranks, neighbor_ranks, tuple(output_rank_ids)


def _device_id(rank, *, tp_axis: int, mesh_ndim: int):
    return tuple(rank if axis == tp_axis else 0 for axis in range(mesh_ndim))


def _neighbor_barrier(neighbors, *, tp_axis: int, mesh_ndim: int) -> None:
    """Synchronize with the sibling core and three ICI neighbors."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in neighbors:
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=_device_id(neighbor, tp_axis=tp_axis, mesh_ndim=mesh_ndim),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, len(neighbors))


def _cube_local_slot(local_buffers, slot):
    if slot == _CUBE_OWNER_SLOT:
        raise ValueError("The output-owner slot lives in o_ref")
    return local_buffers.at[slot - 1]


def _start_core_copy(
    output_rank_ids_ref,
    sibling_ref,
    x_ref,
    o_ref,
    core_recv_sems,
    core_send_sems,
    core_local_sems,
    local_buffers,
    recv_buffers,
    lane: int,
    output_slot: int,
    hidden_start: int,
    stream_hidden: int,
    *,
    my_id,
    sibling,
    token_start,
    block_tokens: int,
    tp_axis: int,
    mesh_ndim: int,
):
    """Start local and sibling copies that form one chip-level partial."""
    output_rank = output_rank_ids_ref[my_id, output_slot]
    sibling_output_rank = sibling_ref[output_rank]
    destination = (
        o_ref.at[
            pl.ds(0, block_tokens),
            pl.ds(hidden_start, stream_hidden),
        ]
        if output_slot == _CUBE_OWNER_SLOT
        else _cube_local_slot(local_buffers, output_slot).at[
            pl.ds(0, block_tokens),
            pl.ds(hidden_start, stream_hidden),
        ]
    )
    local_copy = pltpu.make_async_copy(
        src_ref=x_ref.at[
            output_rank,
            pl.ds(token_start, block_tokens),
            pl.ds(hidden_start, stream_hidden),
        ],
        dst_ref=destination,
        sem=core_local_sems.at[lane],
    )
    sibling_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[
            sibling_output_rank,
            pl.ds(token_start, block_tokens),
            pl.ds(hidden_start, stream_hidden),
        ],
        dst_ref=recv_buffers.at[
            lane,
            pl.ds(0, block_tokens),
            pl.ds(hidden_start, stream_hidden),
        ],
        send_sem=core_send_sems.at[lane],
        recv_sem=core_recv_sems.at[lane],
        device_id=_device_id(sibling, tp_axis=tp_axis, mesh_ndim=mesh_ndim),
        device_id_type=pl.DeviceIdType.MESH,
    )
    local_copy.start()
    sibling_copy.start()
    return lane, hidden_start, stream_hidden, destination, local_copy, sibling_copy


def _wait_core_copy(core_copy, recv_buffers, *, block_tokens: int) -> None:
    """Wait for both core copies and add the sibling contribution."""
    lane, hidden_start, stream_hidden, destination, local_copy, sibling_copy = core_copy
    local_copy.wait()
    sibling_copy.wait()
    destination[...] += recv_buffers[
        lane,
        pl.ds(0, block_tokens),
        pl.ds(hidden_start, stream_hidden),
    ]


def _reduce_core_partials(core_specs, *, start_core_copy, wait_core_copy) -> None:
    """Reduce core-pair partials on lanes not occupied by ICI transfers."""
    lanes = (1, 3, 5)
    for batch_start in range(0, len(core_specs), len(lanes)):
        batch_specs = core_specs[batch_start : batch_start + len(lanes)]
        copies = [
            start_core_copy(
                lane=lanes[index],
                output_slot=spec[0],
                hidden_start=spec[1],
                stream_hidden=spec[2],
            )
            for index, spec in enumerate(batch_specs)
        ]
        for core_copy in copies:
            wait_core_copy(core_copy)


def _reduce_scatter_cube_block(
    sibling_ref,
    neighbor_ranks_ref,
    output_rank_ids_ref,
    x_ref,
    o_ref,
    core_recv_sems,
    core_send_sems,
    core_local_sems,
    ici_recv_sems,
    ici_send_sems,
    ici_capacity_sems,
    local_buffers,
    recv_buffers,
    *,
    axis_name: str,
    block_tokens: int,
    hidden: int,
    token_start,
    stream_offsets: tuple[int, ...],
    stream_widths: tuple[int, ...],
    tp_axis: int,
    mesh_ndim: int,
):
    """Reduce core pairs, then reverse three cube levels over ICI.

    Each core first combines the two partials owned by its chip. The resulting
    eight chip partials traverse antipode, distance-2, and distance-1 cube
    edges. Three streams partition hidden over the X, Y, and Z dimensions, while
    remaining core/HBM staging overlaps the antipode transfers.

    For output owner 0, chips are layered by cube distance. The antipode takes
    a different first edge in each stream, then all three reductions converge::

      stream   phase 1   phase 2                 phase 3
      0        7 -> 6    6 -> 4, 5 -> 1, 3 -> 2  4 -> 0, 1 -> 0, 2 -> 0
      1        7 -> 5    6 -> 4, 5 -> 1, 3 -> 2  4 -> 0, 1 -> 0, 2 -> 0
      2        7 -> 3    6 -> 4, 5 -> 1, 3 -> 2  4 -> 0, 1 -> 0, 2 -> 0

    Only phase 1 differs by stream. Phases 2 and 3 transfer full-hidden
    buffers, carrying all three streams over the same edges.

    The final groups are {0} + {4,6} + {1,5} + {2,3}, with contribution 7
    attached to the first, second, or third remote group for streams 0, 1, and
    2 respectively.
    """
    my_id = lax.axis_index(axis_name)
    sibling = sibling_ref[my_id]
    neighbors = tuple(
        neighbor_ranks_ref[cube_dimension, my_id]
        for cube_dimension in range(len(_CUBE_ICI_DIMENSION_MASKS))
    )
    _neighbor_barrier(
        (sibling, *neighbors),
        tp_axis=tp_axis,
        mesh_ndim=mesh_ndim,
    )

    start_core_copy = functools.partial(
        _start_core_copy,
        output_rank_ids_ref,
        sibling_ref,
        x_ref,
        o_ref,
        core_recv_sems,
        core_send_sems,
        core_local_sems,
        local_buffers,
        recv_buffers,
        my_id=my_id,
        sibling=sibling,
        token_start=token_start,
        block_tokens=block_tokens,
        tp_axis=tp_axis,
        mesh_ndim=mesh_ndim,
    )
    wait_core_copy = functools.partial(
        _wait_core_copy,
        recv_buffers=recv_buffers,
        block_tokens=block_tokens,
    )

    # Stage slot 7 and its phase-1 destination for each hidden stream.
    phase1_core_copies = []
    for stream in range(len(_CUBE_ICI_DIMENSION_MASKS)):
        hidden_start = stream_offsets[stream]
        stream_hidden = stream_widths[stream]
        source_lane = 2 * stream
        phase1_core_copies.append(
            (
                start_core_copy(
                    source_lane,
                    _CUBE_ANTIPODE_SLOT,
                    hidden_start,
                    stream_hidden,
                ),
                start_core_copy(
                    source_lane + 1,
                    _CUBE_PHASE1_DESTINATION_SLOTS[stream],
                    hidden_start,
                    stream_hidden,
                ),
            )
        )

    phase1_copies = [None] * len(_CUBE_ICI_DIMENSION_MASKS)
    for stream in _CUBE_STREAM_ORDER:
        source_copy, _ = phase1_core_copies[stream]
        wait_core_copy(source_copy)
        hidden_start = stream_offsets[stream]
        stream_hidden = stream_widths[stream]
        phase1_destination_slot = _CUBE_PHASE1_DESTINATION_SLOTS[stream]
        lane = 2 * stream
        phase1_copy = pltpu.make_async_remote_copy(
            src_ref=_cube_local_slot(local_buffers, _CUBE_ANTIPODE_SLOT).at[
                pl.ds(0, block_tokens),
                pl.ds(hidden_start, stream_hidden),
            ],
            dst_ref=recv_buffers.at[
                lane,
                pl.ds(0, block_tokens),
                pl.ds(hidden_start, stream_hidden),
            ],
            send_sem=ici_send_sems.at[0, stream],
            recv_sem=ici_recv_sems.at[0, stream],
            device_id=_device_id(
                neighbors[stream],
                tp_axis=tp_axis,
                mesh_ndim=mesh_ndim,
            ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        phase1_copy.start()
        phase1_copies[stream] = (
            lane,
            phase1_destination_slot,
            hidden_start,
            stream_hidden,
            phase1_copy,
        )

    # Launch all phase-1 streams before waiting for destination staging.
    for stream in _CUBE_STREAM_ORDER:
        _, destination_copy = phase1_core_copies[stream]
        wait_core_copy(destination_copy)

    # Stage the remaining stream data from HBM into slots 3, 5, and 6, then
    # reduce it with the sibling core's values to prepare phase 2 inputs.
    _reduce_core_partials(
        (
            (5, stream_offsets[0], stream_widths[0]),
            (5, stream_offsets[2], stream_widths[2]),
            (3, stream_offsets[0], stream_widths[0]),
            (3, stream_offsets[1], stream_widths[1]),
            (6, stream_offsets[1], stream_widths[1]),
            (6, stream_offsets[2], stream_widths[2]),
        ),
        start_core_copy=start_core_copy,
        wait_core_copy=wait_core_copy,
    )

    # Finish phase 1 and launch the three full-hidden phase-2 transfers.
    phase2_copies = []
    for stream in _CUBE_STREAM_ORDER:
        (
            lane,
            phase1_destination_slot,
            hidden_start,
            stream_hidden,
            phase1_copy,
        ) = phase1_copies[stream]
        phase1_copy.wait()
        _cube_local_slot(local_buffers, phase1_destination_slot)[
            pl.ds(0, block_tokens),
            pl.ds(hidden_start, stream_hidden),
        ] += recv_buffers[
            lane,
            pl.ds(0, block_tokens),
            pl.ds(hidden_start, stream_hidden),
        ]
        phase2_cube_dimension, phase2_destination_slot = _CUBE_PHASE2_STEPS[
            phase1_destination_slot
        ]
        phase2_source_slot = phase1_destination_slot
        phase2_recv_buffer = recv_buffers.at[lane]
        phase2_copy = pltpu.make_async_remote_copy(
            src_ref=_cube_local_slot(local_buffers, phase2_source_slot),
            dst_ref=phase2_recv_buffer,
            send_sem=ici_send_sems.at[1, phase2_cube_dimension],
            recv_sem=ici_recv_sems.at[1, phase2_cube_dimension],
            device_id=_device_id(
                neighbors[phase2_cube_dimension],
                tp_axis=tp_axis,
                mesh_ndim=mesh_ndim,
            ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        phase2_copy.start()
        phase2_copies.append(
            (
                phase2_cube_dimension,
                phase2_source_slot,
                phase2_destination_slot,
                phase2_recv_buffer,
                phase2_copy,
            )
        )

    # Stage the owner and phase-2 destination slots during phase 2.
    _reduce_core_partials(
        (
            (_CUBE_OWNER_SLOT, 0, hidden),
            (1, 0, hidden),
            (2, 0, hidden),
            (4, 0, hidden),
        ),
        start_core_copy=start_core_copy,
        wait_core_copy=wait_core_copy,
    )

    # Finish phase 2 and let neighbors overwrite our released source slots.
    phase3_specs = []
    for (
        phase2_cube_dimension,
        phase2_source_slot,
        phase2_destination_slot,
        phase2_recv_buffer,
        phase2_copy,
    ) in phase2_copies:
        phase2_copy.wait()
        _cube_local_slot(local_buffers, phase2_destination_slot)[
            ...
        ] += phase2_recv_buffer[...]
        phase3_cube_dimension = (phase2_cube_dimension + 1) % len(
            _CUBE_ICI_DIMENSION_MASKS
        )
        pl.semaphore_signal(
            ici_capacity_sems.at[phase2_cube_dimension],
            inc=1,
            device_id=_device_id(
                neighbors[phase3_cube_dimension],
                tp_axis=tp_axis,
                mesh_ndim=mesh_ndim,
            ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        phase3_specs.append(
            (
                phase2_cube_dimension,
                phase3_cube_dimension,
                phase2_destination_slot,
                phase2_source_slot,
            )
        )

    # Launch phase 3 after each neighbor releases the remote destination slot.
    phase3_copies = []
    for (
        phase2_cube_dimension,
        phase3_cube_dimension,
        phase3_source_slot,
        phase3_destination_slot,
    ) in phase3_specs:
        pl.semaphore_wait(ici_capacity_sems.at[phase2_cube_dimension], 1)
        phase3_copy = pltpu.make_async_remote_copy(
            src_ref=_cube_local_slot(local_buffers, phase3_source_slot),
            dst_ref=_cube_local_slot(local_buffers, phase3_destination_slot),
            send_sem=ici_send_sems.at[2, phase2_cube_dimension],
            recv_sem=ici_recv_sems.at[2, phase2_cube_dimension],
            device_id=_device_id(
                neighbors[phase3_cube_dimension],
                tp_axis=tp_axis,
                mesh_ndim=mesh_ndim,
            ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        phase3_copy.start()
        phase3_copies.append(phase3_copy)

    for phase3_copy in phase3_copies:
        phase3_copy.wait()

    # All neighbors must finish their final writes before these slots are consumed.
    _neighbor_barrier(
        (sibling, *neighbors),
        tp_axis=tp_axis,
        mesh_ndim=mesh_ndim,
    )

    # Combine the owner partial with the three completed remote reductions.
    o_ref[...] += (
        _cube_local_slot(local_buffers, 3)[...]
        + _cube_local_slot(local_buffers, 6)[...]
        + _cube_local_slot(local_buffers, 5)[...]
    )


def _reduce_scatter_cube_2x2x2_kernel(
    sibling_ref,
    neighbor_ranks_ref,
    output_rank_ids_ref,
    x_ref,
    o_ref,
    core_recv_sems,
    core_send_sems,
    core_local_sems,
    ici_recv_sems,
    ici_send_sems,
    ici_capacity_sems,
    local_buffers,
    recv_buffers,
    output_copy_sem,
    output_buffer,
    *,
    axis_name: str,
    block_tokens: int,
    hidden: int,
    stream_offsets: tuple[int, ...],
    stream_widths: tuple[int, ...],
    tp_axis: int,
    mesh_ndim: int,
):
    """Reduce one grid token block across the cube."""
    token_block = pl.program_id(0)
    token_start = token_block * block_tokens
    _reduce_scatter_cube_block(
        sibling_ref,
        neighbor_ranks_ref,
        output_rank_ids_ref,
        x_ref,
        output_buffer,
        core_recv_sems,
        core_send_sems,
        core_local_sems,
        ici_recv_sems,
        ici_send_sems,
        ici_capacity_sems,
        local_buffers,
        recv_buffers,
        axis_name=axis_name,
        block_tokens=block_tokens,
        hidden=hidden,
        token_start=token_start,
        stream_offsets=stream_offsets,
        stream_widths=stream_widths,
        tp_axis=tp_axis,
        mesh_ndim=mesh_ndim,
    )
    output_copy = pltpu.make_async_copy(
        src_ref=output_buffer,
        dst_ref=o_ref.at[
            pl.ds(token_start, block_tokens),
            pl.ds(0, hidden),
        ],
        sem=output_copy_sem,
    )
    output_copy.start()
    output_copy.wait()


def _reduce_scatter_cube_2x2x2(
    input_arr: jax.Array, mesh: Mesh, *, collective_id: int
) -> jax.Array:
    """Reduce-scatter a BF16 matrix over the reverse cube."""
    if not is_2x2x2_topology(mesh):
        raise ValueError("reduce_scatter_cube_2x2x2 requires a 2x2x2 TPU topology")
    tokens, hidden = input_arr.shape
    if input_arr.dtype != jnp.bfloat16 or hidden % _HIDDEN_BLOCK_ALIGNMENT:
        raise ValueError(
            "Cube reduce-scatter requires BF16 input with hidden divisible by 512; "
            f"got {input_arr.dtype} {input_arr.shape}"
        )
    if tokens < _TP_SIZE or tokens % _TP_SIZE:
        raise ValueError(
            f"Cube token count must be at least {_TP_SIZE} and divisible "
            f"by {_TP_SIZE}; got {tokens}"
        )
    output_tokens = tokens // _TP_SIZE
    padded_output_tokens = output_tokens + output_tokens % 2
    block_tokens = _get_cube_block_tokens(
        padded_output_tokens, hidden, input_arr.dtype.itemsize
    )
    if block_tokens is None:
        raise ValueError(
            "Cube block and scratch exceed VMEM for output shape "
            f"({output_tokens}, {hidden})"
        )

    tp_axis = next(axis for axis, size in enumerate(mesh.devices.shape) if size > 1)
    axis_name = mesh.axis_names[tp_axis]
    rank_maps = tuple(
        jnp.asarray(mapping, dtype=jnp.int32) for mapping in _cube_rank_maps(mesh)
    )
    stream_offsets, stream_widths = _get_cube_stream_shape(hidden)
    reshaped = input_arr.reshape(_TP_SIZE, output_tokens, hidden)
    if padded_output_tokens != output_tokens:
        # BF16 HBM DMA requires token-row pairs; the padded row remains zero.
        reshaped = jnp.pad(reshaped, ((0, 0), (0, 1), (0, 0)))
    num_token_blocks = padded_output_tokens // block_tokens
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        grid=(num_token_blocks,),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA((_CUBE_NUM_LANES,)),  # core_recv_sems
            pltpu.SemaphoreType.DMA((_CUBE_NUM_LANES,)),  # core_send_sems
            pltpu.SemaphoreType.DMA((_CUBE_NUM_LANES,)),  # core_local_sems
            pltpu.SemaphoreType.DMA(
                (3, len(_CUBE_ICI_DIMENSION_MASKS))
            ),  # ici_recv_sems[phase, cube_dimension]
            pltpu.SemaphoreType.DMA(
                (3, len(_CUBE_ICI_DIMENSION_MASKS))
            ),  # ici_send_sems[phase, cube_dimension]
            pltpu.SemaphoreType.REGULAR((3,)),  # ici_capacity_sems
            pltpu.VMEM((7, block_tokens, hidden), input_arr.dtype),  # local_buffers
            pltpu.VMEM(
                (_CUBE_NUM_LANES, block_tokens, hidden), input_arr.dtype
            ),  # recv_buffers
            pltpu.SemaphoreType.DMA,  # output_copy_sem
            pltpu.VMEM((block_tokens, hidden), input_arr.dtype),  # output_buffer
        ),
    )
    kernel_fn = pl.pallas_call(
        functools.partial(
            _reduce_scatter_cube_2x2x2_kernel,
            axis_name=axis_name,
            block_tokens=block_tokens,
            hidden=hidden,
            stream_offsets=stream_offsets,
            stream_widths=stream_widths,
            tp_axis=tp_axis,
            mesh_ndim=mesh.devices.ndim,
        ),
        out_shape=jax.ShapeDtypeStruct((padded_output_tokens, hidden), input_arr.dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            disable_bounds_checks=True,
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_get_usable_vmem_bytes(),
        ),
        name=f"reduce_scatter_cube_2x2x2_bt{block_tokens}",
    )
    return kernel_fn(*rank_maps, reshaped)[:output_tokens]


def reduce_scatter(
    input_arr: jax.Array, mesh: Mesh, *, collective_id: int
) -> jax.Array:
    """Reduce-scatter BF16 partials with the TP16 reverse-cube kernel."""
    if input_arr.ndim != 2:
        raise ValueError(
            "reduce_scatter expects a rank-local [tokens, hidden] array; "
            f"got shape {input_arr.shape}"
        )
    tokens = input_arr.shape[0]
    if tokens % mesh.size:
        raise ValueError(
            f"Token count ({tokens}) must be divisible by mesh size ({mesh.size})"
        )

    non_singleton_axes = sum(size > 1 for size in mesh.devices.shape)
    if non_singleton_axes != 1:
        raise ValueError("reduce_scatter requires one non-singleton mesh axis")

    if mesh.size != _TP_SIZE:
        raise ValueError(
            f"reduce_scatter is not supported with mesh size of {mesh.size}"
        )
    return _reduce_scatter_cube_2x2x2(input_arr, mesh, collective_id=collective_id)
