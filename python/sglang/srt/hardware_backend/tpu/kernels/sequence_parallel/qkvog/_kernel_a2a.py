# SPDX-License-Identifier: Apache-2.0

"""Direct all-to-all schedule for the fused projection."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._epilogue import (
    HEAD_DIM,
    KEY_COLUMNS,
    OUTPUT_GATE_COLUMNS,
    QUERY_COLUMNS,
    VALUE_COLUMNS,
    gathered_interleaved_rope,
)
from ._schedule import (
    COLUMNS_PER_DESTINATION,
    HIDDEN_SIZE,
    TP_SIZE,
    AllToAllPlan,
)


def _signal_all_peers(axis_name: str) -> None:
    barrier = pltpu.get_barrier_semaphore()
    for peer in range(TP_SIZE):
        pltpu.semaphore_signal(
            barrier,
            device_id={axis_name: jnp.int32(peer)},
            device_id_type=pltpu.DeviceIdType.MESH,
        )


def _kernel_a2a(
    rank_metadata_ref,
    x_hbm_ref,
    weight_hbm_ref,
    gain_hbm_ref,
    cos_hbm_ref,
    sin_hbm_ref,
    q_hbm_ref,
    kv_hbm_ref,
    og_hbm_ref,
    *,
    axis_name: str,
    eps: float,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    plan: AllToAllPlan,
):
    """Project local rows and send final Q/K/V/OG tiles to each destination."""
    total_local_rows, hidden_size = x_hbm_ref.shape
    block_rows = plan.row_block_rows
    projection_widths = plan.projection_widths
    tiles_per_destination = len(projection_widths)
    max_projection_width = max(projection_widths)
    send_width = max(QUERY_COLUMNS, OUTPUT_GATE_COLUMNS)
    assert sum(projection_widths) == COLUMNS_PER_DESTINATION
    task_count = TP_SIZE * tiles_per_destination

    assert weight_hbm_ref.shape == (
        hidden_size,
        TP_SIZE * COLUMNS_PER_DESTINATION,
    )
    assert total_local_rows == plan.local_rows
    output_rows = TP_SIZE * total_local_rows
    assert cos_hbm_ref.shape == sin_hbm_ref.shape == (total_local_rows, HEAD_DIM)
    assert q_hbm_ref.shape == (output_rows, QUERY_COLUMNS)
    assert kv_hbm_ref.shape == (output_rows, KEY_COLUMNS + VALUE_COLUMNS)
    assert og_hbm_ref.shape == (output_rows, OUTPUT_GATE_COLUMNS)

    @pl.with_scoped(
        # The serving residual stream is FP32. Normalize from that source
        # precision, then store BF16 for the MXU just like the unfused path's
        # branch-dtype boundary.
        normalized=pltpu.VMEM((block_rows, hidden_size), jnp.bfloat16),
        x_tile=pltpu.VMEM((plan.norm_block_rows, hidden_size), x_hbm_ref.dtype),
        gain=pltpu.VMEM((hidden_size,), gain_hbm_ref.dtype),
        cos=pltpu.VMEM((block_rows, HEAD_DIM), cos_hbm_ref.dtype),
        sin=pltpu.VMEM((block_rows, HEAD_DIM), sin_hbm_ref.dtype),
        weight_tiles=pltpu.VMEM(
            (plan.weight_buffers, hidden_size, max_projection_width),
            weight_hbm_ref.dtype,
        ),
        projection_buffers=pltpu.VMEM(
            (plan.projection_buffers, block_rows, COLUMNS_PER_DESTINATION),
            jnp.bfloat16,
        ),
        send_buffers=pltpu.VMEM(
            (plan.send_slots, block_rows, send_width), q_hbm_ref.dtype
        ),
        kv_send_buffers=pltpu.VMEM(
            (TP_SIZE, block_rows, KEY_COLUMNS + VALUE_COLUMNS), kv_hbm_ref.dtype
        ),
        x_sem=pltpu.SemaphoreType.DMA,
        gain_sem=pltpu.SemaphoreType.DMA,
        rope_sems=pltpu.SemaphoreType.DMA((2,)),
        weight_sems=pltpu.SemaphoreType.DMA((plan.weight_buffers,)),
        send_sems=pltpu.SemaphoreType.DMA((plan.send_slots,)),
        kv_send_sems=pltpu.SemaphoreType.DMA((TP_SIZE,)),
        recv_sem=pltpu.SemaphoreType.DMA,
    )
    def _run(
        normalized,
        x_tile,
        gain,
        cos,
        sin,
        weight_tiles,
        projection_buffers,
        send_buffers,
        kv_send_buffers,
        x_sem,
        gain_sem,
        rope_sems,
        weight_sems,
        send_sems,
        kv_send_sems,
        recv_sem,
    ):
        # TorchTPU's logical TP order is not necessarily physical TPU order.
        # Scheduling and remote-copy destinations use physical IDs; output row
        # placement uses the source's logical rank so attention sees canonical
        # token order.
        my_id = rank_metadata_ref[0]
        logical_id = rank_metadata_ref[1]
        row_start = (
            pl.multiple_of(
                pl.program_id(0) * block_rows,
                plan.norm_block_rows,
            )
            if plan.local_rows > block_rows
            else 0
        )

        def task_coordinates(task):
            destination_round = task // tiles_per_destination
            column_index = task % tiles_per_destination
            destination_mask = lax.select(
                lax.rem(my_id, jnp.int32(2)) == 0,
                jnp.int32(plan.destination_orders[0][destination_round]),
                jnp.int32(plan.destination_orders[1][destination_round]),
            )
            destination = lax.bitwise_xor(my_id, destination_mask)
            column_start = sum(projection_widths[:column_index])
            column_width = projection_widths[column_index]
            return destination, column_start, column_width

        def weight_copy(task, slot):
            destination, column_start, column_width = task_coordinates(task)
            flat_column_start = pl.multiple_of(
                destination * COLUMNS_PER_DESTINATION + column_start,
                128,
            )
            return pltpu.make_async_copy(
                weight_hbm_ref.at[:, pl.ds(flat_column_start, column_width)],
                weight_tiles.at[slot, :, pl.ds(0, column_width)],
                weight_sems.at[slot],
            )

        def wait_send_slot(slot):
            # Reconstructing an async copy with the same refs and semaphore is
            # Pallas's API for waiting on a previously started transfer.
            payload = send_buffers.at[slot]
            pltpu.make_async_copy(
                payload,
                payload,
                send_sems.at[slot],
            ).wait()

        def wait_kv_send_slot(slot):
            pltpu.make_async_copy(
                kv_send_buffers.at[slot],
                kv_send_buffers.at[slot],
                kv_send_sems.at[slot],
            ).wait()

        def start_output_copy(source, target, send_sem, destination):
            @pl.when(destination != my_id)
            def _send_remote():
                pltpu.make_async_remote_copy(
                    src_ref=source,
                    dst_ref=target,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id={axis_name: destination},
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            @pl.when(destination == my_id)
            def _copy_local():
                pltpu.make_async_copy(source, target, send_sem).start()

        def qk_epilogue(value):
            value = value.astype(jnp.bfloat16).astype(jnp.float32)
            if use_qk_norm:
                inverse_rms = lax.rsqrt(
                    jnp.mean(value * value, axis=-1, keepdims=True) + eps_qk
                )
                head_scale = jnp.concatenate(
                    (
                        jnp.full(
                            (QUERY_COLUMNS // HEAD_DIM,),
                            scale_query_by,
                            dtype=jnp.float32,
                        ),
                        jnp.ones((1,), dtype=jnp.float32),
                    )
                )
                normalization_scale = inverse_rms * head_scale[None, :, None]
            if use_rope:
                value = gathered_interleaved_rope(
                    value,
                    cos[...][:, None, :],
                    sin[...][:, None, :],
                )
            if use_qk_norm:
                # A per-head scalar commutes with RoPE. Applying the combined
                # RMS/query scale once after rotation removes one full-width
                # FP32 multiply without changing the mathematical result.
                value *= normalization_scale
            return value.astype(jnp.bfloat16)

        # Protect remote output buffers, but overlap the barrier round trip with
        # local normalization, RoPE-table loads, and the first weight loads.
        _signal_all_peers(axis_name)
        gain_copy = pltpu.make_async_copy(gain_hbm_ref, gain, gain_sem)
        gain_copy.start()
        weight_copy(0, 0).start()
        weight_copy(1, 1).start()
        if use_rope:
            rope_rows = pl.ds(row_start, block_rows)
            cos_copy = pltpu.make_async_copy(
                cos_hbm_ref.at[rope_rows],
                cos,
                rope_sems.at[0],
            )
            sin_copy = pltpu.make_async_copy(
                sin_hbm_ref.at[rope_rows],
                sin,
                rope_sems.at[1],
            )
            cos_copy.start()
            sin_copy.start()
        gain_copy.wait()

        for row_block in range(block_rows // plan.norm_block_rows):
            local_rows = pl.ds(
                row_block * plan.norm_block_rows,
                plan.norm_block_rows,
            )
            input_rows = pl.ds(
                row_start + row_block * plan.norm_block_rows,
                plan.norm_block_rows,
            )
            x_copy = pltpu.make_async_copy(x_hbm_ref.at[input_rows], x_tile, x_sem)
            x_copy.start()
            x_copy.wait()
            x_f32 = x_tile[...].astype(jnp.float32)
            inverse_rms = lax.rsqrt(
                jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps
            )
            normalized.at[local_rows][...] = (
                x_f32 * inverse_rms * gain[...].astype(jnp.float32)[None, :]
            ).astype(normalized.dtype)

        if use_rope:
            cos_copy.wait()
            sin_copy.wait()

        # On every destination rank, rows are stored in canonical source-rank
        # order. This rank therefore always writes the same row interval; only
        # the remote destination changes as the pipeline advances.
        source_rows = pl.ds(
            logical_id * total_local_rows + row_start,
            block_rows,
        )

        def project_destination_into_buffer(destination_round, buffer_index):
            """Project one destination's QKV and OG columns into BF16 VMEM."""
            qkv_task = destination_round * tiles_per_destination
            og_task = qkv_task + 1
            _, _, qkv_width = task_coordinates(qkv_task)
            _, _, og_width = task_coordinates(og_task)

            # Refill each weight slot as soon as its current matmul has
            # consumed it, overlapping the following destination's HBM load
            # with independent MXU work.
            weight_copy(qkv_task, 0).wait()
            with jax.named_scope("qkv_matmul"):
                projection_buffers.at[buffer_index, :, pl.ds(0, qkv_width)][...] = (
                    lax.dot_general(
                        normalized[...],
                        weight_tiles.at[0, :, pl.ds(0, qkv_width)][...],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    ).astype(jnp.bfloat16)
                )
            if og_task + 1 < task_count:
                weight_copy(og_task + 1, 0).start()

            weight_copy(og_task, 1).wait()
            with jax.named_scope("og_matmul"):
                projection_buffers.at[
                    buffer_index,
                    :,
                    pl.ds(qkv_width, og_width),
                ][...] = lax.dot_general(
                    normalized[...],
                    weight_tiles.at[1, :, pl.ds(0, og_width)][...],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).astype(
                    jnp.bfloat16
                )
            if og_task + 2 < task_count:
                weight_copy(og_task + 2, 1).start()

        def finish_destination_from_buffer(destination_round, buffer_index):
            """Apply the attention epilogue and send one buffered destination."""
            qkv_task = destination_round * tiles_per_destination
            og_task = qkv_task + 1
            destination, _, qkv_width = task_coordinates(qkv_task)
            _, _, og_width = task_coordinates(og_task)
            q_send_slot = qkv_task % plan.send_slots
            og_send_slot = og_task % plan.send_slots

            # Keep completed output DMAs in flight until their payload slots
            # are needed again. The intervening destination work hides their
            # send-side latency.
            if qkv_task >= plan.send_slots:
                wait_send_slot(q_send_slot)
                wait_send_slot(og_send_slot)

            projection = projection_buffers.at[buffer_index]
            qkv_projected = projection.at[:, pl.ds(0, qkv_width)][...]
            og_projected = projection.at[:, pl.ds(qkv_width, og_width)][...]
            with jax.named_scope("qkv_epilogue"):
                qk = qk_epilogue(
                    qkv_projected[:, : QUERY_COLUMNS + KEY_COLUMNS].reshape(
                        block_rows,
                        (QUERY_COLUMNS + KEY_COLUMNS) // HEAD_DIM,
                        HEAD_DIM,
                    )
                )
                send_buffers.at[q_send_slot][...] = qk[
                    :, : QUERY_COLUMNS // HEAD_DIM, :
                ].reshape(block_rows, QUERY_COLUMNS)
                kv_send_buffers.at[destination_round, :, pl.ds(0, KEY_COLUMNS)][...] = (
                    qk[:, QUERY_COLUMNS // HEAD_DIM :, :]
                    .reshape(block_rows, KEY_COLUMNS)
                    .astype(kv_hbm_ref.dtype)
                )
                kv_send_buffers.at[
                    destination_round,
                    :,
                    pl.ds(KEY_COLUMNS, VALUE_COLUMNS),
                ][...] = qkv_projected[
                    :,
                    QUERY_COLUMNS
                    + KEY_COLUMNS : QUERY_COLUMNS
                    + KEY_COLUMNS
                    + VALUE_COLUMNS,
                ].astype(
                    kv_hbm_ref.dtype
                )
            send_buffers.at[og_send_slot][...] = og_projected.astype(jnp.bfloat16)

            # Projection is independent of remote output buffers, so wait for
            # the setup barrier only immediately before the first send.
            if destination_round == 0:
                pltpu.semaphore_wait(pltpu.get_barrier_semaphore(), TP_SIZE)

            start_output_copy(
                send_buffers.at[q_send_slot],
                q_hbm_ref.at[source_rows, :],
                send_sems.at[q_send_slot],
                destination,
            )
            start_output_copy(
                kv_send_buffers.at[destination_round],
                kv_hbm_ref.at[source_rows, :],
                kv_send_sems.at[destination_round],
                destination,
            )
            start_output_copy(
                send_buffers.at[og_send_slot],
                og_hbm_ref.at[source_rows, :],
                send_sems.at[og_send_slot],
                destination,
            )

        # Prime a two-stage software pipeline. Storing the raw BF16 result in
        # VMEM releases MXU accumulators before the epilogue. In steady state,
        # destination d projects into one buffer while destination d-1 applies
        # Q/K norm, RoPE, casts, and output copies from the other buffer.
        assert plan.projection_buffers == 2
        project_destination_into_buffer(0, 0)
        for destination_round in range(1, TP_SIZE):
            current_buffer = destination_round % plan.projection_buffers
            previous_buffer = (destination_round - 1) % plan.projection_buffers
            project_destination_into_buffer(destination_round, current_buffer)
            finish_destination_from_buffer(destination_round - 1, previous_buffer)
        finish_destination_from_buffer(
            TP_SIZE - 1,
            (TP_SIZE - 1) % plan.projection_buffers,
        )
        # Reset all send and aggregate receive semaphore counts before return.
        for slot in range(plan.send_slots):
            wait_send_slot(slot)
        for slot in range(TP_SIZE):
            wait_kv_send_slot(slot)

        # Each remote source contributes every output component. Reconstructing
        # three waits with the shared receive semaphore consumes their aggregate
        # byte counts while avoiding any additional data movement.
        received_rows = pl.ds(0, (TP_SIZE - 1) * block_rows)
        for output_ref in (q_hbm_ref, kv_hbm_ref, og_hbm_ref):
            payload = output_ref.at[received_rows, :]
            pltpu.make_async_copy(payload, payload, recv_sem).wait()

    _run()


def run_a2a(
    local_x: jax.Array,
    replicated_weight: jax.Array,
    replicated_gain: jax.Array,
    local_cos: jax.Array,
    local_sin: jax.Array,
    local_rank_metadata: jax.Array,
    *,
    axis_name: str,
    eps: float,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    kv_dtype: Any,
    plan: AllToAllPlan,
    collective_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Create the plan-specialized Pallas call for one rank."""
    if local_x.shape != (plan.local_rows, HIDDEN_SIZE):
        raise ValueError(
            f"plan expects x shape ({plan.local_rows}, {HIDDEN_SIZE}), "
            f"got {local_x.shape}"
        )
    if local_cos.shape != (plan.local_rows, HEAD_DIM) or local_sin.shape != (
        plan.local_rows,
        HEAD_DIM,
    ):
        raise ValueError("local cos/sin rows must match the plan's local token rows")
    if local_rank_metadata.shape != (TP_SIZE + 2,):
        raise ValueError(
            f"rank metadata must have local shape ({TP_SIZE + 2},), "
            f"got {local_rank_metadata.shape}"
        )

    grid = (
        (plan.local_rows // plan.row_block_rows,)
        if plan.local_rows > plan.row_block_rows
        else ()
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 3,
        grid=grid,
    )
    q, kv, og = pl.pallas_call(
        functools.partial(
            _kernel_a2a,
            axis_name=axis_name,
            eps=eps,
            eps_qk=eps_qk,
            scale_query_by=scale_query_by,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
            plan=plan,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (TP_SIZE * plan.local_rows, QUERY_COLUMNS),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(
                (TP_SIZE * plan.local_rows, KEY_COLUMNS + VALUE_COLUMNS),
                kv_dtype,
            ),
            jax.ShapeDtypeStruct(
                (TP_SIZE * plan.local_rows, OUTPUT_GATE_COLUMNS),
                jnp.bfloat16,
            ),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=plan.vmem_limit_bytes,
            collective_id=collective_id,
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        name="fused_rmsnorm_qkvog_mm_a2a_epilogue",
    )(
        local_rank_metadata,
        local_x,
        replicated_weight,
        replicated_gain,
        local_cos,
        local_sin,
    )
    return q, kv[:, :KEY_COLUMNS], kv[:, KEY_COLUMNS:], og
