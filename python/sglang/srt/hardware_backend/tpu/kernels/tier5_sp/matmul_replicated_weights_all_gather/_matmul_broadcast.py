# SPDX-License-Identifier: Apache-2.0
"""The small-T fused broadcast kernel: ``all_gather(x) @ W`` for a REPLICATED weight.

``_matmul_all_gather_broadcast`` -- each core matmuls only its own ``[T/P, K]`` rows against the
replicated ``[K, N]`` (or ``[N, K]``) weight, streamed over N tiles, and broadcasts each output
tile to all peers directly from double-buffered VMEM so the output gather (ICI) overlaps the
next tile's W-read (HBM). The best small-T path (needs ``T/P >= 16`` and a small output);
dispatched by ``api.matmul_all_gather`` when each rank owns at least 16 rows.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import util

P = jax.sharding.PartitionSpec


def _butterfly_barrier(my_id, num_devices, peer_id_fn):
    """All-to-all barrier in log2(P) 1-hop rounds (vs a flat P-way signal storm).

    Round r pairs each core with its XOR-2^r neighbor (a single hypercube-bit flip
    -> one physical ICI hop on the 2x2x2x2 mesh); after ``log2(P)`` rounds every
    pair of cores is transitively synchronized. Each round uses its own fresh
    ``run_scoped`` REGULAR semaphore, so there is no cross-round or cross-call
    (collective_id reuse) carryover. NOTE: measured *slower* than the flat
    double-barrier here (its 4 sequential rounds each stall on inter-host drift);
    kept as an option. Requires ``num_devices`` to be a power of two.
    """
    num_rounds = num_devices.bit_length() - 1

    @functools.partial(pl.run_scoped, rsem=pltpu.SemaphoreType.REGULAR((num_rounds,)))
    def _(rsem):
        for r in range(num_rounds):
            partner = lax.bitwise_xor(my_id, jnp.int32(1 << r))
            pltpu.semaphore_signal(
                rsem.at[r],
                inc=1,
                device_id=peer_id_fn(partner),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            pltpu.semaphore_wait(rsem.at[r], 1)


# ---------------------------------------------------------------------------
# Small-T fused kernel: broadcast directly from DOUBLE-BUFFERED VMEM (no HBM
# source staging, no per-step local-copy wait). Tile i's matmul writes o_vmem[i%2];
# the self-copy + 15 peer broadcasts read straight from that VMEM buffer; tile
# i-2's copies are waited just before its buffer is reused. Reading the broadcast
# source from VMEM (not HBM) keeps it off the HBM bus the matmul's W-read needs.
# ---------------------------------------------------------------------------


def _mm_ag_fused_kernel(
    x_ref,  # VMEM [m_local, k]
    w_ref,  # VMEM [k, bn] ([K, N] layout), or [bn, k] if w_transposed ([N, K])
    x_scale_ref,  # VMEM [m_local, 1] per-token online-FP8 scale, or None (no scale)
    o_hbm_ref,  # HBM [m, n]
    o_vmem,  # VMEM [2, m_local, bn]  double-buffered matmul result
    self_sems,  # DMA [2]            self copy (VMEM -> own HBM)
    send_sems,  # DMA [2, num_peers] remote broadcasts
    recv_sem,  # DMA []              accumulates all incoming
    *,
    axis_names: tuple[str, ...],
    strides: tuple[int, ...],
    sizes: tuple[int, ...],
    mesh_axis_names: tuple[str, ...],
    num_devices: int,
    m_local: int,
    bn: int,
    n: int,
    grid_n: int,
    double_barrier: bool,
    do_gather: bool,
    barrier_kind: str,
    w_transposed: bool,
):
    i = pl.program_id(0)  # 0 .. grid_n+1
    slot = lax.rem(i, jnp.int32(2))
    num_peers = num_devices - 1
    my_id = util.combined_axis_index(axis_names, strides)
    my_row0 = my_id * m_local
    _peer_id = util.make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names)

    def _self_desc(sl, j):
        return pltpu.make_async_copy(
            src_ref=o_vmem.at[sl],
            dst_ref=o_hbm_ref.at[pl.ds(my_row0, m_local), pl.ds(j * bn, bn)],
            sem=self_sems.at[sl],
        )

    def _send_desc(sl, j, d):
        pidx = lax.rem(my_id + d, jnp.int32(num_devices))
        return pltpu.make_async_remote_copy(
            src_ref=o_vmem.at[sl],
            dst_ref=o_hbm_ref.at[pl.ds(my_row0, m_local), pl.ds(j * bn, bn)],
            send_sem=send_sems.at[sl, d - 1],
            recv_sem=recv_sem,
            device_id=_peer_id(pidx),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _recv_desc(d):
        src_p = lax.rem(my_id + num_devices - d, jnp.int32(num_devices))
        return pltpu.make_async_remote_copy(
            src_ref=o_hbm_ref.at[pl.ds(src_p * m_local, m_local), :],
            dst_ref=o_hbm_ref.at[pl.ds(src_p * m_local, m_local), :],
            send_sem=send_sems.at[0, 0],
            recv_sem=recv_sem,
            device_id=_peer_id(src_p),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    @pl.when(i == 0)
    def _setup():
        if barrier_kind == "butterfly":
            _butterfly_barrier(my_id, num_devices, _peer_id)
        else:  # "flat" all-to-all (optionally double) -- the best measured option
            peers = [
                _peer_id(lax.rem(my_id + d, jnp.int32(num_devices)))
                for d in range(1, num_devices)
            ]
            util.hypercube_local_barrier(peers, double_barrier=double_barrier)

    # Free o_vmem[slot] by waiting tile (i-2)'s copies before matmul i overwrites it
    # (and to confirm completion in the two epilogue steps).
    @pl.when(jnp.logical_and(i >= 2, i < grid_n + 2))
    def _wait_prev():
        j = i - 2  # tile whose buffer (slot = i%2) is about to be reused
        pltpu.make_async_copy(
            src_ref=o_vmem.at[slot],
            dst_ref=o_hbm_ref.at[pl.ds(my_row0, m_local), pl.ds(j * bn, bn)],
            sem=self_sems.at[slot],
        ).wait()
        if do_gather:
            for d in range(1, num_devices):
                pidx = lax.rem(my_id + d, jnp.int32(num_devices))
                pltpu.make_async_remote_copy(
                    src_ref=o_vmem.at[slot],
                    dst_ref=o_hbm_ref.at[pl.ds(my_row0, m_local), pl.ds(j * bn, bn)],
                    send_sem=send_sems.at[slot, d - 1],
                    recv_sem=recv_sem,
                    device_id=_peer_id(pidx),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).wait_send()

    @pl.when(i < grid_n)
    def _compute_and_broadcast():
        # In-kernel requant: an fp8 x_ref stays fp8 in VMEM; reorder_matmul dequants it by the
        # per-token x_scale (raw fp8 -> f32 * scale -> W dtype) at the matmul, else just widens it.
        o_vmem.at[slot][...] = util.reorder_matmul(
            x_ref[...],
            w_ref[...],
            w_transposed=w_transposed,
            x_scale=x_scale_ref[...] if x_scale_ref is not None else None,
            preferred_element_type=jnp.float32,
        ).astype(o_vmem.dtype)
        _self_desc(slot, i).start()
        if do_gather:
            for d in range(1, num_devices):
                _send_desc(slot, i, d).start()

    @pl.when(i == grid_n + 1)
    def _drain_recv():
        if do_gather:
            for d in range(1, num_devices):
                _recv_desc(d).wait_recv()


def _pick_bn(n: int, target_tiles: int = 8) -> int:
    """N-tile width giving ~``target_tiles`` tiles, each a multiple of 128 (MXU lane).

    More tiles = finer matmul/gather overlap + smaller exposed tail, but more DMAs;
    ~8 tiles was the measured sweet spot. Searches tile counts outward from
    ``target_tiles``.
    """
    order = sorted(
        range(1, 2 * target_tiles + 1), key=lambda g: (abs(g - target_tiles), g)
    )
    for g in order:
        if n % g == 0 and (n // g) % 128 == 0:
            return n // g
    return n


def _matmul_all_gather_broadcast(
    x: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str | tuple[str, ...],
    *,
    bn: int | None = None,
    collective_id: int = 0,
    double_barrier: bool = True,
    do_gather: bool = True,
    barrier_kind: str = "flat",
    w_transposed: bool = False,
    out_dtype: jax.typing.DTypeLike | None = None,
    x_scale: jax.Array | None = None,
) -> jax.Array:
    """Fused small-T matmul-then-all-gather: matmul own rows, broadcast from VMEM.

    Each core matmuls only its own ``[T/P, K]`` rows against the replicated ``[K, N]``
    ``w`` (read once, streamed over N tiles of width ``bn``) and broadcasts each
    ``[T/P, bn]`` output tile to all peers **directly from a double-buffered VMEM
    scratch** (no HBM source staging). Tile ``i``'s broadcast (ICI) overlaps tile
    ``i+1``'s W-read (HBM). Current Mosaic packed-bf16
    HBM DMA requires 16-row-aligned local shards. For large T the output
    gather is bandwidth-bound and the broadcast loses -- use the pure-XLA reorder
    (``matmul_all_gather`` picks automatically).

    Args:
      bn: N-tile width (default: auto ~8 tiles via ``_pick_bn`` -- the measured best).
      do_gather: if False, skip the broadcast (debug: isolate the in-kernel matmul).
      barrier_kind: ``"flat"`` (default) -- all-to-all setup barrier (reuse-safe with
        ``double_barrier=True``, the cheapest correct option here); ``"butterfly"`` --
        log2(P) 1-hop rounds (correct but slower under multi-host drift).
      w_transposed: if True, ``w`` is ``[N, K]`` (ColumnParallelLinear ``[out, in]``
        native layout); the N-tiles are sliced ``[bn, K]`` and the matmul contracts K
        in-kernel, so no ``[N, K] -> [K, N]`` transpose ``copy`` is materialized.
      x_scale: per-token online-FP8 row scale ``[T]`` for a raw fp8 ``x`` -- passed INTO the kernel
        and dequantized in VMEM at the matmul (``reorder_matmul``); ``x`` stays fp8 in VMEM.
    """
    axis_names = axis_name if isinstance(axis_name, (tuple, list)) else (axis_name,)
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    mesh_axis_names = tuple(mesh.axis_names)
    m, k = x.shape
    n = w.shape[0] if w_transposed else w.shape[1]
    m_local = m // tp_size
    if m_local < 16:
        raise ValueError(
            "broadcast matmul-all-gather requires at least 16 local token rows"
        )
    # x_scale (per-token online-FP8 row scale) is dequanted IN-KERNEL by reorder_matmul (a raw fp8
    # x stays fp8 in VMEM), not up front. When there is no scale the operand + its BlockSpec are
    # omitted and the kernel is given a None x_scale_ref (no dummy tensor). Output dtype defaults to
    # bf16 when a scale is present (fp8 x -> bf16 output), else x's dtype.
    has_x_scale = x_scale is not None
    out_dtype = (
        out_dtype
        if out_dtype is not None
        else (jnp.bfloat16 if has_x_scale else x.dtype)
    )
    if has_x_scale:
        # [m, 1] column -- the shape reorder_matmul + the Pallas VMEM ref expect.
        x_scale = x_scale.reshape(m, 1)
    if bn is None:
        bn = _pick_bn(n)
    if n % bn != 0:
        raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")
    grid_n = n // bn
    # W N-tile: [bn, K] on the N (row) dim when transposed ([N, K]); else [K, bn].
    w_spec = (
        pl.BlockSpec((bn, k), lambda i: (jnp.minimum(i, grid_n - 1), 0))
        if w_transposed
        else pl.BlockSpec((k, bn), lambda i: (0, jnp.minimum(i, grid_n - 1)))
    )

    # x_scale is an optional Pallas input: a None BlockSpec paired with a None operand makes the
    # kernel's x_scale_ref None (fixed kernel signature, no dummy tensor).
    x_scale_spec = (
        pl.BlockSpec((m_local, 1), lambda i: (0, 0)) if has_x_scale else None  # [m, 1]
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((m_local, k), lambda i: (0, 0)), w_spec, x_scale_spec],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=(
            pltpu.VMEM((2, m_local, bn), out_dtype),  # o_vmem (double-buffered)
            pltpu.SemaphoreType.DMA((2,)),  # self_sems
            pltpu.SemaphoreType.DMA((2, tp_size - 1)),  # send_sems
            pltpu.SemaphoreType.DMA,  # recv_sem
        ),
        grid=(grid_n + 2,),
    )

    def _call(*operands):
        # operands = (xx, ww[, ss]); ss (and its BlockSpec) is None when there is no scale.
        ss = operands[2] if has_x_scale else None
        return pl.pallas_call(
            functools.partial(
                _mm_ag_fused_kernel,
                axis_names=axis_names,
                strides=strides,
                sizes=sizes,
                mesh_axis_names=mesh_axis_names,
                num_devices=tp_size,
                m_local=m_local,
                bn=bn,
                n=n,
                grid_n=grid_n,
                double_barrier=double_barrier,
                do_gather=do_gather,
                barrier_kind=barrier_kind,
                w_transposed=w_transposed,
            ),
            out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                # collective_id pairs with get_barrier_semaphore (the "custom
                # barrier"); the butterfly path uses run_scoped sems instead, so it
                # must leave collective_id unset.
                collective_id=None if barrier_kind == "butterfly" else collective_id,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            name=f"mm_ag_fused_bn{bn}_g{int(do_gather)}_{barrier_kind[:3]}",
        )(operands[0], operands[1], ss)

    sm_in_specs = [P(axis_names, None), P(None, None)]
    operands = [x, w]
    if has_x_scale:
        sm_in_specs.append(P(axis_names, None))
        operands.append(x_scale)

    return jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=tuple(sm_in_specs),
            out_specs=P(None, None),
            check_vma=False,
        )
    )(*operands)
