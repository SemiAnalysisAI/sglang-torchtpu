# SPDX-License-Identifier: Apache-2.0
"""Bridge-overlapped half-weight matmul that preserves global token sharding.

Each core owns ``x[T/tp, K]`` and one output-feature half of ``w[N/2, K]``.
The kernel exchanges ``x`` only with the physical sibling core, computes the
local-token/local-weight result while that exchange is in flight, then computes
the sibling-token/local-weight result and writes it directly into the sibling's
output.  No ``[2*T/tp, ...]`` gathered activation or output is materialized.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import util
from ._routing import _physical_topo_maps

P = jax.sharding.PartitionSpec


def _mm_tp_intra_local_output_kernel(
    x_ref,  # VMEM [m_local, k]
    w_ref,  # VMEM [bn, k] ([N,K]) or [k, bn] ([K,N])
    o_hbm_ref,  # HBM [m_local, n]
    x_sib,  # VMEM [m_local, k]
    o_vmem,  # VMEM [2 slots, 2 destinations, m_local, bn]
    xchg_sem,  # DMA []
    self_sems,  # DMA [2]
    send_sems,  # DMA [2]
    recv_sems,  # DMA [2]
    *,
    axis_names: tuple[str, ...],
    strides: tuple[int, ...],
    sizes: tuple[int, ...],
    mesh_axis_names: tuple[str, ...],
    half_n: int,
    bn: int,
    grid_n: int,
    double_barrier: bool,
    w_transposed: bool,
):
    i = pl.program_id(0)
    slot = lax.rem(i, jnp.int32(2))
    my = util.combined_axis_index(axis_names, strides)
    core = lax.rem(my, jnp.int32(2))
    sib = lax.bitwise_xor(my, jnp.int32(1))
    peer = util.make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names)
    sib_col0 = lax.bitwise_xor(core, jnp.int32(1)) * jnp.int32(half_n)

    def _self_desc(sl, ti):
        return pltpu.make_async_copy(
            src_ref=o_vmem.at[sl, 0],
            dst_ref=o_hbm_ref.at[:, pl.ds(core * half_n + ti * bn, bn)],
            sem=self_sems.at[sl],
        )

    def _send_desc(sl, ti):
        return pltpu.make_async_remote_copy(
            src_ref=o_vmem.at[sl, 1],
            dst_ref=o_hbm_ref.at[:, pl.ds(core * half_n + ti * bn, bn)],
            send_sem=send_sems.at[sl],
            recv_sem=recv_sems.at[sl],
            device_id=peer(sib),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _recv_desc(sl, ti):
        block = o_hbm_ref.at[:, pl.ds(sib_col0 + ti * bn, bn)]
        return pltpu.make_async_remote_copy(
            src_ref=block,
            dst_ref=block,
            send_sem=send_sems.at[sl],
            recv_sem=recv_sems.at[sl],
            device_id=peer(sib),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _xchg_desc():
        return pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=x_sib,
            send_sem=xchg_sem,
            recv_sem=xchg_sem,
            device_id=peer(sib),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    @pl.when(i == 0)
    def _start_exchange():
        util.hypercube_local_barrier([peer(sib)], double_barrier=double_barrier)
        _xchg_desc().start()

    @pl.when(jnp.logical_and(i >= 2, i < grid_n + 2))
    def _wait_previous_tile():
        ti = i - 2
        sl = lax.rem(ti, jnp.int32(2))
        _self_desc(sl, ti).wait()
        _send_desc(sl, ti).wait_send()
        _recv_desc(sl, ti).wait_recv()

    @pl.when(i < grid_n)
    def _compute_tile():
        # This first matmul provides the useful work under the token exchange.
        o_vmem.at[slot, 0][...] = util.reorder_matmul(
            x_ref[...],
            w_ref[...],
            w_transposed=w_transposed,
            preferred_element_type=jnp.float32,
        ).astype(o_vmem.dtype)
        _self_desc(slot, i).start()

        # Only tile zero can race the exchange. Later tiles observe the completed
        # x_sib buffer through the semaphore dependency established here.
        @pl.when(i == 0)
        def _finish_exchange():
            op = _xchg_desc()
            op.wait_send()
            op.wait_recv()

        o_vmem.at[slot, 1][...] = util.reorder_matmul(
            x_sib[...],
            w_ref[...],
            w_transposed=w_transposed,
            preferred_element_type=jnp.float32,
        ).astype(o_vmem.dtype)
        _send_desc(slot, i).start()


def _pick_local_bn(half_n: int, m_local: int) -> int:
    # BN512 halves the tile-level DMA/semaphore overhead and wins for the small
    # TP16 shards (m_local=8..128). Keep BN256 from m_local=256 onward: wider
    # tiles increase VMEM/codegen pressure when composed into the full 62-layer
    # T=4096 backbone executable.
    candidates = (512, 256, 128) if m_local < 256 else (256, 128)
    for bn in candidates:
        if half_n % bn == 0:
            return bn
    return half_n


def _matmul_intra_tp_local_output(
    x: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    chip_axis: str,
    core_axis: str,
    *,
    bn: int | None = None,
    collective_id: int = 0,
    double_barrier: bool = True,
    disable_bounds_checks: bool = False,
    out_dtype: jax.typing.DTypeLike | None = None,
    w_transposed: bool = False,
) -> jax.Array:
    """Run the bridge-overlapped local-output pooler on a physical core pair."""
    axis_names = (chip_axis, core_axis)
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    mesh_axis_names = tuple(mesh.axis_names)
    if int(mesh.shape[core_axis]) != 2:
        raise ValueError("tp-intra local output requires a size-2 core axis")

    pos_to_phys, _ = _physical_topo_maps(mesh, axis_names)
    if any((pos_to_phys[p] & 1) != (p & 1) for p in range(tp_size)):
        raise ValueError(
            "tp-intra local output requires core_axis to be the physical on-chip pair"
        )

    m, k = x.shape
    n = w.shape[0] if w_transposed else w.shape[1]
    if m % tp_size != 0 or n % 2 != 0:
        raise ValueError(f"incompatible shapes x={x.shape}, w={w.shape}, tp={tp_size}")
    m_local = m // tp_size
    half_n = n // 2
    if bn is None:
        bn = _pick_local_bn(half_n, m_local)
    if bn <= 0:
        raise ValueError(f"bn must be positive, got {bn}")
    if half_n % bn != 0:
        raise ValueError(f"half_n ({half_n}) must be divisible by bn ({bn})")
    grid_n = half_n // bn
    od = out_dtype if out_dtype is not None else x.dtype

    w_spec = (
        pl.BlockSpec(
            (bn, k),
            lambda i: (jnp.minimum(i, grid_n - 1), 0),
        )
        if w_transposed
        else pl.BlockSpec(
            (k, bn),
            lambda i: (0, jnp.minimum(i, grid_n - 1)),
        )
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((m_local, k), lambda i: (0, 0)),
            w_spec,
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=(
            pltpu.VMEM((m_local, k), x.dtype),
            pltpu.VMEM((2, 2, m_local, bn), od),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        grid=(grid_n + 2,),
    )

    def _call(xx, ww):
        return pl.pallas_call(
            functools.partial(
                _mm_tp_intra_local_output_kernel,
                axis_names=axis_names,
                strides=strides,
                sizes=sizes,
                mesh_axis_names=mesh_axis_names,
                half_n=half_n,
                bn=bn,
                grid_n=grid_n,
                double_barrier=double_barrier,
                w_transposed=w_transposed,
            ),
            out_shape=jax.ShapeDtypeStruct((m_local, n), od),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=64 * 1024 * 1024,
                disable_bounds_checks=disable_bounds_checks,
            ),
            name=f"mm_tp_intra_local_output_bn{bn}",
        )(xx, ww)

    w_shard = P(core_axis, None) if w_transposed else P(None, core_axis)
    return jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(P(axis_names, None), w_shard),
            out_specs=P(axis_names, None),
            check_vma=False,
        )
    )(x, w)
