# SPDX-License-Identifier: Apache-2.0
"""Fused tp-intra-chip unpooler: half-W matmul with a pre-matmul core-bridge token exchange.

The unpooler is O[T,N] = X[T,K] @ W[N,K]^T with X EP-token-sharded [T/16,K] and, in tp-intra-chip
mode, W sharded on N 2-way over the 2 on-chip cores (core-bridge, physical bit 0). Compared to a
full-W broadcast (each core reads the FULL W [N,K] and matmuls its own [T/16,K] token slice), this
variant:

  1. PRE-MATMUL core-bridge exchange: each core swaps its [T/16,K] EP slice with its on-chip sibling
     so both hold the chip's full token block [T/8,K]. (Keeps FLOPs non-redundant: each chip's
     tokens are matmul'd once, not replicated 8x across chips.) v1 waits the exchange up front; it
     can later be overlapped (matmul your own rows while the sibling's arrive).
  2. HALF-W matmul: each core reads only its N-half [N/2,K] (HALF the W HBM read -- the win, since
     a full-W broadcast is W-read-bound at small T) and computes [T/8, N/2].
  3. POST-MATMUL broadcast: each core's [T/8, N/2] piece is 2D-placed at (its chip's T/8 token rows,
     its core's N/2 column-half) and broadcast to all peers -> [T,N] replicated. Same total gather
     traffic as a full-W broadcast (T*N), but sourced from a half-N matmul.

Mesh: a (chip, core) mesh (core = the size-2 on-chip pair). x is EP-sharded over BOTH axes
(P((chip,core), None)); W is sharded on N over ``core`` (P(core, None)) -> each core's operand is
[N/2,K], BlockSpec-tiled so Pallas reads only that half. Requires a hypercube-ordered mesh so the
raveled position is ``chip*2 + core`` and the sibling is ``pos ^ 1`` (the physical core-bridge).

v1 = correctness-first: full pre-exchange wait (no overlap) + flat 15-peer broadcast (no core-bridge
routing of the N-combine). Both are optimizations to layer on once correct.
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


def _mm_ag_tp_intra_kernel(
    x_ref,  # VMEM [m_local, k]         this core's EP token slice (T/16); raw fp8 if x_scale
    w_ref,  # VMEM [bn, k] ([N,K]) or [k, bn] ([K,N])  this core's N-half tile i (per w_transposed)
    x_scale_ref,  # VMEM [m_local, 1]   per-token online-FP8 scale, or None (no scale)
    o_hbm_ref,  # HBM  [T, n]           full replicated output
    x_full,  # VMEM [m_chip, k]         scratch: chip token block ([core0; core1], T/8)
    x_sib,  # VMEM [m_local, k]         scratch: sibling's EP slice (received whole)
    x_full_scale,  # VMEM [m_chip, 1]   scratch: chip token block per-token scales
    x_sib_scale,  # VMEM [m_local, 1]   scratch: sibling's per-token scales
    o_vmem,  # VMEM [2, m_chip, bn]     double-buffered matmul-result tile
    xchg_sem,  # DMA []                 core-bridge token exchange (send+recv)
    xchg_scale_sem,  # DMA []           core-bridge scale exchange (concurrent with token)
    self_sems,  # DMA [2]               self copy (VMEM -> own HBM block)
    send_sems,  # DMA [2, num_peers]    remote broadcasts
    recv_sem,  # DMA []                 accumulates all incoming
    *,
    axis_names: tuple[str, ...],
    strides: tuple[int, ...],
    sizes: tuple[int, ...],
    mesh_axis_names: tuple[str, ...],
    num_devices: int,
    m_local: int,
    bn: int,
    half_n: int,
    grid_n: int,
    double_barrier: bool,
    do_gather: bool,
    w_transposed: bool,
):
    i = pl.program_id(0)  # 0 .. grid_n + 1
    slot = lax.rem(i, jnp.int32(2))
    num_peers = num_devices - 1
    m_chip = 2 * m_local
    my = util.combined_axis_index(axis_names, strides)
    chip = my // jnp.int32(2)
    core = lax.rem(my, jnp.int32(2))
    sib = lax.bitwise_xor(
        my, jnp.int32(1)
    )  # on-chip sibling (core-bridge); hypercube order
    _peer = util.make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names)
    my_row0 = chip * jnp.int32(m_chip)  # this chip's token block start row
    my_col0 = core * jnp.int32(half_n)  # this core's N-half start column

    def _self_desc(sl, j):
        return pltpu.make_async_copy(
            src_ref=o_vmem.at[sl],
            dst_ref=o_hbm_ref.at[pl.ds(my_row0, m_chip), pl.ds(my_col0 + j * bn, bn)],
            sem=self_sems.at[sl],
        )

    def _send_desc(sl, j, d):
        pidx = lax.rem(my + d, jnp.int32(num_devices))
        return pltpu.make_async_remote_copy(
            src_ref=o_vmem.at[sl],
            dst_ref=o_hbm_ref.at[pl.ds(my_row0, m_chip), pl.ds(my_col0 + j * bn, bn)],
            send_sem=send_sems.at[sl, d - 1],
            recv_sem=recv_sem,
            device_id=_peer(pidx),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _recv_desc(d):
        # Drain: the full [m_chip, half_n] block from source src_p (its chip rows, its N-half cols).
        src_p = lax.rem(my + num_devices - d, jnp.int32(num_devices))
        s_chip = src_p // jnp.int32(2)
        s_core = lax.rem(src_p, jnp.int32(2))
        blk = o_hbm_ref.at[
            pl.ds(s_chip * m_chip, m_chip), pl.ds(s_core * half_n, half_n)
        ]
        return pltpu.make_async_remote_copy(
            src_ref=blk,
            dst_ref=blk,
            send_sem=send_sems.at[0, 0],
            recv_sem=recv_sem,
            device_id=_peer(src_p),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    @pl.when(i == 0)
    def _setup_and_exchange():
        peers = [
            _peer(lax.rem(my + d, jnp.int32(num_devices)))
            for d in range(1, num_devices)
        ]
        util.hypercube_local_barrier(peers, double_barrier=double_barrier)
        # Core-bridge exchange the token slice into a WHOLE [m_local, k] scratch (x_sib), then assemble
        # x_full = [chip core0 tokens; core1 tokens] via a STATIC concat. This avoids slicing a VMEM
        # buffer at the dynamic m_local row offset (core * m_local), which Mosaic rejects when m_local
        # < the sublane tile (T < 256): a dynamic sub-tile offset is not provably tile-aligned. The
        # concat is static, so it lowers for any m_local (incl. T=128). When there's an fp8 scale,
        # START its exchange (separate xchg_scale_sem) CONCURRENTLY with the token DMA -- the tiny
        # scale copy then overlaps the token copy instead of serializing after it.
        op = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=x_sib,
            send_sem=xchg_sem,
            recv_sem=xchg_sem,
            device_id=_peer(sib),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        op.start()
        if x_scale_ref is not None:
            ops = pltpu.make_async_remote_copy(
                src_ref=x_scale_ref,
                dst_ref=x_sib_scale,
                send_sem=xchg_scale_sem,
                recv_sem=xchg_scale_sem,
                device_id=_peer(sib),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            ops.start()
        op.wait_send()
        op.wait_recv()  # x_sib now holds the sibling's [m_local, k] EP token slice
        own = x_ref[...]
        sib_rows = x_sib[...]
        # core 0's tokens occupy the chip's low rows, core 1's the high rows -> pick order per core.
        lo = jnp.where(core == jnp.int32(0), own, sib_rows)
        hi = jnp.where(core == jnp.int32(0), sib_rows, own)
        x_full[...] = jnp.concatenate([lo, hi], axis=0)
        if x_scale_ref is not None:
            # Drain the (already in-flight) scale exchange and assemble x_full_scale with the SAME
            # static concat, so reorder_matmul dequants the raw-fp8 x_full by its per-token scale.
            ops.wait_send()
            ops.wait_recv()
            own_s = x_scale_ref[...]
            sib_s = x_sib_scale[...]
            lo_s = jnp.where(core == jnp.int32(0), own_s, sib_s)
            hi_s = jnp.where(core == jnp.int32(0), sib_s, own_s)
            x_full_scale[...] = jnp.concatenate([lo_s, hi_s], axis=0)

    @pl.when(jnp.logical_and(i >= 2, i < grid_n + 2))
    def _wait_prev():
        j = i - 2
        _self_desc(slot, j).wait()
        if do_gather:
            for d in range(1, num_devices):
                _send_desc(slot, j, d).wait_send()

    @pl.when(i < grid_n)
    def _compute_and_broadcast():
        o_vmem.at[slot][...] = util.reorder_matmul(
            x_full[...],
            w_ref[...],
            w_transposed=w_transposed,  # [bn,k] ([N,K]) or [k,bn] ([K,N])
            # raw-fp8 x_full dequanted in-VMEM by its per-token scale at the matmul (None -> bf16 x).
            x_scale=x_full_scale[...] if x_scale_ref is not None else None,
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


def _matmul_all_gather_tp_intra_bcast(
    x: jax.Array,  # [T, K] EP-sharded P((chip, core), None) -> local [T/16, K]
    w: jax.Array,  # [N, K] sharded on N over `core` P(core, None) -> local [N/2, K]
    mesh: jax.sharding.AbstractMesh,
    chip_axis: str,
    core_axis: str,
    *,
    bn: int | None = None,
    collective_id: int = 0,
    double_barrier: bool = True,
    do_gather: bool = True,
    disable_bounds_checks: bool = False,
    out_dtype: jax.typing.DTypeLike | None = None,
    w_transposed: bool = False,
    x_scale: jax.Array | None = None,
) -> jax.Array:
    """Fused tp-intra-chip unpooler: pre-matmul core-bridge token exchange + half-W matmul +
    2D-placed broadcast -> [T, N] replicated. ``core_axis`` must be the size-2 on-chip-core axis;
    the mesh must be hypercube-ordered (sibling = raveled pos ^ 1).

    ``w_transposed=True`` -> ``w`` is ``[N, K]`` (sharded on N = dim 0 over ``core_axis``); ``False``
    -> ``[K, N]`` (dim 1). ``x_scale`` (per-token row scale for a RAW fp8 ``x``) is passed INTO the
    kernel (+ exchanged over the core-bridge with x) and dequanted in-VMEM at the matmul; ``x`` stays
    fp8 on the wire (mirrors matmul_replicated_weights_all_gather._matmul_topo)."""
    from ._routing import _pick_bn

    axis_names = (chip_axis, core_axis)
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    mesh_axis_names = tuple(mesh.axis_names)
    core_size = mesh.shape[core_axis]
    if core_size != 2:
        raise ValueError(
            f"tp-intra needs a size-2 core axis, got {core_axis}={core_size}"
        )
    m, k = x.shape
    # w sharded on N over core_axis: N = dim 0 for [N, K] (w_transposed), dim 1 for [K, N].
    n = w.shape[0] if w_transposed else w.shape[1]
    half_n = n // core_size  # local N-half
    m_local = m // tp_size
    # x_scale (per-token online-FP8 row scale for a raw fp8 x) is passed INTO the kernel and dequanted
    # IN-KERNEL at the matmul (x stays fp8 through the core-bridge exchange). od -> bf16 for an fp8 x,
    # else x's dtype. When there is no scale the operand + BlockSpec are omitted (kernel ref = None).
    has_x_scale = x_scale is not None
    od = (
        out_dtype
        if out_dtype is not None
        else (jnp.bfloat16 if has_x_scale else x.dtype)
    )
    sdt = (
        x_scale.dtype if has_x_scale else jnp.float32
    )  # per-token scale dtype (VMEM + DMA)
    if has_x_scale:
        x_scale = x_scale.reshape(
            m, 1
        )  # [m, 1] -- the shape reorder_matmul + the VMEM ref expect
    if bn is None:
        bn = _pick_bn(half_n)
    if half_n % bn != 0:
        raise ValueError(f"half_n ({half_n}) must be divisible by bn ({bn}).")
    grid_n = half_n // bn
    # W N-tile: [bn, K] on the N (row) dim when transposed ([N, K]); else [K, bn] on dim 1 ([K, N]).
    w_spec = (
        pl.BlockSpec((bn, k), lambda i: (jnp.minimum(i, grid_n - 1), 0))
        if w_transposed
        else pl.BlockSpec((k, bn), lambda i: (0, jnp.minimum(i, grid_n - 1)))
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((m_local, k), lambda i: (0, 0)),  # x (resident)
            w_spec,  # w N-half tile ([bn,k] or [k,bn] per w_transposed)
            (
                pl.BlockSpec((m_local, 1), lambda i: (0, 0)) if has_x_scale else None
            ),  # per-token scale [m_local, 1], or None (no scale)
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=(
            pltpu.VMEM(
                (2 * m_local, k), x.dtype
            ),  # x_full (chip token block; raw fp8 if scaled)
            pltpu.VMEM(
                (m_local, k), x.dtype
            ),  # x_sib (sibling's EP slice, received whole)
            pltpu.VMEM((2 * m_local, 1), sdt),  # x_full_scale (chip per-token scales)
            pltpu.VMEM((m_local, 1), sdt),  # x_sib_scale (sibling per-token scales)
            pltpu.VMEM((2, 2 * m_local, bn), od),  # o_vmem (double-buffered)
            pltpu.SemaphoreType.DMA,  # xchg_sem
            pltpu.SemaphoreType.DMA,  # xchg_scale_sem
            pltpu.SemaphoreType.DMA((2,)),  # self_sems
            pltpu.SemaphoreType.DMA((2, tp_size - 1)),  # send_sems
            pltpu.SemaphoreType.DMA,  # recv_sem
        ),
        grid=(grid_n + 2,),
    )

    def _call(*operands):
        # operands = (xx, ww[, ss]); ss (+ its BlockSpec) is None when there is no scale -> the
        # kernel's x_scale_ref is None. x stays RAW fp8 into the kernel; the per-token scale is
        # exchanged over the core-bridge with x and dequant is in-VMEM at the matmul.
        ss = operands[2] if has_x_scale else None
        return pl.pallas_call(
            functools.partial(
                _mm_ag_tp_intra_kernel,
                axis_names=axis_names,
                strides=strides,
                sizes=sizes,
                mesh_axis_names=mesh_axis_names,
                num_devices=tp_size,
                m_local=m_local,
                bn=bn,
                half_n=half_n,
                grid_n=grid_n,
                double_barrier=double_barrier,
                do_gather=do_gather,
                w_transposed=w_transposed,
            ),
            out_shape=jax.ShapeDtypeStruct((m, n), od),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=64 * 1024 * 1024,
                disable_bounds_checks=disable_bounds_checks,
            ),
            name=f"mm_ag_tp_intra_bn{bn}_g{int(do_gather)}",
        )(operands[0], operands[1], ss)

    # w sharded on N over core_axis: dim 0 for [N, K] (w_transposed), dim 1 for [K, N].
    w_shard = P(core_axis, None) if w_transposed else P(None, core_axis)
    # x_scale ([m, 1] per-token row scale for a raw fp8 x) is a shard_map input, sharded on tokens;
    # omitted (with its BlockSpec) when there is no scale so the kernel's x_scale_ref is None.
    sm_in_specs = [P(axis_names, None), w_shard]
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
