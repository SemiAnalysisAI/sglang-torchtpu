# SPDX-License-Identifier: Apache-2.0
"""COMBINE: tp-intra half-W matmul + topo bridge-last routed gather (fastest unpooler variant).

It combines the two levers that each help a different half of the unpooler:
  * the HALF-W matmul + pre-matmul core-bridge token exchange of ``_matmul_tp_intra_bcast`` (each
    core reads only [N/2,K]), and
  * a topology-aware hypercube store-and-forward gather (schedule in ``_routing.py``): 3 ICI steps
    gather the same-parity slots, then a core-bridge swap LAST -> ~2x less ICI than a flat broadcast.

Build the chip's [T/8,K] token block (whole-buffer exchange + static concat -- the tile-safe form that
compiles for any T, incl. T<256), do the HALF-W matmul -> [T/8, N/2] pieces, then route those pieces
through that schedule. The 16 pieces are (chip, N-half) blocks indexed by hypercube coord L (bit0 =
N-half/core, bits1-3 = chip torus); the ICI phase gathers the 8 same-parity (same-N-half) blocks over
the torus -> [T, N/2], and the final bridge swap combines the two N-halves -> [T, N]. That is precisely
the ICI-gather-narrow-then-bridge-combine routing.

Measured (scan single-dispatch us/iter, T128/256/512): the FASTEST variant -- it beats a FULL-W routed
gather 1.22x/1.11x/1.02x. The half-W read is only PARTIALLY hidden under the routed gather (saves
~9us@T128, ~5us@T256, ~1us@T512; more hidden as the gather grows), so it STACKS on the routed gather
(which itself beats the flat broadcast for T>=256) -- the combine of both levers wins.

FLEXIBLE MESH: routes by PHYSICAL hypercube coord via ``_routing._physical_topo_maps``
(``pos2phys``/``phys2pos`` scalar-prefetch), so ANY 16-core 2x2x2 order works (hypercube, the serve's
create_device_mesh ring/snake, ...) -- the piece IDENTITY + 2D placement stay in raveled space (the
token/output layout), only the DMA neighbour is physical. The one tp-intra-specific requirement:
``core_axis`` must be the PHYSICAL on-chip pair (bit 0), so the weight N-half split lines up with the
core-bridge -- ``_physical_topo_maps`` is checked to preserve bit 0 and the builder raises otherwise.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_attention_moe_prelude import (
    _prelude_values,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_router import (
    route_hidden_block,
)

from . import util
from ._routing import (
    _flatten_ici_schedule,
    _ici_offsets,
    _physical_topo_maps,
    _topo_bn,
    antipode_core_split,
    BRIDGE_MASK,
    ICI_MASKS,
    NUM_DEVICES,
    _TOPO_RING,
)

P = jax.sharding.PartitionSpec


def _mm_ag_tp_intra_topo_body(
    pos2phys_ref,  # SMEM [16] i32     raveled position -> physical hypercube coord
    phys2pos_ref,  # SMEM [16] i32     physical hypercube coord -> raveled position
    x_ref,  # VMEM [m_local, k]        this core's EP token slice (T/16); raw fp8 if x_scale
    w_ref,  # VMEM [bn, k]             this core's N-half, tile i ([N/2,K] operand)
    x_scale_ref,  # VMEM [m_local, 1]  per-token online-FP8 scale, or None (no scale)
    o_hbm_ref,  # HBM  [T, n]          full replicated output
    x_full,  # VMEM [m_chip, k]        scratch: chip token block ([core0; core1], T/8)
    x_sib,  # VMEM [m_local, k]        scratch: sibling's EP slice (received whole)
    x_full_scale,  # VMEM [m_chip, 1]  scratch: chip token block per-token scales
    x_sib_scale,  # VMEM [m_local, 1]  scratch: sibling's per-token scales
    o_vmem,  # VMEM [2, m_chip, bn]    double-buffered matmul-result tile ([T/8, bn] piece)
    xchg_sem,  # DMA []                core-bridge token exchange
    xchg_scale_sem,  # DMA []          core-bridge scale exchange (concurrent with token)
    seed_sems,  # DMA [ring]           seed self-copy (own piece -> own HBM 2D block)
    ici_send_sems,  # DMA [ring, n_ici]
    ici_recv_sems,  # DMA [ring, n_ici]
    br_send_sems,  # DMA [ring, n_bridge]
    br_recv_sems,  # DMA [ring, n_bridge]
    *,
    steps,
    all_masks,
    ici_offsets,
    antipode_split,
    axis_names: tuple[str, ...],
    strides: tuple[int, ...],
    sizes: tuple[int, ...],
    mesh_axis_names: tuple[str, ...],
    m_local: int,
    m_chip: int,
    half_n: int,
    bn: int,
    grid_n: int,
    ring: int,
    double_barrier: bool,
    w_transposed: bool,
    xchg_overlap_fn=None,
):
    # Software-pipelined store-and-forward (topo's 5-stage schedule) on HALF-W matmul pieces.
    # Stage `a` operates on tile `i-a`; the token exchange happens once at i==0 before a0's matmul.
    # Routing is by PHYSICAL coord (pos2phys/phys2pos) so any 2x2x2 mesh order works; piece identity
    # + 2D placement stay in RAVELED space (the token/output layout). core_axis == physical bit 0
    # (verified in the builder), so the weight N-half == my_pos%2 lines up with the core-bridge.
    i = pl.program_id(0)  # 0 .. grid_n + 3
    my_pos = util.combined_axis_index(
        axis_names, strides
    )  # raveled position (token/output layout)
    my_phys = pos2phys_ref[my_pos]  # physical hypercube coord (for routing)
    core = lax.rem(
        my_pos, jnp.int32(2)
    )  # raveled bit 0 == N-half id (== physical bit 0)
    _peer = util.make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names)
    last_step = len(steps) - 1

    def _rav(x):  # raveled position of the physical neighbour reached by XOR mask `x`
        return phys2pos_ref[lax.bitwise_xor(my_phys, x)]

    def _tile(g, ti):
        # HBM 2D block of the piece owned by RAVELED coord g: its chip's T/8 rows, its N-half's tile
        # ti. Raveled -> matches the EP token layout (chip g//2's tokens are contiguous [g//2*m_chip]).
        cc = g // jnp.int32(2)  # chip of piece g
        kk = lax.rem(g, jnp.int32(2))  # N-half of piece g
        return o_hbm_ref.at[
            pl.ds(cc * m_chip, m_chip), pl.ds(kk * half_n + ti * bn, bn)
        ]

    def _nb(mask):  # peer id of the physical neighbour reached by XOR `mask`
        return _peer(_rav(mask))

    def _ici(ti, buf, rs, si, entry):
        link_mask, slot_xor, gidx = entry
        if si == last_step:
            # antipode step: each on-chip core pulls the farthest piece over a DIFFERENT ICI axis.
            la, xa, lb, xb = antipode_split
            link_mask = jnp.where(core == jnp.int32(0), la, lb)
            slot_xor = jnp.where(core == jnp.int32(0), xa, xb)
        sl = _rav(slot_xor)  # raveled position of the physical slot being forwarded
        src = o_vmem.at[buf] if si == 0 else _tile(sl, ti)
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=_tile(sl, ti),
            send_sem=ici_send_sems.at[rs, gidx],
            recv_sem=ici_recv_sems.at[rs, gidx],
            device_id=_nb(link_mask),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _bridge(ti, rs, bi, off):
        # Swap one same-parity (same-N-half) slot with the on-chip sibling -> gains the other N-half.
        sl = _rav(off)
        return pltpu.make_async_remote_copy(
            src_ref=_tile(sl, ti),
            dst_ref=_tile(sl, ti),
            send_sem=br_send_sems.at[rs, bi],
            recv_sem=br_recv_sems.at[rs, bi],
            device_id=_nb(jnp.int32(BRIDGE_MASK)),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _seed_copy(ti, buf, rs):
        return pltpu.make_async_copy(
            src_ref=o_vmem.at[buf], dst_ref=_tile(my_pos, ti), sem=seed_sems.at[rs]
        )

    @pl.when(i == 0)
    def _prologue():
        # Barrier with the 4 physical neighbours we DMA to (3 ICI axes + the core-bridge).
        util.hypercube_local_barrier(
            [_nb(jnp.int32(m)) for m in all_masks], double_barrier=double_barrier
        )
        # Core-bridge exchange the token slice into a WHOLE [m_local,k] scratch, then assemble x_full
        # via a STATIC concat (tile-safe for any m_local; see _matmul_tp_intra_bcast). When there's an
        # fp8 scale, START its exchange (separate xchg_scale_sem) CONCURRENTLY with the token DMA so
        # the tiny scale copy overlaps it instead of serializing after.
        op = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=x_sib,
            send_sem=xchg_sem,
            recv_sem=xchg_sem,
            device_id=_nb(jnp.int32(BRIDGE_MASK)),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        op.start()
        if x_scale_ref is not None:
            ops = pltpu.make_async_remote_copy(
                src_ref=x_scale_ref,
                dst_ref=x_sib_scale,
                send_sem=xchg_scale_sem,
                recv_sem=xchg_scale_sem,
                device_id=_nb(jnp.int32(BRIDGE_MASK)),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            ops.start()
        if xchg_overlap_fn is not None:
            xchg_overlap_fn()
        op.wait_send()
        op.wait_recv()
        own = x_ref[...]
        sib_rows = x_sib[...]
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

    # ---- a0: tile i -- HALF-W matmul on the chip block, seed, ICI step 1 ----
    @pl.when(i < grid_n)
    def _a0():
        ti = i
        buf = lax.rem(ti, jnp.int32(2))
        rs = lax.rem(ti, jnp.int32(ring))
        o_vmem.at[buf][...] = util.reorder_matmul(
            x_full[...],
            w_ref[...],
            w_transposed=w_transposed,  # [bn,k] ([N,K]) or [k,bn] ([K,N])
            # raw-fp8 x_full dequanted in-VMEM by its per-token scale at the matmul (None -> bf16 x).
            x_scale=x_full_scale[...] if x_scale_ref is not None else None,
            preferred_element_type=jnp.float32,
        ).astype(o_vmem.dtype)
        _seed_copy(ti, buf, rs).start()
        for entry in steps[0]:
            _ici(ti, buf, rs, 0, entry).start()

    # ---- a1: tile i-1 -- drain step1 + seed, start step2 ----
    @pl.when(jnp.logical_and(i >= 1, (i - 1) < grid_n))
    def _a1():
        ti = i - 1
        buf = lax.rem(ti, jnp.int32(2))
        rs = lax.rem(ti, jnp.int32(ring))
        for entry in steps[0]:
            op = _ici(ti, buf, rs, 0, entry)
            op.wait_send()  # frees o_vmem[buf] for tile ti+2
            op.wait_recv()  # dist-1 slots present for step 2
        _seed_copy(ti, buf, rs).wait()
        for entry in steps[1]:
            _ici(ti, buf, rs, 1, entry).start()

    # ---- a2: tile i-2 -- drain step2, start step3 ----
    @pl.when(jnp.logical_and(i >= 2, (i - 2) < grid_n))
    def _a2():
        ti = i - 2
        buf = lax.rem(ti, jnp.int32(2))
        rs = lax.rem(ti, jnp.int32(ring))
        for entry in steps[1]:
            op = _ici(ti, buf, rs, 1, entry)
            op.wait_send()
            op.wait_recv()
        for entry in steps[2]:
            _ici(ti, buf, rs, 2, entry).start()

    # ---- a3: tile i-3 -- drain step3, start core-bridge swap ----
    @pl.when(jnp.logical_and(i >= 3, (i - 3) < grid_n))
    def _a3():
        ti = i - 3
        buf = lax.rem(ti, jnp.int32(2))
        rs = lax.rem(ti, jnp.int32(ring))
        for entry in steps[2]:
            op = _ici(ti, buf, rs, 2, entry)
            op.wait_send()
            op.wait_recv()  # antipode slot present for the bridge
        for bi, off in enumerate(ici_offsets):
            _bridge(ti, rs, bi, off).start()

    # ---- a4: tile i-4 -- drain the bridge (tile complete) ----
    @pl.when(jnp.logical_and(i >= 4, (i - 4) < grid_n))
    def _a4():
        ti = i - 4
        rs = lax.rem(ti, jnp.int32(ring))
        for bi, off in enumerate(ici_offsets):
            op = _bridge(ti, rs, bi, off)
            op.wait_send()
            op.wait_recv()  # opposite-N-half slot landed -> [T,N] complete for this tile


def _mm_ag_tp_intra_topo_kernel(*args, **kwargs):
    _mm_ag_tp_intra_topo_body(*args, **kwargs)


def _fused_prelude_router_mm_ag_tp_intra_topo_body(
    pos2phys_ref,
    phys2pos_ref,
    attn_ref,
    stream_ref,
    post_weight_ref,
    gate_ref,
    pre_weight_ref,
    router_weight_ref,
    expert_bias_ref,
    pooler_weight_ref,
    pre_expert_weight_ref,
    h_hbm_ref,
    pooled_hbm_ref,
    prepared_expert_hbm_ref,
    routing_weights_hbm_ref,
    routing_indices_hbm_ref,
    attn_tile_vmem,
    stream_tile_vmem,
    routing_weights_vmem,
    routing_indices_vmem,
    prepared_vmem,
    x_full,
    x_sib,
    x_full_scale,
    x_sib_scale,
    o_vmem,
    xchg_sem,
    xchg_scale_sem,
    seed_sems,
    ici_send_sems,
    ici_recv_sems,
    br_send_sems,
    br_recv_sems,
    attn_copy_sem,
    stream_copy_sem,
    h_copy_sem,
    routing_weights_copy_sem,
    routing_indices_copy_sem,
    norm_copy_sem,
    *,
    prelude_kwargs,
    router_kwargs,
    block_tokens: int,
    router_block_tokens: int,
    norm_block_tokens: int,
    norm_eps: float,
    has_pre_expert_weight: bool,
    norm_zero_centered_gamma: bool,
    norm_gain_center: float,
    **topo_kwargs,
):
    """Prelude, router, pooler, and pre-expert norm in one custom call."""
    i = pl.program_id(0)
    m_local = attn_ref.shape[0]
    num_token_blocks = m_local // block_tokens

    @pl.when(i == 0)
    def _prepare():
        for token_block in range(num_token_blocks):
            rows = pl.ds(token_block * block_tokens, block_tokens)
            attn_copy = pltpu.make_async_copy(
                src_ref=attn_ref.at[rows, :],
                dst_ref=attn_tile_vmem,
                sem=attn_copy_sem,
            )
            stream_copy = pltpu.make_async_copy(
                src_ref=stream_ref.at[rows, :],
                dst_ref=stream_tile_vmem,
                sem=stream_copy_sem,
            )
            attn_copy.start()
            stream_copy.start()
            attn_copy.wait()
            stream_copy.wait()
            h, _, prepared = _prelude_values(
                attn_tile_vmem[...],
                stream_tile_vmem[...],
                post_weight_ref[...],
                gate_ref[...],
                pre_weight_ref[...],
                **prelude_kwargs,
            )
            prepared_vmem[rows, :] = prepared
            # Both input tiles are dead after the prepared value is committed.
            # Reusing one for h permits a 256-row prelude tile without adding
            # another hidden-sized VMEM allocation.
            stream_tile_vmem[...] = h
            h_copy = pltpu.make_async_copy(
                src_ref=stream_tile_vmem,
                dst_ref=h_hbm_ref.at[rows, :],
                sem=h_copy_sem,
            )
            h_copy.start()
            h_copy.wait()

    def _route_while_xchg_runs():
        num_router_blocks = m_local // router_block_tokens
        for token_block in range(num_router_blocks):
            rows = pl.ds(token_block * router_block_tokens, router_block_tokens)
            routing_weights, routing_indices = route_hidden_block(
                prepared_vmem[rows, :],
                router_weight_ref[...],
                expert_bias_ref[...],
                **router_kwargs,
            )
            topk = router_kwargs["topk"]
            routing_weights_vmem[rows, :] = routing_weights[:, :topk]
            routing_indices_vmem[rows, :] = routing_indices[:, :topk].astype(jnp.uint8)
        pltpu.make_async_copy(
            src_ref=routing_weights_vmem,
            dst_ref=routing_weights_hbm_ref,
            sem=routing_weights_copy_sem,
        ).start()
        pltpu.make_async_copy(
            src_ref=routing_indices_vmem,
            dst_ref=routing_indices_hbm_ref,
            sem=routing_indices_copy_sem,
        ).start()

    _mm_ag_tp_intra_topo_body(
        pos2phys_ref,
        phys2pos_ref,
        prepared_vmem,
        pooler_weight_ref,
        None,
        pooled_hbm_ref,
        x_full,
        x_sib,
        x_full_scale,
        x_sib_scale,
        o_vmem,
        xchg_sem,
        xchg_scale_sem,
        seed_sems,
        ici_send_sems,
        ici_recv_sems,
        br_send_sems,
        br_recv_sems,
        xchg_overlap_fn=_route_while_xchg_runs,
        **topo_kwargs,
    )

    @pl.when(i == 0)
    def _drain_output_copies():
        pltpu.make_async_copy(
            src_ref=routing_weights_vmem,
            dst_ref=routing_weights_hbm_ref,
            sem=routing_weights_copy_sem,
        ).wait()
        pltpu.make_async_copy(
            src_ref=routing_indices_vmem,
            dst_ref=routing_indices_hbm_ref,
            sem=routing_indices_copy_sem,
        ).wait()

    pooler_grid_end = topo_kwargs["grid_n"] + 4
    norm_program = i - pooler_grid_end
    norm_grid = pooled_hbm_ref.shape[0] // norm_block_tokens

    @pl.when(jnp.logical_and(i >= pooler_grid_end, norm_program < norm_grid))
    def _pre_expert_norm():
        rows = pl.ds(norm_program * norm_block_tokens, norm_block_tokens)
        pooled_hidden = pooled_hbm_ref.shape[1]
        norm_tile_vmem = x_full.at[
            pl.ds(0, norm_block_tokens),
            pl.ds(0, pooled_hidden),
        ]
        norm_copy = pltpu.make_async_copy(
            src_ref=pooled_hbm_ref.at[rows, :],
            dst_ref=norm_tile_vmem,
            sem=norm_copy_sem,
        )
        norm_copy.start()
        norm_copy.wait()

        pooled = norm_tile_vmem[...]
        pooled_f32 = pooled.astype(jnp.float32)
        variance = jnp.mean(pooled_f32 * pooled_f32, axis=1, keepdims=True)
        normalized = (pooled_f32 * jax.lax.rsqrt(variance + norm_eps)).astype(
            pooled.dtype
        )
        if has_pre_expert_weight:
            weight = pre_expert_weight_ref[...]
            if norm_zero_centered_gamma:
                weight = weight + norm_gain_center
            normalized = normalized * weight
        norm_tile_vmem[...] = normalized.astype(pooled.dtype)

        @functools.partial(
            pl.run_scoped,
            prepared_expert_vmem=pltpu.VMEM(
                (norm_block_tokens, pooled_hidden // 128, 128),
                pooled.dtype,
            ),
        )
        def _store_prepared(prepared_expert_vmem):
            prepared_expert_vmem[...] = normalized.reshape(
                norm_block_tokens, pooled_hidden // 128, 128
            ).astype(pooled.dtype)
            prepared_copy = pltpu.make_async_copy(
                src_ref=prepared_expert_vmem,
                dst_ref=prepared_expert_hbm_ref.at[rows, :, :],
                sem=norm_copy_sem,
            )
            prepared_copy.start()
            prepared_copy.wait()


def _fused_prelude_router_mm_ag_tp_intra_topo_kernel(*args, **kwargs):
    _fused_prelude_router_mm_ag_tp_intra_topo_body(*args, **kwargs)


def _matmul_all_gather_tp_intra_topo(
    x: jax.Array,  # [T, K] EP-sharded P((chip, core), None) -> local [T/16, K]
    w: jax.Array,  # [N, K] sharded on N over `core` P(core, None) -> local [N/2, K]
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
    x_scale: jax.Array | None = None,
) -> jax.Array:
    """COMBINE: pre-matmul core-bridge token exchange + half-W matmul + topo bridge-last routed
    gather -> [T, N] replicated. Works on ANY 16-core 2x2x2 slice (routes by physical coord), as long
    as ``core_axis`` (size 2) is the PHYSICAL on-chip pair (bit 0) so the weight N-half split lines up
    with the core-bridge -- the builder verifies this and raises otherwise.

    ``w_transposed=True`` -> ``w`` is ``[N, K]`` (sharded on N = dim 0 over ``core_axis``); ``False``
    -> ``[K, N]`` (dim 1). ``x_scale`` (per-token row scale for a RAW fp8 ``x``) is passed INTO the
    kernel (+ exchanged over the core-bridge with x) and dequanted in-VMEM at the matmul; ``x`` stays
    fp8 on the wire (mirrors matmul_replicated_weights_all_gather._matmul_topo)."""
    axis_names = (chip_axis, core_axis)
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    if tp_size != NUM_DEVICES:
        raise ValueError(
            f"tp-intra-topo needs a {NUM_DEVICES}-core mesh, got {tp_size}."
        )
    mesh_axis_names = tuple(mesh.axis_names)
    core_size = mesh.shape[core_axis]
    if core_size != 2:
        raise ValueError(
            f"tp-intra-topo needs a size-2 core axis, got {core_axis}={core_size}"
        )
    # Physical topology from the device coords (any 2x2x2 order) -> raveled <-> physical maps.
    pos_to_phys, phys_to_pos = _physical_topo_maps(mesh, axis_names)
    # tp-intra requires core_axis == the physical on-chip pair (bit 0): the weight N-half (raveled
    # bit 0) must equal the physical core so the core-bridge combines the two N-halves. Verify the
    # permutation preserves bit 0.
    if any((pos_to_phys[p] & 1) != (p & 1) for p in range(NUM_DEVICES)):
        raise ValueError(
            "tp-intra-topo needs core_axis to be the PHYSICAL on-chip pair (bit 0); the mesh's "
            "raveled core index does not match the physical on-chip core. Build the mesh so the "
            "size-2 core_axis groups the 2 physically-adjacent cores."
        )
    m, k = x.shape
    # w sharded on N over core_axis: N = dim 0 for [N, K] (w_transposed), dim 1 for [K, N].
    n = w.shape[0] if w_transposed else w.shape[1]
    half_n = n // core_size  # local N-half
    m_local = m // tp_size
    m_chip = core_size * m_local  # = 2 * m_local (the chip's token block height, T/8)
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
        # Size bn against the gathered [T, bn] band (t = m), but it must divide the N-HALF.
        bn = _topo_bn(half_n, m, k, w.dtype.itemsize, jnp.dtype(od).itemsize)
    if half_n % bn != 0:
        raise ValueError(f"half_n ({half_n}) must be divisible by bn ({bn}).")
    grid_n = half_n // bn
    # W N-tile: [bn, K] on the N (row) dim when transposed ([N, K]); else [K, bn] on dim 1 ([K, N]).
    # (index_map takes `i, *_` -- the 2 scalar-prefetch args follow the grid index.)
    w_spec = (
        pl.BlockSpec((bn, k), lambda i, *_: (jnp.minimum(i, grid_n - 1), 0))
        if w_transposed
        else pl.BlockSpec((k, bn), lambda i, *_: (0, jnp.minimum(i, grid_n - 1)))
    )
    steps, n_ici = _flatten_ici_schedule()
    ici_offsets = _ici_offsets()
    n_bridge = len(ici_offsets)
    all_masks = ICI_MASKS + (BRIDGE_MASK,)
    _split = antipode_core_split()
    (_l0, _x0), (_l1, _x1) = _split[0], _split[1]
    antipode_split = (_l0, _x0, _l1, _x1)
    ring = _TOPO_RING
    pos2phys = jnp.asarray(pos_to_phys, jnp.int32)
    phys2pos = jnp.asarray(phys_to_pos, jnp.int32)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,  # pos2phys, phys2pos
        in_specs=[
            pl.BlockSpec((m_local, k), lambda i, *_: (0, 0)),  # x (resident EP slice)
            w_spec,  # w N-half tile ([bn,k] or [k,bn] per w_transposed)
            (
                pl.BlockSpec((m_local, 1), lambda i, *_: (0, 0))
                if has_x_scale
                else None
            ),  # per-token scale [m_local, 1], or None (no scale)
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=(
            pltpu.VMEM(
                (m_chip, k), x.dtype
            ),  # x_full (chip token block; raw fp8 if scaled)
            pltpu.VMEM((m_local, k), x.dtype),  # x_sib (sibling slice, received whole)
            pltpu.VMEM((m_chip, 1), sdt),  # x_full_scale (chip per-token scales)
            pltpu.VMEM((m_local, 1), sdt),  # x_sib_scale (sibling per-token scales)
            pltpu.VMEM((2, m_chip, bn), od),  # o_vmem (double-buffered [T/8, bn] piece)
            pltpu.SemaphoreType.DMA,  # xchg_sem
            pltpu.SemaphoreType.DMA,  # xchg_scale_sem
            pltpu.SemaphoreType.DMA((ring,)),  # seed
            pltpu.SemaphoreType.DMA((ring, n_ici)),  # ici send
            pltpu.SemaphoreType.DMA((ring, n_ici)),  # ici recv
            pltpu.SemaphoreType.DMA((ring, n_bridge)),  # bridge send
            pltpu.SemaphoreType.DMA((ring, n_bridge)),  # bridge recv
        ),
        grid=(grid_n + 4,),
    )

    def _call(*operands):
        # operands = (p2ph, ph2p, xx, ww[, ss]); ss (+ its BlockSpec) is None when there is no scale
        # -> the kernel's x_scale_ref is None. x stays RAW fp8 into the kernel; the per-token scale is
        # exchanged over the core-bridge with x and dequant is in-VMEM at the matmul.
        p2ph, ph2p, xx, ww = operands[0], operands[1], operands[2], operands[3]
        ss = operands[4] if has_x_scale else None
        return pl.pallas_call(
            functools.partial(
                _mm_ag_tp_intra_topo_kernel,
                steps=steps,
                all_masks=all_masks,
                ici_offsets=ici_offsets,
                antipode_split=antipode_split,
                axis_names=axis_names,
                strides=strides,
                sizes=sizes,
                mesh_axis_names=mesh_axis_names,
                m_local=m_local,
                m_chip=m_chip,
                half_n=half_n,
                bn=bn,
                grid_n=grid_n,
                ring=ring,
                double_barrier=double_barrier,
                w_transposed=w_transposed,
            ),
            out_shape=jax.ShapeDtypeStruct((m, n), od),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=128 * 1024 * 1024,
                disable_bounds_checks=disable_bounds_checks,
            ),
            name=f"mm_ag_tp_intra_topo_bn{bn}",
        )(p2ph, ph2p, xx, ww, ss)

    # w sharded on N over core_axis: dim 0 for [N, K] (w_transposed), dim 1 for [K, N].
    w_shard = P(core_axis, None) if w_transposed else P(None, core_axis)
    # x_scale ([m, 1] per-token row scale for a raw fp8 x) is a shard_map input, sharded on tokens;
    # omitted (with its BlockSpec) when there is no scale so the kernel's x_scale_ref is None.
    sm_in_specs = [P(None), P(None), P(axis_names, None), w_shard]
    operands = [pos2phys, phys2pos, x, w]
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


def fused_attention_moe_prelude_router_pooler_topo(
    attn_out: jax.Array,
    residual_stream: jax.Array,
    post_weight: jax.Array,
    gate: jax.Array,
    pre_weight: jax.Array,
    router_weight: jax.Array,
    expert_bias: jax.Array,
    pooler_weight: jax.Array,
    pre_expert_weight: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    chip_axis: str,
    core_axis: str,
    *,
    has_post_weight: bool,
    post_eps: float,
    pre_eps: float,
    topk: int,
    temperature: float = 0.3,
    residual_eps: float = 1e-3,
    router_output_multiplier: float = 1.0,
    route_scale: float = 1.0,
    division_eps: float = 0.0,
    score_normalization: str = "sigmoid",
    has_pre_expert_weight: bool = True,
    norm_eps: float = 1e-5,
    norm_zero_centered_gamma: bool = False,
    norm_gain_center: float = 1.0,
    bn: int = 128,
    collective_id: int = 0,
    double_barrier: bool = True,
    disable_bounds_checks: bool = False,
):
    """Fuse attention-to-MoE prelude, router, pooler, and pre-expert norm.

    The prepared pre-FFN activation stays in VMEM. Router top-k runs while the
    prepared activation is exchanged over the on-chip core bridge, after which
    the existing half-W matmul and topology-aware output gather produce a
    replicated pooler output. A final token-block phase normalizes that output
    into the representation consumed by fused-RS GMM.
    """
    if attn_out.shape != residual_stream.shape or attn_out.ndim != 2:
        raise ValueError("attn_out and residual_stream must be equal rank-2 shapes")
    m, k = attn_out.shape
    if post_weight.shape != (k,) or gate.shape != (k,) or pre_weight.shape != (k,):
        raise ValueError("prelude vectors must match the activation hidden dimension")
    if router_weight.ndim != 2 or router_weight.shape[1] != k:
        raise ValueError("router_weight must have shape [num_experts, hidden]")
    num_experts = router_weight.shape[0]
    if expert_bias.shape != (num_experts,):
        raise ValueError("expert_bias must match router_weight's expert dimension")
    if pooler_weight.ndim != 2 or pooler_weight.shape[1] != k:
        raise ValueError("pooler_weight must have shape [pooled_hidden, hidden]")
    n = pooler_weight.shape[0]
    if n > k:
        raise ValueError("pooled hidden size cannot exceed the input hidden size")
    if pre_expert_weight.shape != (n,):
        raise ValueError(
            f"pre_expert_weight must have shape {(n,)}, got {pre_expert_weight.shape}"
        )
    if k % 256 or num_experts % 128:
        raise ValueError("fused router requires 256-aligned hidden and experts")
    if not 1 <= topk <= min(8, num_experts):
        raise ValueError(f"topk must be in [1, {min(8, num_experts)}]")
    if score_normalization not in ("sigmoid", "softmax"):
        raise ValueError(f"unsupported score normalization {score_normalization!r}")

    axis_names = (chip_axis, core_axis)
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    if tp_size != NUM_DEVICES:
        raise ValueError(f"fused topo kernel needs {NUM_DEVICES} cores, got {tp_size}")
    mesh_axis_names = tuple(mesh.axis_names)
    core_size = mesh.shape[core_axis]
    if core_size != 2:
        raise ValueError(f"fused topo kernel needs {core_axis}=2, got {core_size}")
    pos_to_phys, phys_to_pos = _physical_topo_maps(mesh, axis_names)
    if any((pos_to_phys[p] & 1) != (p & 1) for p in range(NUM_DEVICES)):
        raise ValueError("core_axis must be the physical on-chip core pair")

    m_local = m // tp_size
    router_block_tokens = 128
    if m_local % router_block_tokens:
        raise ValueError(
            f"local token count {m_local} must be divisible by {router_block_tokens}"
        )
    block_tokens = router_block_tokens
    half_n = n // core_size
    if half_n % bn:
        raise ValueError(f"half pooler width {half_n} must be divisible by bn={bn}")
    m_chip = core_size * m_local
    grid_n = half_n // bn
    norm_block_tokens = min(m_chip, 512)
    if m % norm_block_tokens:
        raise ValueError(
            f"token count {m} must be divisible by {norm_block_tokens} for norm"
        )
    norm_grid = m // norm_block_tokens
    steps, n_ici = _flatten_ici_schedule()
    ici_offsets = _ici_offsets()
    n_bridge = len(ici_offsets)
    all_masks = ICI_MASKS + (BRIDGE_MASK,)
    split = antipode_core_split()
    (l0, x0), (l1, x1) = split[0], split[1]
    antipode_split = (l0, x0, l1, x1)
    ring = _TOPO_RING
    pos2phys = jnp.asarray(pos_to_phys, jnp.int32)
    phys2pos = jnp.asarray(phys_to_pos, jnp.int32)

    resident_matrix = lambda shape: pl.BlockSpec(shape, lambda i, *_: (0, 0))
    resident_vector = lambda shape: pl.BlockSpec(shape, lambda i, *_: (0,))
    pooler_weight_spec = pl.BlockSpec(
        (bn, k), lambda i, *_: (jnp.minimum(i, grid_n - 1), 0)
    )
    hbm_input = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    hbm_output = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    out_specs = [hbm_output] * 5
    scratch_shapes = [
        pltpu.VMEM((block_tokens, k), attn_out.dtype),
        pltpu.VMEM((block_tokens, k), residual_stream.dtype),
        pltpu.VMEM((m_local, topk), jnp.float32),
        pltpu.VMEM((m_local, topk), jnp.uint8),
        pltpu.VMEM((m_local, k), attn_out.dtype),
        pltpu.VMEM((m_chip, k), attn_out.dtype),
        pltpu.VMEM((m_local, k), attn_out.dtype),
        pltpu.VMEM((m_chip, 1), jnp.float32),
        pltpu.VMEM((m_local, 1), jnp.float32),
        pltpu.VMEM((2, m_chip, bn), attn_out.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((ring,)),
        pltpu.SemaphoreType.DMA((ring, n_ici)),
        pltpu.SemaphoreType.DMA((ring, n_ici)),
        pltpu.SemaphoreType.DMA((ring, n_bridge)),
        pltpu.SemaphoreType.DMA((ring, n_bridge)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=(
            hbm_input,
            hbm_input,
            resident_vector((k,)),
            resident_vector((k,)),
            resident_vector((k,)),
            resident_matrix((num_experts, k)),
            resident_vector((num_experts,)),
            pooler_weight_spec,
            resident_vector((n,)),
        ),
        out_specs=tuple(out_specs),
        scratch_shapes=tuple(scratch_shapes),
        grid=(grid_n + 4 + norm_grid,),
    )
    prelude_kwargs = {
        "has_post_weight": has_post_weight,
        "post_eps": post_eps,
        "pre_eps": pre_eps,
        "temperature": temperature,
        "residual_eps": residual_eps,
        "has_pre_weight": True,
    }
    router_kwargs = {
        "num_experts": num_experts,
        "topk": topk,
        "router_output_multiplier": router_output_multiplier,
        "route_scale": route_scale,
        "division_eps": division_eps,
        "score_normalization": score_normalization,
    }
    topo_kwargs = {
        "steps": steps,
        "all_masks": all_masks,
        "ici_offsets": ici_offsets,
        "antipode_split": antipode_split,
        "axis_names": axis_names,
        "strides": strides,
        "sizes": sizes,
        "mesh_axis_names": mesh_axis_names,
        "m_local": m_local,
        "m_chip": m_chip,
        "half_n": half_n,
        "bn": bn,
        "grid_n": grid_n,
        "ring": ring,
        "double_barrier": double_barrier,
        "w_transposed": True,
    }

    def _call(p2ph, ph2p, aa, ss, pw, gg, prew, rw, eb, poolw, normw):
        return pl.pallas_call(
            functools.partial(
                _fused_prelude_router_mm_ag_tp_intra_topo_kernel,
                prelude_kwargs=prelude_kwargs,
                router_kwargs=router_kwargs,
                block_tokens=block_tokens,
                router_block_tokens=router_block_tokens,
                norm_block_tokens=norm_block_tokens,
                norm_eps=norm_eps,
                has_pre_expert_weight=has_pre_expert_weight,
                norm_zero_centered_gamma=norm_zero_centered_gamma,
                norm_gain_center=norm_gain_center,
                **topo_kwargs,
            ),
            out_shape=(
                jax.ShapeDtypeStruct((m_local, k), residual_stream.dtype),
                jax.ShapeDtypeStruct((m, n), residual_stream.dtype),
                jax.ShapeDtypeStruct((m, n // 128, 128), residual_stream.dtype),
                jax.ShapeDtypeStruct((m_local, topk), jnp.float32),
                jax.ShapeDtypeStruct((m_local, topk), jnp.uint8),
            ),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=128 * 1024 * 1024,
                disable_bounds_checks=disable_bounds_checks,
            ),
            name=f"fused_attention_router_mm_ag_topo_bn{bn}",
        )(p2ph, ph2p, aa, ss, pw, gg, prew, rw, eb, poolw, normw)

    output_specs = [
        P(axis_names, None),
        P(None, None),
        P(None, None, None),
        P(axis_names, None),
        P(axis_names, None),
    ]
    return jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(None),
                P(None),
                P(axis_names, None),
                P(axis_names, None),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(core_axis, None),
                P(),
            ),
            out_specs=tuple(output_specs),
            check_vma=False,
        )
    )(
        pos2phys,
        phys2pos,
        attn_out,
        residual_stream,
        post_weight,
        gate,
        pre_weight,
        router_weight,
        expert_bias,
        pooler_weight,
        pre_expert_weight,
    )
