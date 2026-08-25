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

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    cost_estimate_metadata,
    estimate_activation_compute_cost,
    logical_comm_bytes,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_attention_moe_prelude import (
    _prelude_values,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_router import (
    estimate_moe_routing_compute_cost,
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


def get_matmul_all_gather_cost_estimate(
    *,
    m: int,
    k: int,
    n: int,
    tp_size: int,
    core_size: int,
    x_dtype,
    w_dtype,
    out_dtype,
    x_scale_dtype=None,
) -> pl.CostEstimate:
    """Estimate one rank's half-weight matmul and logical gathered payload."""
    m_local = m // tp_size
    m_chip = core_size * m_local
    half_n = n // core_size
    x_bytes = m_local * k * jnp.dtype(x_dtype).itemsize
    x_scale_bytes = (
        m_local * jnp.dtype(x_scale_dtype).itemsize if x_scale_dtype is not None else 0
    )
    local_output_bytes = m_chip * half_n * jnp.dtype(out_dtype).itemsize
    gathered_output_bytes = m * n * jnp.dtype(out_dtype).itemsize
    # Later topology stages forward already-written HBM pieces: 3 reads in ICI
    # step 2, 1 in step 3, and 8 in the on-chip bridge.
    forwarding_read_bytes = 12 * local_output_bytes
    remote_bytes_transferred = logical_comm_bytes(
        x_bytes + x_scale_bytes + local_output_bytes,
        x_bytes + x_scale_bytes + gathered_output_bytes,
    )
    return pl.CostEstimate(
        flops=int(2 * m_chip * k * half_n),
        bytes_accessed=int(
            x_bytes
            + half_n * k * jnp.dtype(w_dtype).itemsize
            + x_scale_bytes
            + gathered_output_bytes
            + forwarding_read_bytes
        ),
        transcendentals=0,
        remote_bytes_transferred=int(remote_bytes_transferred),
    )


def get_matmul_all_gather_cost_metadata(
    cost_estimate: pl.CostEstimate,
) -> dict[str, str]:
    """Describe the topology matmul/all-gather cost estimate."""
    return cost_estimate_metadata(
        flops_basis="static per-rank half-weight matmul shapes",
        bytes_accessed_basis=(
            "modeled HBM input reads, replicated-output writes, and topology "
            "forwarding reads"
        ),
        remote_bytes_transferred=cost_estimate.remote_bytes_transferred,
        remote_bytes_transferred_basis=(
            "logical max of communicated input and output payloads across "
            "the core exchange and topology gather; not physical TPU-link traffic"
        ),
    )


def get_fused_attention_moe_prelude_cost_estimate(
    *,
    m: int,
    k: int,
    n: int,
    num_experts: int,
    topk: int,
    tp_size: int,
    core_size: int,
    block_tokens: int,
    norm_block_tokens: int,
    attn_dtype,
    residual_dtype,
    post_weight_dtype,
    gate_dtype,
    pre_weight_dtype,
    router_weight_dtype,
    expert_bias_dtype,
    pooler_weight_dtype,
    pre_expert_weight_dtype,
    has_post_weight: bool,
    has_pre_expert_weight: bool,
    norm_zero_centered_gamma: bool,
    score_normalization: str,
) -> pl.CostEstimate:
    """Estimate the fused prelude, router, pooler, and pre-expert norm."""
    m_local = m // tp_size
    m_chip = core_size * m_local
    half_n = n // core_size
    num_token_blocks = m_local // block_tokens
    norm_grid = m // norm_block_tokens

    prelude_elements = m_local * k
    gate_sigmoid_flops, gate_sigmoid_transcendentals = estimate_activation_compute_cost(
        "sigmoid", 2 * k * num_token_blocks
    )
    prelude_flops = (
        10 * prelude_elements
        + 2 * m_local
        + 4 * k * num_token_blocks
        + gate_sigmoid_flops
    )
    if has_post_weight:
        prelude_flops += prelude_elements
    prelude_transcendentals = (
        2 * m_local + gate_sigmoid_transcendentals + k * num_token_blocks
    )

    router_flops, router_transcendentals = estimate_moe_routing_compute_cost(
        tokens=m_local,
        num_experts=num_experts,
        topk=topk,
        score_normalization=score_normalization,
    )
    router_flops += 2 * m_local * k * num_experts

    pooler_flops = 2 * m_chip * k * half_n
    norm_elements = m * n
    norm_flops = 3 * norm_elements + m
    if has_pre_expert_weight:
        norm_flops += norm_elements
        if norm_zero_centered_gamma:
            norm_flops += n * norm_grid
    norm_transcendentals = m

    attn_itemsize = jnp.dtype(attn_dtype).itemsize
    residual_itemsize = jnp.dtype(residual_dtype).itemsize
    bytes_accessed = (
        m_local * k * attn_itemsize
        + m_local * k * residual_itemsize
        # These operands are loaded into resident VMEM refs once by program 0;
        # their repeated uses inside the token loop are VMEM traffic, not HBM.
        + (k * jnp.dtype(post_weight_dtype).itemsize if has_post_weight else 0)
        + k * jnp.dtype(gate_dtype).itemsize
        + k * jnp.dtype(pre_weight_dtype).itemsize
        + num_experts * k * jnp.dtype(router_weight_dtype).itemsize
        + num_experts * jnp.dtype(expert_bias_dtype).itemsize
        + half_n * k * jnp.dtype(pooler_weight_dtype).itemsize
        + (
            norm_grid * n * jnp.dtype(pre_expert_weight_dtype).itemsize
            if has_pre_expert_weight
            else 0
        )
        + m_local * k * residual_itemsize
        # pooled_hbm is written by the gather, read for normalization, and
        # returned; prepared_expert_hbm is a separate output write.
        + 3 * m * n * attn_itemsize
        + m_local
        * topk
        * (jnp.dtype(jnp.float32).itemsize + jnp.dtype(jnp.uint8).itemsize)
    )
    local_activation_bytes = m_local * k * attn_itemsize
    local_pooler_output_bytes = m_chip * half_n * attn_itemsize
    gathered_pooler_output_bytes = m * n * attn_itemsize
    forwarding_read_bytes = 12 * local_pooler_output_bytes
    bytes_accessed += forwarding_read_bytes
    remote_bytes_transferred = logical_comm_bytes(
        local_activation_bytes + local_pooler_output_bytes,
        local_activation_bytes + gathered_pooler_output_bytes,
    )
    return pl.CostEstimate(
        flops=int(prelude_flops + router_flops + pooler_flops + norm_flops),
        bytes_accessed=int(bytes_accessed),
        transcendentals=int(
            prelude_transcendentals + router_transcendentals + norm_transcendentals
        ),
        remote_bytes_transferred=int(remote_bytes_transferred),
    )


def get_fused_attention_moe_prelude_cost_metadata(
    cost_estimate: pl.CostEstimate,
) -> dict[str, str]:
    """Describe the fused prelude/router/pooler topology cost estimate."""
    return cost_estimate_metadata(
        flops_basis=(
            "static per-rank prelude, router, half-weight pooler, and norm shapes"
        ),
        bytes_accessed_basis=(
            "modeled HBM reads and writes, including repeated resident-vector "
            "reads, pooled-output reread, and topology forwarding reads"
        ),
        remote_bytes_transferred=cost_estimate.remote_bytes_transferred,
        remote_bytes_transferred_basis=(
            "logical max of communicated input and output payloads across "
            "the core exchange and topology gather; not physical TPU-link traffic"
        ),
    )


def get_post_moe_epilogue_cost_estimate(
    *,
    m: int,
    k: int,
    n: int,
    tp_size: int,
    core_size: int,
    x_dtype,
    w_dtype,
    residual_dtype,
    norm_weight_dtype,
    residual_gate_dtype,
    unpooler_bias_dtype,
    bn: int,
    has_norm_weight: bool,
    norm_weight_before: bool,
    norm_zero_centered_gamma: bool,
    has_unpooler_bias: bool,
    use_residual_gate: bool,
) -> pl.CostEstimate:
    """Estimate the two-pass unpooler and fused post-MoE epilogue."""
    m_local = m // tp_size
    m_chip = core_size * m_local
    half_n = n // core_size
    grid_n = half_n // bn
    local_elements = m_chip * half_n

    flops = 4 * m_chip * k * half_n
    # First-pass sum of squares plus the second-pass normalization multiply.
    flops += 3 * local_elements + m_chip
    flops += 2 * m_chip * grid_n
    if has_unpooler_bias:
        flops += 2 * local_elements
    if has_norm_weight:
        flops += 2 * local_elements if norm_weight_before else local_elements
        if norm_zero_centered_gamma:
            flops += 2 * half_n if norm_weight_before else half_n
    if use_residual_gate:
        gate_sigmoid_flops, gate_sigmoid_transcendentals = (
            estimate_activation_compute_cost("sigmoid", 2 * half_n)
        )
        flops += 4 * half_n + gate_sigmoid_flops + 3 * local_elements
        transcendentals = m_chip * grid_n + gate_sigmoid_transcendentals + half_n
    else:
        flops += local_elements
        transcendentals = m_chip * grid_n

    x_itemsize = jnp.dtype(x_dtype).itemsize
    residual_itemsize = jnp.dtype(residual_dtype).itemsize
    norm_weight_reads = (
        half_n * (2 if norm_weight_before else 1) if has_norm_weight else 0
    )
    residual_gate_reads = half_n if use_residual_gate else 0
    unpooler_bias_reads = 2 * half_n if has_unpooler_bias else 0
    bytes_accessed = (
        m_local * k * x_itemsize
        + 2 * half_n * k * jnp.dtype(w_dtype).itemsize
        + m_local * n * residual_itemsize
        + norm_weight_reads * jnp.dtype(norm_weight_dtype).itemsize
        + residual_gate_reads * jnp.dtype(residual_gate_dtype).itemsize
        + unpooler_bias_reads * jnp.dtype(unpooler_bias_dtype).itemsize
        + m * n * residual_itemsize
    )
    x_exchange_bytes = m_local * k * x_itemsize
    residual_exchange_bytes = m_local * half_n * residual_itemsize
    sumsq_exchange_bytes = m_chip * jnp.dtype(jnp.float32).itemsize
    local_output_bytes = local_elements * residual_itemsize
    gathered_output_bytes = m * n * residual_itemsize
    forwarding_read_bytes = 12 * local_output_bytes
    bytes_accessed += forwarding_read_bytes
    remote_bytes_transferred = logical_comm_bytes(
        x_exchange_bytes
        + residual_exchange_bytes
        + sumsq_exchange_bytes
        + local_output_bytes,
        x_exchange_bytes
        + residual_exchange_bytes
        + sumsq_exchange_bytes
        + gathered_output_bytes,
    )
    return pl.CostEstimate(
        flops=int(flops),
        bytes_accessed=int(bytes_accessed),
        transcendentals=int(transcendentals),
        remote_bytes_transferred=int(remote_bytes_transferred),
    )


def get_post_moe_epilogue_cost_metadata(
    cost_estimate: pl.CostEstimate,
) -> dict[str, str]:
    """Describe the two-pass post-MoE topology cost estimate."""
    return cost_estimate_metadata(
        flops_basis="static two-pass half-weight matmul and post-MoE arithmetic",
        bytes_accessed_basis=(
            "modeled HBM reads and writes, including both weight passes and "
            "topology forwarding reads"
        ),
        remote_bytes_transferred=cost_estimate.remote_bytes_transferred,
        remote_bytes_transferred_basis=(
            "logical max of communicated input and output payloads across core "
            "exchanges and topology gather; not physical TPU-link traffic"
        ),
    )


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
    residual_ref=None,
    residual_sib=None,
    program_offset: int = 0,
    inputs_preloaded: bool = False,
    fuse_post_moe: bool = False,
    post_moe_sumsq=None,
    norm_weight_ref=None,
    residual_gate_ref=None,
    unpooler_bias_ref=None,
    norm_eps: float = 0.0,
    has_norm_weight: bool = False,
    norm_weight_before: bool = False,
    norm_zero_centered_gamma: bool = False,
    norm_gain_center: float = 1.0,
    has_unpooler_bias: bool = False,
    use_residual_gate: bool = False,
    residual_temperature: float = 1.0,
    residual_eps: float = 0.0,
):
    # Software-pipelined store-and-forward (topo's 5-stage schedule) on HALF-W matmul pieces.
    # Stage `a` operates on tile `i-a`; the token exchange happens once at i==0 before a0's matmul.
    # Routing is by PHYSICAL coord (pos2phys/phys2pos) so any 2x2x2 mesh order works; piece identity
    # + 2D placement stay in RAVELED space (the token/output layout). core_axis == physical bit 0
    # (verified in the builder), so the weight N-half == my_pos%2 lines up with the core-bridge.
    i = pl.program_id(0) - program_offset  # 0 .. grid_n + 3
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
        if not inputs_preloaded:
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
    @pl.when(jnp.logical_and(i >= 0, i < grid_n))
    def _a0():
        ti = i
        buf = lax.rem(ti, jnp.int32(2))
        rs = lax.rem(ti, jnp.int32(ring))
        branch = util.reorder_matmul(
            x_full[...],
            w_ref[...],
            w_transposed=w_transposed,  # [bn,k] ([N,K]) or [k,bn] ([K,N])
            # raw-fp8 x_full dequanted in-VMEM by its per-token scale at the matmul (None -> bf16 x).
            x_scale=x_full_scale[...] if x_scale_ref is not None else None,
            preferred_element_type=jnp.float32,
        ).astype(o_vmem.dtype)
        if fuse_post_moe:
            feature_col = core * half_n + ti * bn
            branch_f32 = branch.astype(jnp.float32)
            if has_unpooler_bias:
                branch_f32 = branch_f32 + unpooler_bias_ref.at[pl.ds(feature_col, bn)][
                    ...
                ].astype(jnp.float32)
            norm_weight = norm_weight_ref.at[pl.ds(feature_col, bn)][...].astype(
                jnp.float32
            )
            if norm_zero_centered_gamma:
                norm_weight = norm_weight + norm_gain_center
            if has_norm_weight and norm_weight_before:
                branch_f32 = branch_f32 * norm_weight
            normalized = branch_f32 * jax.lax.rsqrt(
                post_moe_sumsq[...] / (2 * half_n) + norm_eps
            )
            if has_norm_weight and not norm_weight_before:
                normalized = normalized * norm_weight
            normalized = normalized.astype(o_vmem.dtype).astype(jnp.float32)
            own_residual = residual_ref.at[:, pl.ds(feature_col, bn)][...]
            sibling_residual = residual_sib.at[:, pl.ds(ti * bn, bn)][...]
            residual_lo = jnp.where(
                core == jnp.int32(0), own_residual, sibling_residual
            )
            residual_hi = jnp.where(
                core == jnp.int32(0), sibling_residual, own_residual
            )
            residual_tile = jnp.concatenate([residual_lo, residual_hi], axis=0).astype(
                jnp.float32
            )
            if use_residual_gate:
                gate = (
                    residual_gate_ref.at[pl.ds(feature_col, bn)][...].astype(
                        jnp.float32
                    )
                    / residual_temperature
                )
                beta = jax.nn.sigmoid(gate)
                alpha_sq = jax.nn.sigmoid(-gate) * (1.0 + beta)
                alpha = jnp.sqrt(jnp.maximum(alpha_sq, residual_eps))
                branch = alpha * residual_tile + beta * normalized
            else:
                branch = residual_tile + normalized
            o_vmem.at[buf][...] = branch.astype(o_vmem.dtype)
        else:
            o_vmem.at[buf][...] = branch
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
            # The serving residual can be FP32 while the FFN activation is BF16.
            # Pallas Ref stores require an explicit conversion.
            prepared_vmem[rows, :] = prepared.astype(prepared_vmem.dtype)
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
        # The prepared FFN activation is dead after the pooler matmul and has
        # the same dtype as the pooler output. Reuse its leading columns for
        # norm so an FP32 residual stream does not promote the MoE activation
        # or require another hidden-sized VMEM allocation.
        norm_tile_vmem = prepared_vmem.at[
            pl.ds(0, norm_block_tokens),
            pl.ds(0, pooled_hidden),
        ]

        @functools.partial(
            pl.run_scoped,
            prepared_expert_vmem=pltpu.VMEM(
                (norm_block_tokens, pooled_hidden // 128, 128),
                pooled_hbm_ref.dtype,
            ),
        )
        def _normalize_and_store(prepared_expert_vmem):
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
    cost_estimate = get_matmul_all_gather_cost_estimate(
        m=m,
        k=k,
        n=n,
        tp_size=tp_size,
        core_size=core_size,
        x_dtype=x.dtype,
        w_dtype=w.dtype,
        out_dtype=od,
        x_scale_dtype=x_scale.dtype if has_x_scale else None,
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
            cost_estimate=cost_estimate,
            metadata=get_matmul_all_gather_cost_metadata(cost_estimate),
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
    router_block_tokens = min(m_local, 128)
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
    # The norm phase reuses the residual-stream tile after the prelude.
    norm_block_tokens = block_tokens
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
        # The MoE activation follows the attention/FFN dtype, independently of
        # the serving residual stream (which may remain FP32).
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
    cost_estimate = get_fused_attention_moe_prelude_cost_estimate(
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
        topk=topk,
        tp_size=tp_size,
        core_size=core_size,
        block_tokens=block_tokens,
        norm_block_tokens=norm_block_tokens,
        attn_dtype=attn_out.dtype,
        residual_dtype=residual_stream.dtype,
        post_weight_dtype=post_weight.dtype,
        gate_dtype=gate.dtype,
        pre_weight_dtype=pre_weight.dtype,
        router_weight_dtype=router_weight.dtype,
        expert_bias_dtype=expert_bias.dtype,
        pooler_weight_dtype=pooler_weight.dtype,
        pre_expert_weight_dtype=pre_expert_weight.dtype,
        has_post_weight=has_post_weight,
        has_pre_expert_weight=has_pre_expert_weight,
        norm_zero_centered_gamma=norm_zero_centered_gamma,
        score_normalization=score_normalization,
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
                jax.ShapeDtypeStruct((m, n), attn_out.dtype),
                jax.ShapeDtypeStruct((m, n // 128, 128), attn_out.dtype),
                jax.ShapeDtypeStruct((m_local, topk), jnp.float32),
                jax.ShapeDtypeStruct((m_local, topk), jnp.uint8),
            ),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=128 * 1024 * 1024,
                disable_bounds_checks=disable_bounds_checks,
            ),
            cost_estimate=cost_estimate,
            metadata=get_fused_attention_moe_prelude_cost_metadata(cost_estimate),
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


def _post_moe_stats_body(
    pos2phys_ref,
    phys2pos_ref,
    x_ref,
    w_ref,
    residual_ref,
    norm_weight_ref,
    unpooler_bias_ref,
    x_full,
    x_sib,
    residual_sib,
    sumsq_vmem,
    sumsq_sib,
    xchg_sem,
    residual_xchg_sem,
    sumsq_xchg_sem,
    *,
    axis_names: tuple[str, ...],
    strides: tuple[int, ...],
    sizes: tuple[int, ...],
    mesh_axis_names: tuple[str, ...],
    m_chip: int,
    half_n: int,
    bn: int,
    grid_n: int,
    double_barrier: bool,
    has_norm_weight: bool,
    norm_weight_before: bool,
    norm_zero_centered_gamma: bool,
    norm_gain_center: float,
    has_unpooler_bias: bool,
    **_,
):
    """First unpooler pass: assemble chip tokens and reduce RMS statistics."""
    i = pl.program_id(0)
    my_pos = util.combined_axis_index(axis_names, strides)
    my_phys = pos2phys_ref[my_pos]
    core = lax.rem(my_pos, jnp.int32(2))
    _peer = util.make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names)

    def _rav(mask):
        return phys2pos_ref[lax.bitwise_xor(my_phys, mask)]

    def _sibling():
        return _peer(_rav(jnp.int32(BRIDGE_MASK)))

    @pl.when(i == 0)
    def _prepare_inputs():
        util.hypercube_local_barrier([_sibling()], double_barrier=double_barrier)
        x_op = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=x_sib,
            send_sem=xchg_sem,
            recv_sem=xchg_sem,
            device_id=_sibling(),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        residual_op = pltpu.make_async_remote_copy(
            src_ref=residual_ref.at[:, pl.ds((1 - core) * half_n, half_n)],
            dst_ref=residual_sib,
            send_sem=residual_xchg_sem,
            recv_sem=residual_xchg_sem,
            device_id=_sibling(),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        x_op.start()
        residual_op.start()
        x_op.wait_send()
        x_op.wait_recv()
        residual_op.wait_send()
        residual_op.wait_recv()

        own_x = x_ref[...]
        sibling_x = x_sib[...]
        x_lo = jnp.where(core == jnp.int32(0), own_x, sibling_x)
        x_hi = jnp.where(core == jnp.int32(0), sibling_x, own_x)
        x_full[...] = jnp.concatenate([x_lo, x_hi], axis=0)

        sumsq_vmem[...] = jnp.zeros((m_chip, 1), dtype=jnp.float32)

    @pl.when(i < grid_n)
    def _accumulate():
        feature_col = core * half_n + i * bn
        branch = util.reorder_matmul(
            x_full[...],
            w_ref[...],
            w_transposed=True,
            preferred_element_type=jnp.float32,
        ).astype(residual_ref.dtype)
        branch_f32 = branch.astype(jnp.float32)
        if has_unpooler_bias:
            branch_f32 = branch_f32 + unpooler_bias_ref.at[pl.ds(feature_col, bn)][
                ...
            ].astype(jnp.float32)
        if has_norm_weight and norm_weight_before:
            norm_weight = norm_weight_ref.at[pl.ds(feature_col, bn)][...].astype(
                jnp.float32
            )
            if norm_zero_centered_gamma:
                norm_weight = norm_weight + norm_gain_center
            branch_f32 = branch_f32 * norm_weight
        sumsq_vmem[...] = sumsq_vmem[...] + jnp.sum(
            branch_f32 * branch_f32, axis=1, keepdims=True
        )

    @pl.when(i == grid_n)
    def _combine_core_halves():
        op = pltpu.make_async_remote_copy(
            src_ref=sumsq_vmem,
            dst_ref=sumsq_sib,
            send_sem=sumsq_xchg_sem,
            recv_sem=sumsq_xchg_sem,
            device_id=_sibling(),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        op.start()
        op.wait_send()
        op.wait_recv()
        sumsq_vmem[...] = sumsq_vmem[...] + sumsq_sib[...]


def _matmul_tp_intra_topo_post_moe_kernel(
    pos2phys_ref,
    phys2pos_ref,
    x_ref,
    w_ref,
    residual_ref,
    norm_weight_ref,
    residual_gate_ref,
    unpooler_bias_ref,
    output_hbm_ref,
    x_full,
    x_sib,
    x_full_scale,
    x_sib_scale,
    o_vmem,
    residual_sib,
    sumsq_vmem,
    sumsq_sib,
    xchg_sem,
    xchg_scale_sem,
    residual_xchg_sem,
    sumsq_xchg_sem,
    seed_sems,
    ici_send_sems,
    ici_recv_sems,
    br_send_sems,
    br_recv_sems,
    *,
    stats_kwargs,
    route_kwargs,
):
    _post_moe_stats_body(
        pos2phys_ref,
        phys2pos_ref,
        x_ref,
        w_ref,
        residual_ref,
        norm_weight_ref,
        unpooler_bias_ref,
        x_full,
        x_sib,
        residual_sib,
        sumsq_vmem,
        sumsq_sib,
        xchg_sem,
        residual_xchg_sem,
        sumsq_xchg_sem,
        **stats_kwargs,
    )
    _mm_ag_tp_intra_topo_body(
        pos2phys_ref,
        phys2pos_ref,
        x_ref,
        w_ref,
        None,
        output_hbm_ref,
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
        residual_ref=residual_ref,
        residual_sib=residual_sib,
        fuse_post_moe=True,
        post_moe_sumsq=sumsq_vmem,
        norm_weight_ref=norm_weight_ref,
        residual_gate_ref=residual_gate_ref,
        unpooler_bias_ref=unpooler_bias_ref,
        **route_kwargs,
    )


def matmul_intra_tp_all_gather_post_moe_epilogue(
    x: jax.Array,
    w: jax.Array,
    residual: jax.Array,
    norm_weight: jax.Array,
    residual_gate: jax.Array,
    unpooler_bias: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    chip_axis: str,
    core_axis: str,
    *,
    norm_eps: float,
    has_norm_weight: bool,
    norm_weight_before: bool,
    norm_zero_centered_gamma: bool,
    norm_gain_center: float,
    has_unpooler_bias: bool,
    use_residual_gate: bool,
    residual_temperature: float,
    residual_eps: float,
    bn: int | None = None,
    collective_id: int = 0,
    double_barrier: bool = True,
    disable_bounds_checks: bool = False,
) -> jax.Array:
    """Two-pass half-W unpooler with fused post-FFN norm and residual add.

    The first pass computes the per-token RMS statistic across the two output
    feature halves. The second pass recomputes each unpooler tile, normalizes it,
    combines the matching token-sharded residual tile, and sends only the final
    layer-output tile through the existing topology gather. No replicated
    residual or packed HBM temporary is materialized.
    """
    axis_names = (chip_axis, core_axis)
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    if tp_size != NUM_DEVICES:
        raise ValueError(f"post-MoE epilogue needs {NUM_DEVICES} cores, got {tp_size}")
    if mesh.shape[core_axis] != 2:
        raise ValueError("post-MoE epilogue requires a size-2 on-chip core axis")

    pos_to_phys, phys_to_pos = _physical_topo_maps(mesh, axis_names)
    if any((pos_to_phys[p] & 1) != (p & 1) for p in range(NUM_DEVICES)):
        raise ValueError("core_axis must match the physical on-chip core pair")

    if x.ndim != 2 or w.ndim != 2 or residual.ndim != 2:
        raise ValueError("x, w, and residual must be rank-2")
    m, k = x.shape
    n = w.shape[0]
    if w.shape[1] != k:
        raise ValueError("post-MoE epilogue expects CPL-native w=[N,K]")
    if residual.shape != (m, n):
        raise ValueError(f"residual must have shape {(m, n)}, got {residual.shape}")
    for name, value in (
        ("norm_weight", norm_weight),
        ("residual_gate", residual_gate),
        ("unpooler_bias", unpooler_bias),
    ):
        if value.shape != (n,):
            raise ValueError(f"{name} must have shape {(n,)}, got {value.shape}")
    core_size = int(mesh.shape[core_axis])
    m_local = m // tp_size
    m_chip = core_size * m_local
    half_n = n // core_size
    if bn is None:
        bn = _topo_bn(half_n, m, k, w.dtype.itemsize, x.dtype.itemsize)
        # The T=8192 serving epilogue carries FP32 expert and residual streams.
        # bn=512 makes its full-chip activation/residual scratch plus the
        # double-buffered W/output tiles exceed v7x's 64 MiB VMEM by ~5.7 MiB.
        # Halving only the streamed feature tile preserves FP32 residual math
        # and brings the program below the hardware limit.
        if m_local >= 512 and (x.dtype.itemsize > 2 or residual.dtype.itemsize > 2):
            bn = min(bn, 256)
    if half_n % bn:
        raise ValueError(f"half_n ({half_n}) must be divisible by bn ({bn})")
    if bn % 256:
        raise ValueError("post-MoE epilogue bn must be a multiple of 256")
    grid_n = half_n // bn
    steps, n_ici = _flatten_ici_schedule()
    ici_offsets = _ici_offsets()
    n_bridge = len(ici_offsets)
    split = antipode_core_split()
    (l0, x0), (l1, x1) = split[0], split[1]
    ring = _TOPO_RING
    pos2phys = jnp.asarray(pos_to_phys, jnp.int32)
    phys2pos = jnp.asarray(phys_to_pos, jnp.int32)
    mesh_axis_names = tuple(mesh.axis_names)

    route_offset = grid_n + 1

    def _weight_index(i, *_):
        stats_tile = jnp.minimum(i, grid_n - 1)
        route_tile = jnp.minimum(jnp.maximum(i - route_offset, 0), grid_n - 1)
        return (jnp.where(i < route_offset, stats_tile, route_tile), 0)

    w_spec = pl.BlockSpec((bn, k), _weight_index)
    hbm_output = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=(
            pl.BlockSpec((m_local, k), lambda i, *_: (0, 0)),
            w_spec,
            pl.BlockSpec((m_local, n), lambda i, *_: (0, 0)),
            pl.BlockSpec((n,), lambda i, *_: (0,)),
            pl.BlockSpec((n,), lambda i, *_: (0,)),
            pl.BlockSpec((n,), lambda i, *_: (0,)),
        ),
        out_specs=hbm_output,
        scratch_shapes=(
            pltpu.VMEM((m_chip, k), x.dtype),
            pltpu.VMEM((m_local, k), x.dtype),
            pltpu.VMEM((m_chip, 1), jnp.float32),
            pltpu.VMEM((m_local, 1), jnp.float32),
            pltpu.VMEM((2, m_chip, bn), residual.dtype),
            pltpu.VMEM((m_local, half_n), residual.dtype),
            pltpu.VMEM((m_chip, 1), jnp.float32),
            pltpu.VMEM((m_chip, 1), jnp.float32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((ring,)),
            pltpu.SemaphoreType.DMA((ring, n_ici)),
            pltpu.SemaphoreType.DMA((ring, n_ici)),
            pltpu.SemaphoreType.DMA((ring, n_bridge)),
            pltpu.SemaphoreType.DMA((ring, n_bridge)),
        ),
        grid=(route_offset + grid_n + 4,),
    )
    cost_estimate = get_post_moe_epilogue_cost_estimate(
        m=m,
        k=k,
        n=n,
        tp_size=tp_size,
        core_size=core_size,
        x_dtype=x.dtype,
        w_dtype=w.dtype,
        residual_dtype=residual.dtype,
        norm_weight_dtype=norm_weight.dtype,
        residual_gate_dtype=residual_gate.dtype,
        unpooler_bias_dtype=unpooler_bias.dtype,
        bn=bn,
        has_norm_weight=has_norm_weight,
        norm_weight_before=norm_weight_before,
        norm_zero_centered_gamma=norm_zero_centered_gamma,
        has_unpooler_bias=has_unpooler_bias,
        use_residual_gate=use_residual_gate,
    )
    topo_kwargs = {
        "steps": steps,
        "all_masks": ICI_MASKS + (BRIDGE_MASK,),
        "ici_offsets": ici_offsets,
        "antipode_split": (l0, x0, l1, x1),
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
    stats_kwargs = {
        "axis_names": axis_names,
        "strides": strides,
        "sizes": sizes,
        "mesh_axis_names": mesh_axis_names,
        "m_chip": m_chip,
        "half_n": half_n,
        "bn": bn,
        "grid_n": grid_n,
        "double_barrier": double_barrier,
        "has_norm_weight": has_norm_weight,
        "norm_weight_before": norm_weight_before,
        "norm_zero_centered_gamma": norm_zero_centered_gamma,
        "norm_gain_center": norm_gain_center,
        "has_unpooler_bias": has_unpooler_bias,
    }
    route_kwargs = {
        **topo_kwargs,
        "program_offset": route_offset,
        "inputs_preloaded": True,
        "norm_eps": norm_eps,
        "has_norm_weight": has_norm_weight,
        "norm_weight_before": norm_weight_before,
        "norm_zero_centered_gamma": norm_zero_centered_gamma,
        "norm_gain_center": norm_gain_center,
        "has_unpooler_bias": has_unpooler_bias,
        "use_residual_gate": use_residual_gate,
        "residual_temperature": residual_temperature,
        "residual_eps": residual_eps,
    }

    def _call(p2ph, ph2p, xx, ww, rr, nw, rg, ub):
        return pl.pallas_call(
            functools.partial(
                _matmul_tp_intra_topo_post_moe_kernel,
                stats_kwargs=stats_kwargs,
                route_kwargs=route_kwargs,
            ),
            out_shape=jax.ShapeDtypeStruct((m, n), residual.dtype),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=128 * 1024 * 1024,
                disable_bounds_checks=disable_bounds_checks,
            ),
            cost_estimate=cost_estimate,
            metadata=get_post_moe_epilogue_cost_metadata(cost_estimate),
            name=f"unpooler_post_moe_two_pass_topo_bn{bn}",
        )(p2ph, ph2p, xx, ww, rr, nw, rg, ub)

    return jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(None),
                P(None),
                P(axis_names, None),
                P(core_axis, None),
                P(axis_names, None),
                P(),
                P(),
                P(),
            ),
            out_specs=P(None, None),
            check_vma=False,
        )
    )(
        pos2phys,
        phys2pos,
        x,
        w,
        residual,
        norm_weight,
        residual_gate,
        unpooler_bias,
    )
