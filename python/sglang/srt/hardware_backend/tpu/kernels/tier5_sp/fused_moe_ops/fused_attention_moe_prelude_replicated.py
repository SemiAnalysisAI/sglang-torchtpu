# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Replicated attention-to-MoE prelude for the SP-off fallback path.

SP-off presents the full ``[T, H]`` activation on every model shard.  This
kernel keeps that contract while combining the post-attention RMSNorm,
norm-preserving residual add, pre-FFN RMSNorm, router, two-way on-chip pooler,
and pre-expert RMSNorm in one collective Pallas call.

The pooler weight is split over the physical on-chip core pair only.  Each core
computes its output-feature half and sends that half directly into its sibling's
replicated output buffer.  No inter-chip token exchange is needed because every
chip already owns all tokens.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_attention_moe_prelude import (
    _prelude_values,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_router import (
    route_hidden_block,
    router_block_tokens,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_intra_tp_all_gather import (
    util,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_intra_tp_all_gather._matmul_tp_intra_topo import (
    _physical_topo_maps,
)


def _replicated_prelude_router_pooler_kernel(
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
    prepared_vmem,
    routing_weights_vmem,
    routing_indices_vmem,
    pooler_vmem,
    norm_tile_vmem,
    self_sems,
    send_sems,
    recv_sems,
    attn_copy_sem,
    stream_copy_sem,
    h_copy_sem,
    routing_weights_copy_sem,
    routing_indices_copy_sem,
    norm_copy_sem,
    *,
    prelude_kwargs,
    router_kwargs,
    axis_names: tuple[str, ...],
    strides: tuple[int, ...],
    sizes: tuple[int, ...],
    mesh_axis_names: tuple[str, ...],
    block_tokens: int,
    norm_block_tokens: int,
    half_n: int,
    bn: int,
    grid_n: int,
    norm_eps: float,
    has_pre_expert_weight: bool,
    norm_zero_centered_gamma: bool,
    norm_gain_center: float,
    double_barrier: bool,
):
    i = pl.program_id(0)
    tokens = attn_ref.shape[0]
    topk = router_kwargs["topk"]
    num_token_blocks = tokens // block_tokens
    my_rank = util.combined_axis_index(axis_names, strides)
    core = lax.rem(my_rank, jnp.int32(2))
    sibling = lax.bitwise_xor(my_rank, jnp.int32(1))
    peer = util.make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names)

    def _self_desc(slot, tile):
        col0 = core * jnp.int32(half_n) + tile * bn
        return pltpu.make_async_copy(
            src_ref=pooler_vmem.at[slot],
            dst_ref=pooled_hbm_ref.at[:, pl.ds(col0, bn)],
            sem=self_sems.at[slot],
        )

    def _send_desc(slot, tile):
        col0 = core * jnp.int32(half_n) + tile * bn
        return pltpu.make_async_remote_copy(
            src_ref=pooler_vmem.at[slot],
            dst_ref=pooled_hbm_ref.at[:, pl.ds(col0, bn)],
            send_sem=send_sems.at[slot],
            recv_sem=recv_sems.at[slot],
            device_id=peer(sibling),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _recv_desc(slot, tile):
        sibling_col0 = (
            lax.bitwise_xor(core, jnp.int32(1)) * jnp.int32(half_n) + tile * bn
        )
        block = pooled_hbm_ref.at[:, pl.ds(sibling_col0, bn)]
        return pltpu.make_async_remote_copy(
            src_ref=block,
            dst_ref=block,
            send_sem=send_sems.at[slot],
            recv_sem=recv_sems.at[slot],
            device_id=peer(sibling),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    @pl.when(i == 0)
    def _prepare_and_route():
        util.hypercube_local_barrier([peer(sibling)], double_barrier=double_barrier)
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
            stream_tile_vmem[...] = h
            h_copy = pltpu.make_async_copy(
                src_ref=stream_tile_vmem,
                dst_ref=h_hbm_ref.at[rows, :],
                sem=h_copy_sem,
            )
            h_copy.start()
            h_copy.wait()

            routing_weights, routing_indices = route_hidden_block(
                prepared,
                router_weight_ref[...],
                expert_bias_ref[...],
                **router_kwargs,
            )
            routing_weights_vmem[rows, :] = routing_weights[:, :topk]
            routing_indices_vmem[rows, :] = routing_indices[:, :topk].astype(jnp.uint8)

        routing_weights_copy = pltpu.make_async_copy(
            src_ref=routing_weights_vmem,
            dst_ref=routing_weights_hbm_ref,
            sem=routing_weights_copy_sem,
        )
        routing_indices_copy = pltpu.make_async_copy(
            src_ref=routing_indices_vmem,
            dst_ref=routing_indices_hbm_ref,
            sem=routing_indices_copy_sem,
        )
        routing_weights_copy.start()
        routing_indices_copy.start()

    @pl.when(jnp.logical_and(i >= 2, i < grid_n + 2))
    def _wait_previous_pooler_tile():
        tile = i - 2
        slot = lax.rem(tile, jnp.int32(2))
        _self_desc(slot, tile).wait()
        _send_desc(slot, tile).wait_send()
        _recv_desc(slot, tile).wait_recv()

    @pl.when(i < grid_n)
    def _compute_pooler_tile():
        slot = lax.rem(i, jnp.int32(2))
        pooler_vmem.at[slot][...] = util.reorder_matmul(
            prepared_vmem[...],
            pooler_weight_ref[...],
            w_transposed=True,
            preferred_element_type=jnp.float32,
            out_dtype=pooled_hbm_ref.dtype,
        )
        _self_desc(slot, i).start()
        _send_desc(slot, i).start()

    @pl.when(i == 1)
    def _drain_routing_outputs():
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

    pooler_grid_end = grid_n + 2
    norm_program = i - pooler_grid_end
    norm_grid = tokens // norm_block_tokens

    @pl.when(jnp.logical_and(i >= pooler_grid_end, norm_program < norm_grid))
    def _pre_expert_norm():
        rows = pl.ds(norm_program * norm_block_tokens, norm_block_tokens)
        pooled_hidden = pooled_hbm_ref.shape[1]
        norm_copy = pltpu.make_async_copy(
            src_ref=pooled_hbm_ref.at[rows, :],
            dst_ref=norm_tile_vmem,
            sem=norm_copy_sem,
        )
        norm_copy.start()
        norm_copy.wait()

        pooled_f32 = norm_tile_vmem[...].astype(jnp.float32)
        variance = jnp.mean(pooled_f32 * pooled_f32, axis=1, keepdims=True)
        normalized = pooled_f32 * jax.lax.rsqrt(variance + norm_eps)
        if has_pre_expert_weight:
            weight = pre_expert_weight_ref[...].astype(jnp.float32)
            if norm_zero_centered_gamma:
                weight = weight + norm_gain_center
            normalized = normalized * weight

        @functools.partial(
            pl.run_scoped,
            prepared_expert_vmem=pltpu.VMEM(
                (norm_block_tokens, pooled_hidden // 128, 128),
                pooled_hbm_ref.dtype,
            ),
        )
        def _store_prepared(prepared_expert_vmem):
            prepared_expert_vmem[...] = normalized.astype(pooled_hbm_ref.dtype).reshape(
                norm_block_tokens, pooled_hidden // 128, 128
            )
            prepared_copy = pltpu.make_async_copy(
                src_ref=prepared_expert_vmem,
                dst_ref=prepared_expert_hbm_ref.at[rows, :, :],
                sem=norm_copy_sem,
            )
            prepared_copy.start()
            prepared_copy.wait()


def fused_attention_moe_prelude_router_pooler_replicated(
    attn_out,
    residual_stream,
    post_weight,
    gate,
    pre_weight,
    router_weight,
    expert_bias,
    pooler_weight,
    pre_expert_weight,
    mesh,
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
    bn: int = 512,
    collective_id: int = 0,
    double_barrier: bool = True,
    disable_bounds_checks: bool = True,
):
    """Return replicated residual, pooler, expert-input, and routing outputs."""
    if attn_out.shape != residual_stream.shape or attn_out.ndim != 2:
        raise ValueError("attn_out and residual_stream must be equal rank-2 shapes")
    tokens, hidden = attn_out.shape
    if post_weight.shape != (hidden,) or gate.shape != (hidden,):
        raise ValueError("post_weight and gate must match the hidden dimension")
    if pre_weight.shape != (hidden,):
        raise ValueError("pre_weight must match the hidden dimension")
    num_experts, router_hidden = router_weight.shape
    if router_hidden != hidden or expert_bias.shape != (num_experts,):
        raise ValueError("router inputs do not match the activation shape")
    pooled_hidden, pooler_hidden = pooler_weight.shape
    if pooler_hidden != hidden:
        raise ValueError("pooler_weight must have shape [pooled_hidden, hidden]")
    if pre_expert_weight.shape != (pooled_hidden,):
        raise ValueError("pre_expert_weight must match the pooler output width")
    if hidden % 256 or num_experts % 128 or pooled_hidden % 256:
        raise ValueError("hidden, experts, and pooled hidden must be TPU aligned")
    if not 1 <= topk <= min(8, num_experts):
        raise ValueError(f"topk must be in [1, {min(8, num_experts)}]")
    if score_normalization not in ("sigmoid", "softmax"):
        raise ValueError(f"unsupported score normalization {score_normalization!r}")

    axis_names = (chip_axis, core_axis)
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    if tp_size != 16 or int(mesh.shape[core_axis]) != 2:
        raise ValueError("replicated fused prelude requires a TP16 on-chip core mesh")
    pos_to_phys, _ = _physical_topo_maps(mesh, axis_names)
    if any((pos_to_phys[pos] & 1) != (pos & 1) for pos in range(tp_size)):
        raise ValueError("core_axis must be the physical on-chip core pair")

    block_tokens = (
        min(tokens, 256)
        if tokens % min(tokens, 256) == 0
        else router_block_tokens(tokens)
    )
    norm_block_tokens = min(tokens, 512)
    if tokens % block_tokens or tokens % norm_block_tokens:
        raise ValueError(
            f"tokens={tokens} must divide router={block_tokens} and norm={norm_block_tokens} tiles"
        )
    half_n = pooled_hidden // 2
    if bn <= 0 or half_n % bn:
        raise ValueError(f"half pooler width {half_n} must be divisible by bn={bn}")
    grid_n = half_n // bn
    norm_grid = tokens // norm_block_tokens
    mesh_axis_names = tuple(mesh.axis_names)

    hbm_input = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    hbm_output = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    resident_matrix = lambda shape: pl.BlockSpec(shape, lambda i: (0, 0))
    resident_vector = lambda shape: pl.BlockSpec(shape, lambda i: (0,))
    pooler_weight_spec = pl.BlockSpec(
        (bn, hidden), lambda i: (jnp.minimum(i, grid_n - 1), 0)
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=(
            hbm_input,
            hbm_input,
            resident_vector((hidden,)),
            resident_vector((hidden,)),
            resident_vector((hidden,)),
            resident_matrix((num_experts, hidden)),
            resident_vector((num_experts,)),
            pooler_weight_spec,
            resident_vector((pooled_hidden,)),
        ),
        out_specs=(hbm_output,) * 5,
        scratch_shapes=(
            pltpu.VMEM((block_tokens, hidden), attn_out.dtype),
            pltpu.VMEM((block_tokens, hidden), residual_stream.dtype),
            pltpu.VMEM((tokens, hidden), residual_stream.dtype),
            pltpu.VMEM((tokens, topk), jnp.float32),
            pltpu.VMEM((tokens, topk), jnp.uint8),
            pltpu.VMEM((2, tokens, bn), residual_stream.dtype),
            pltpu.VMEM((norm_block_tokens, pooled_hidden), residual_stream.dtype),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
        grid=(grid_n + 2 + norm_grid,),
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

    def _call(aa, ss, pw, gg, prew, rw, eb, poolw, normw):
        return pl.pallas_call(
            functools.partial(
                _replicated_prelude_router_pooler_kernel,
                prelude_kwargs=prelude_kwargs,
                router_kwargs=router_kwargs,
                axis_names=axis_names,
                strides=strides,
                sizes=sizes,
                mesh_axis_names=mesh_axis_names,
                block_tokens=block_tokens,
                norm_block_tokens=norm_block_tokens,
                half_n=half_n,
                bn=bn,
                grid_n=grid_n,
                norm_eps=norm_eps,
                has_pre_expert_weight=has_pre_expert_weight,
                norm_zero_centered_gamma=norm_zero_centered_gamma,
                norm_gain_center=norm_gain_center,
                double_barrier=double_barrier,
            ),
            out_shape=(
                jax.ShapeDtypeStruct((tokens, hidden), residual_stream.dtype),
                jax.ShapeDtypeStruct((tokens, pooled_hidden), residual_stream.dtype),
                jax.ShapeDtypeStruct(
                    (tokens, pooled_hidden // 128, 128), residual_stream.dtype
                ),
                jax.ShapeDtypeStruct((tokens, topk), jnp.float32),
                jax.ShapeDtypeStruct((tokens, topk), jnp.uint8),
            ),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=64 * 1024 * 1024,
                disable_bounds_checks=disable_bounds_checks,
            ),
            name=f"fused_attention_router_pooler_replicated_bn{bn}",
        )(aa, ss, pw, gg, prew, rw, eb, poolw, normw)

    return jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(P(), P(), P(), P(), P(), P(), P(), P(core_axis, None), P()),
            out_specs=(P(), P(), P(), P(), P()),
            check_vma=False,
        )
    )(
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


__all__ = ["fused_attention_moe_prelude_router_pooler_replicated"]
