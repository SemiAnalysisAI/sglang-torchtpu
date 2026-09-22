# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS expert outputs delivered directly to attention-island token owners."""

import torch

_OP = None


def prepare_gptoss_token_owned_moe(layer: torch.nn.Module) -> None:
    """Validate logical ownership and register physical destinations before tracing."""
    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_torchtpu_export_devices,
    )
    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import prepare_fused_moe_rs
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp_group = parallel.tp_group
    attn_group = parallel.attn_tp_group
    if (
        parallel.pp_group.world_size != 1
        or parallel.attn_dcp_size != 1
        or parallel.attn_cp_group.world_size != 1
        or parallel.moe_ep_size != tp_group.world_size
        or parallel.moe_tp_size != 1
        or parallel.moe_dp_group.world_size != 1
    ):
        raise NotImplementedError(
            "GPT-OSS token-owned MoE requires PP1/DCP1/CP1, EP=TP, MoE-TP1/DP1."
        )
    start = (tp_group.rank_in_group // attn_group.world_size) * attn_group.world_size
    expected = list(tp_group.ranks[start : start + parallel.attn_tp_size])
    if list(attn_group.ranks) != expected:
        raise ValueError(
            "Attention islands must be contiguous in logical EP rank order."
        )
    if layer.num_local_experts * parallel.moe_ep_size != 128:
        raise ValueError("GPT-OSS token-owned MoE requires the original 128 experts.")
    # Resolve export devices before the first traced call, alongside physical IDs.
    get_torchtpu_export_devices(tp_group.world_size)
    prepare_fused_moe_rs(layer)
    _build_op()


def _build_op():
    global _OP
    if _OP is not None:
        return _OP
    import jax
    from torch_tpu._internal.pallas import jax_op

    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import _get_mesh
    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import fused_moe_func_rs

    mesh = _get_mesh()

    def fused(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        b1: jax.Array,
        b2: jax.Array,
        weights: jax.Array,
        ids: jax.Array,
        group_offset: jax.Array,
        destinations: jax.Array,
    ) -> jax.Array:
        if x.shape[0] % mesh.shape["model"]:
            raise ValueError(
                "Token ownership requires a full EP-divisible physical batch."
            )
        return fused_moe_func_rs(
            hidden_states=x,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=b1,
            w2_bias=b2,
            gating_output=None,
            topk=4,
            renormalize=False,
            mesh=mesh,
            activation="swigluoai",
            scoring_fn="softmax",
            topk_weights=weights,
            topk_indices=ids,
            group_offset_input=group_offset,
            fp8_direct_write=False,
            tp_rank_to_device_id=destinations,
        )

    op = jax_op("sglang_tpu_pallas::gptoss_token_owned_moe", fused)

    @op.register_fake
    def fake(x, w1, w2, b1, b2, weights, ids, group_offset, destinations):
        return x.new_empty((x.shape[0] // destinations.shape[0], x.shape[1]))

    _OP = op
    return op


def gptoss_token_owned_moe(x, layer, topk_weights, topk_ids, *, return_owner=False):
    """Return this attention island's rows after exact token/slot destination mapping.

    The input is the existing replicated global MoE batch. Expert intermediates
    stay in the fused kernel and each output slot goes to its unique EP owner.
    ``return_owner`` retains the owner shard; otherwise its attention island
    gathers the combined output token shards.
    """
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if x.dtype != torch.bfloat16 or layer.w13_weight.dtype != torch.bfloat16:
        raise TypeError(
            "GPT-OSS token-owned MoE requires BF16 activations and weights."
        )
    if layer.w2_weight.dtype != torch.bfloat16 or topk_ids.shape[-1] != 4:
        raise TypeError("GPT-OSS token-owned MoE requires BF16 weights and top-k 4.")
    ids = topk_ids.to(torch.int32)
    valid = (ids >= 0) & (ids < 128)
    ids = torch.where(valid, ids, 0)
    weights = topk_weights.to(x.dtype) * valid.to(x.dtype)
    shard = _build_op()(
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_bias,
        layer.w2_weight_bias,
        weights,
        ids,
        layer._tpu_fused_rs_group_offset,
        layer._tpu_fused_rs_tp_rank_to_device_id,
    )
    if return_owner:
        return shard
    return parallel.attn_tp_group.all_gather(shard, dim=0)
