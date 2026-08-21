# SPDX-License-Identifier: Apache-2.0
"""PyTorch custom op wrapper for the native Pallas fused MoE + reduce-scatter kernel."""

import functools
import math

import torch

_FUSED_MOE_RS_OP_NAME = "sglang_tpu_pallas::fused_moe_rs"
_FUSED_MOE_RS_OP = None
_FUSED_MOE_RS_ROUTE = None
_FUSED_RS_ROUTED_ROW_ALIGNMENT = 16


def fused_moe_rs_padded_tokens(num_tokens: int, topk: int, ep_size: int) -> int:
    """Pad tokens for both EP sharding and the fused kernel's routed rows."""
    routed_row_token_alignment = _FUSED_RS_ROUTED_ROW_ALIGNMENT // math.gcd(
        _FUSED_RS_ROUTED_ROW_ALIGNMENT, topk
    )
    token_alignment = math.lcm(ep_size, routed_row_token_alignment)
    return ((num_tokens + token_alignment - 1) // token_alignment) * token_alignment


def prepare_fused_moe_rs(layer: torch.nn.Module) -> None:
    """Prepare physical-rank metadata before Dynamo captures the MoE layer."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_tpu_group_rank_to_device_id,
    )
    from sglang.srt.runtime_context import get_parallel

    tp_group = get_tp_group()
    rank_to_device_id = get_tpu_group_rank_to_device_id(
        tuple(tp_group.ranks),
        tp_group.cpu_group,
    )
    group_offset = torch.tensor(
        [
            get_parallel().moe_ep_rank * layer.num_local_experts,
            0,
        ],
        dtype=torch.int32,
        device=layer.w13_weight.device,
    )
    tp_rank_to_device_id_tensor = torch.tensor(
        rank_to_device_id,
        dtype=torch.int32,
        device=layer.w13_weight.device,
    )
    layer.register_buffer("_tpu_fused_rs_group_offset", group_offset, persistent=False)
    layer.register_buffer(
        "_tpu_fused_rs_tp_rank_to_device_id",
        tp_rank_to_device_id_tensor,
        persistent=False,
    )
    _build_fused_moe_rs_op(rank_to_device_id)


@functools.cache
def _get_mesh(rank_to_device_id: tuple[int, ...]):
    import numpy as np
    from jax.sharding import Mesh

    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_torchtpu_export_devices,
    )

    export_devices = np.asarray(
        get_torchtpu_export_devices(rank_to_device_id)
    ).reshape(1, -1)
    return Mesh(export_devices, ("data", "model"))


def _build_fused_moe_rs_op(
    rank_to_device_id: tuple[int, ...] | None = None,
):
    global _FUSED_MOE_RS_OP, _FUSED_MOE_RS_ROUTE
    if _FUSED_MOE_RS_OP is not None:
        if (
            rank_to_device_id is not None
            and rank_to_device_id != _FUSED_MOE_RS_ROUTE
        ):
            raise RuntimeError(
                "The fused MoE reduce-scatter op was already built for route "
                f"{_FUSED_MOE_RS_ROUTE}, got {rank_to_device_id}"
            )
        return _FUSED_MOE_RS_OP
    if rank_to_device_id is None:
        raise RuntimeError(
            "prepare_fused_moe_rs must initialize TPU topology before the fused "
            "MoE reduce-scatter op is used"
        )

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from torch_tpu._internal.pallas import jax_op
    from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import _MOE_GMM_LOCAL_RS

    mesh = _get_mesh(rank_to_device_id)

    def _wrapped(
        hidden_states: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        post_expert_norm_weight: jax.Array,
        group_offset: jax.Array,
        topk: int,
        activation: str,
        has_post_expert_norm: bool,
        fp8_direct_write: bool,
        tp_rank_to_device_id: jax.Array | None = None,
    ) -> jax.Array:
        with tpu_info_for_cpu_export():
            num_tokens = hidden_states.shape[0]
            ep_size = mesh.shape["model"]
            padded_tokens = fused_moe_rs_padded_tokens(num_tokens, topk, ep_size)
            if padded_tokens != num_tokens:
                token_padding = padded_tokens - num_tokens
                hidden_states = jnp.pad(hidden_states, ((0, token_padding), (0, 0)))
                topk_weights = jnp.pad(topk_weights, ((0, token_padding), (0, 0)))
                topk_ids = jnp.pad(topk_ids, ((0, token_padding), (0, 0)))

            _has_pn = has_post_expert_norm and post_expert_norm_weight is not None
            _pnw = (
                post_expert_norm_weight
                if _has_pn
                else jnp.zeros((1,), dtype=hidden_states.dtype)
            )
            if tp_rank_to_device_id is None:
                raise ValueError("TPU rank-to-mesh routing metadata is required")

            # shard_map supplies the multi-device context required by Pallas
            # ICI DMAs. Inputs and outputs are local, so all specs are P().
            result = jax.shard_map(
                functools.partial(
                    _MOE_GMM_LOCAL_RS,
                    activation=activation,
                    topk=topk,
                    ep_size=ep_size,
                    ep_axis_name="model",
                    has_post_norm=_has_pn,
                    sp_enabled=False,
                    fp8_direct_write=fp8_direct_write,
                ),
                mesh=mesh,
                in_specs=(
                    P(),  # hidden_states_local
                    P(),  # w1
                    P(),  # w1_scale
                    None,  # w1_bias
                    P(),  # w2
                    P(),  # w2_scale
                    None,  # w2_bias
                    None,  # w1_global_scale
                    None,  # w2_global_scale
                    P(),  # group_offset
                    P(),  # topk_weights
                    P(),  # topk_indices
                    P(),  # post_expert_norm_weight_input
                    P(),  # tp_rank_to_device_id
                ),
                out_specs=P(),
                check_vma=False,
            )(
                hidden_states,
                w1,
                w1_scale,
                None,
                w2,
                w2_scale,
                None,
                None,
                None,
                group_offset,
                topk_weights,
                topk_ids,
                _pnw,
                tp_rank_to_device_id,
            )
            if post_expert_norm_weight is not None and not has_post_expert_norm:
                result = result + post_expert_norm_weight[0].astype(result.dtype) * 0
            return result

    # Bridge the JAX kernel into Torch as clean non-sharded custom op.
    custom_op = jax_op(_FUSED_MOE_RS_OP_NAME, _wrapped)

    def _fake_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm_weight: torch.Tensor,
        group_offset: torch.Tensor,
        topk: int = 1,
        activation: str = "silu",
        has_post_expert_norm: bool = False,
        fp8_direct_write: bool = False,
        tp_rank_to_device_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ep_size = (
            int(tp_rank_to_device_id.shape[0])
            if tp_rank_to_device_id is not None
            else int(mesh.shape["model"])
        )
        padded_tokens = fused_moe_rs_padded_tokens(
            hidden_states.shape[0], topk, ep_size
        )
        shard_size = padded_tokens // ep_size
        return hidden_states.new_empty((shard_size, hidden_states.shape[1]))

    custom_op.register_fake(_fake_impl)
    _FUSED_MOE_RS_OP = custom_op
    _FUSED_MOE_RS_ROUTE = rank_to_device_id
    return _FUSED_MOE_RS_OP


def fused_moe_rs_call(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    group_offset: torch.Tensor,
    topk: int,
    activation: str,
    has_post_expert_norm: bool,
    fp8_direct_write: bool,
    tp_rank_to_device_id: torch.Tensor | None = None,
) -> torch.Tensor:
    if tp_rank_to_device_id is None:
        raise ValueError("TPU rank-to-mesh routing metadata is required")
    return _build_fused_moe_rs_op()(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        post_expert_norm_weight,
        group_offset,
        topk,
        activation,
        has_post_expert_norm,
        fp8_direct_write,
        tp_rank_to_device_id,
    )


def fused_moe_rs_gather(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    group_offset: torch.Tensor,
    topk: int,
    activation: str,
    has_post_expert_norm: bool,
    fp8_direct_write: bool,
    tp_rank_to_device_id: torch.Tensor | None = None,
) -> torch.Tensor:
    from sglang.srt.distributed import get_tp_group

    token_shard = fused_moe_rs_call(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        post_expert_norm_weight=post_expert_norm_weight,
        group_offset=group_offset,
        topk=topk,
        activation=activation,
        has_post_expert_norm=has_post_expert_norm,
        fp8_direct_write=fp8_direct_write,
        tp_rank_to_device_id=tp_rank_to_device_id,
    )
    output = get_tp_group().all_gather(token_shard, dim=0)
    return output
