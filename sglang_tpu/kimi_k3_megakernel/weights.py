# SPDX-License-Identifier: Apache-2.0
"""Per-rank Kimi-K3 megakernel weights as torch tensors on the TPU.

The megakernel takes one weight dictionary per rank (the vendored loader's
``selected_local_shapes``), filled from two sources:

* Dense families (attention projections, norms, router, latent MoE front and
  back, shared experts, embedding, lm_head) are read from the checkpoint with
  the vendored reader's own per-rank slicing, the layout the kernel was
  validated with. SGLang keeps its own copies for prefill (about 5 GiB per rank
  held twice; sharing them is a follow-up).
* The routed experts are SGLang's processed MXFP4 tensors, restacked once into
  ``[1, layers, experts, ...]`` and shared with SGLang's prefill MoE, which
  reads one layer of them through ``fused_moe_call(w_layer_index=...)``. Only
  the E8M0 scales are rebuilt, because SGLang keeps them as f32.

Every tensor handed to the kernel is a whole base tensor: torch_tpu
re-materializes a view chain on every compiled call.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import Parameter

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.fp8_moe import _free_replaced_storage

logger = logging.getLogger(__name__)

EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
_TORCH_DTYPES = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int32): torch.int32,
}


def base_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` as a contiguous base tensor (``_base is None``, offset 0)."""
    tensor = tensor.contiguous()
    if tensor._base is None and tensor.storage_offset() == 0:
        return tensor
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    out.copy_(tensor)
    return out


def _host_array_to_torch(value: np.ndarray) -> torch.Tensor:
    """Host numpy array (including ml_dtypes bfloat16) -> torch, same bits."""
    array = np.ascontiguousarray(value)
    if array.dtype.name == "bfloat16":
        out = torch.empty(array.shape, dtype=torch.bfloat16)
        out.view(torch.uint16).copy_(torch.from_numpy(array.view(np.uint16)))
        return out
    if array.dtype not in _TORCH_DTYPES:
        raise TypeError(f"unsupported megakernel weight dtype {array.dtype}")
    return torch.from_numpy(array)


def load_dense_families(
    checkpoint_path: str, *, rank: int, layers: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """The dense families for ``rank``, each ``[1, ...]`` on ``device``."""
    from ._vendor.kimi.load import Checkpoint, _family_layers, selected_local_shapes

    checkpoint = Checkpoint(checkpoint_path)
    shapes = selected_local_shapes(
        layers, checkpoint.config.dim, vocab=checkpoint.config.vocab
    )
    out = {}
    for name, shape in shapes.items():
        if name in EXPERT_FAMILIES:
            continue
        layer_ids = _family_layers(checkpoint.config, name, layers)
        if layer_ids is None:
            value = checkpoint.dense_selected(name, 0, rank)
        else:
            with ThreadPoolExecutor(checkpoint.READ_THREADS) as pool:
                value = np.stack(
                    list(
                        pool.map(
                            lambda layer: checkpoint.dense_selected(name, layer, rank),
                            layer_ids,
                        )
                    )
                )
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"{name}: checkpoint gives {value.shape}, the kernel expects {shape}"
            )
        out[name] = base_tensor(_host_array_to_torch(value[None]).to(device))
    return out


def e8m0_from_f32_scales(scales: torch.Tensor) -> torch.Tensor:
    """f32 MXFP4 block scales -> E8M0 bytes, exactly.

    The inverse of ``mxfp4_tpu_util._e8m0_to_fp32``: a code is the f32
    exponent field of ``2**(code - 127)``, and code 0 is the subnormal
    ``2**-127``. Anything else raises, so a rounded scale never reaches the
    kernel.
    """
    bits = scales.contiguous().view(torch.int32)
    codes = (bits >> 23) & 0xFF
    power_of_two = ((bits & 0x807FFFFF) == 0) & (codes != 0)
    subnormal = bits == (1 << 22)
    if not bool((power_of_two | subnormal).all()):
        raise ValueError("MXFP4 scales must be positive powers of two")
    return codes.to(torch.uint8)


def _check_moe_layer(moe: FusedMoE) -> None:
    method = moe.quant_method
    if not isinstance(method, Mxfp4MoEMethod):
        raise TypeError(
            "the Kimi-K3 megakernel needs MXFP4 routed experts, got "
            f"{type(method).__name__}"
        )
    config = method.moe_runner_config
    if (
        moe.w13_weight.dtype != torch.uint8
        or moe.w13_weight.ndim != 3
        or moe.w13_weight_scale.dtype != torch.float32
        or moe.w13_weight_bias is not None
        or config.no_combine
        or config.apply_router_weight_on_input
        or config.num_fused_shared_experts
    ):
        raise ValueError(
            "the Kimi-K3 megakernel needs SGLang's packed MXFP4 experts "
            "(uint8 [E, K/2, N], f32 scales, no bias, combined output)"
        )


def _move_to_host(param: Parameter, destination: torch.Tensor) -> None:
    """Copy ``param`` into host ``destination``, then free its device storage."""
    old = param.data
    destination.copy_(old)
    param.data = torch.empty(0, dtype=old.dtype, device=old.device)
    _free_replaced_storage(old, param.data)


def _stage_layer(moe: FusedMoE, staging: dict[str, torch.Tensor], index: int) -> None:
    """Copy one layer's experts to host staging, freeing their device copies."""
    _move_to_host(moe.w13_weight, staging["expert_gu"][0, index])
    _move_to_host(moe.w2_weight, staging["expert_down"][0, index])
    # The f32 scales stay on device for prefill; the kernel reads E8M0 bytes.
    staging["expert_gus"][0, index].copy_(
        e8m0_from_f32_scales(moe.w13_weight_scale.cpu()).squeeze(-2)
    )
    staging["expert_ds"][0, index].copy_(
        e8m0_from_f32_scales(moe.w2_weight_scale.cpu()).squeeze(-2)
    )


def stack_routed_experts(
    moe_layers: list[FusedMoE], device: torch.device
) -> dict[str, torch.Tensor]:
    """Restack every MoE layer's experts into shared ``[1, L, E, ...]`` tensors.

    Each layer's packed weights pass through host memory (40 GiB per rank for
    Kimi-K3) so the device never holds the per-layer and the stacked copy at
    once. Afterwards every layer's ``w13_weight`` / ``w2_weight`` is the stacked
    tensor and its quant method reads its own layer through ``w_layer_index``.
    """
    for moe in moe_layers:
        _check_moe_layer(moe)
    first = moe_layers[0]
    layers = len(moe_layers)
    w13, w2 = first.w13_weight, first.w2_weight
    w13_scale, w2_scale = first.w13_weight_scale, first.w2_weight_scale
    staging = {
        "expert_gu": torch.empty((1, layers, *w13.shape), dtype=torch.uint8),
        "expert_down": torch.empty((1, layers, *w2.shape), dtype=torch.uint8),
        "expert_gus": torch.empty(
            (1, layers, w13_scale.shape[0], w13_scale.shape[1], w13_scale.shape[3]),
            dtype=torch.uint8,
        ),
        "expert_ds": torch.empty(
            (1, layers, w2_scale.shape[0], w2_scale.shape[1], w2_scale.shape[3]),
            dtype=torch.uint8,
        ),
    }
    for index, moe in enumerate(moe_layers):
        _stage_layer(moe, staging, index)
    experts = {}
    for family in EXPERT_FAMILIES:
        experts[family] = base_tensor(staging.pop(family).to(device))
    shared_w13 = Parameter(experts["expert_gu"], requires_grad=False)
    shared_w2 = Parameter(experts["expert_down"], requires_grad=False)
    for index, moe in enumerate(moe_layers):
        moe.w13_weight = shared_w13
        moe.w2_weight = shared_w2
        moe.quant_method = StackedMxfp4MoEMethod(
            moe_runner_config=moe.quant_method.moe_runner_config,
            layer_index=torch.tensor([index], dtype=torch.int32, device=device),
        )
    return experts


class StackedMxfp4MoEMethod(FusedMoEMethodBase):
    """The TPU MXFP4 MoE forward over layer-stacked experts.

    Mirrors ``UnquantizedFusedMoEMethod.forward_tpu`` (which
    ``Mxfp4MoEMethod.apply`` runs on TPU) for Kimi-K3's case, adding the layer
    index; ``stack_routed_experts`` checked that case at attach.
    """

    def __init__(
        self, *, moe_runner_config: MoeRunnerConfig, layer_index: torch.Tensor
    ):
        self.moe_runner_config = moe_runner_config
        # int32 [1] on device: one compiled MoE kernel serves every layer.
        self.layer_index = layer_index

    def apply(
        self, layer: FusedMoE, dispatch_output: StandardDispatchOutput
    ) -> StandardCombineInput:
        from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

        config = self.moe_runner_config
        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_ids = topk_ids.to(torch.int32)
        parallel = get_parallel()
        output = fused_moe_call(
            hidden_states=hidden_states,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights.to(hidden_states.dtype),
            topk_ids=topk_ids,
            topk=int(topk_ids.shape[-1]),
            activation=config.activation,
            global_num_experts=layer.num_local_experts * parallel.moe_ep_size,
            expert_group_offset=layer.num_local_experts * parallel.moe_ep_rank,
            gemm1_alpha=config.gemm1_alpha,
            gemm1_clamp_limit=config.gemm1_clamp_limit,
            w_layer_index=self.layer_index,
        )
        return StandardCombineInput(hidden_states=output)
