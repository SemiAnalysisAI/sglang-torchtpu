# SPDX-License-Identifier: Apache-2.0
"""TPU NVFP4 (``modelopt_fp4``) MoE helpers.

Serves NVFP4 MoE weights via the packed-FP4 path of the Pallas ``gmm_v2`` kernel.
NVFP4's two-level scale (fp8 per-16-block + fp32 per-tensor global) is folded at
load into gmm_v2's single per-block scale (``real = fp4 * fp8_block * global``).
No native fp4 MXU on TPU. By default gmm_v2 folds that block scale into the weights
before the matmul, so they unpack fp4->bf16 and multiply against bf16 activations
(W4-stored / A16-compute); the static ``input_scale`` is unused. (Setting
``SGLANG_TPU_NVFP4_PRESCALE=0`` falls back to the legacy path: unpack fp4->fp8,
quantize activations to fp8, apply the block scale per 16 after the matmul.)
"""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.tpu.fp8_moe import _clear_replaced_param
from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.srt.layers.quantization.mxfp4_tpu_util import (
    _pad_gate_up_halves_axis1,
    _pad_last_axis,
)
from sglang.srt.layers.utils import copy_or_rebind_param

# NVFP4 always uses a group/block size of 16 along the contracting dim.
NVFP4_GROUP_SIZE = 16
PACKING = 2  # two FP4 codes per uint8 byte

# NVFP4 create_weights params the TPU gmm_v2 path doesn't consume (globals folded
# into the block scale; input scales unused; *_swizzled / *_mma are CUDA-only).
_UNUSED_NVFP4_MOE_PARAMS = (
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
    "w13_blockscale_mma",
    "w2_blockscale_mma",
    "g1_alphas",
    "g2_alphas",
    "w13_input_scale_quant",
    "w2_input_scale_quant",
)


def _fold_w13_global_scale(
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    is_gated: bool,
) -> torch.Tensor:
    """Fold NVFP4's per-tensor global into the fp8 per-block scale (-> fp32).

    Gate and up carry different globals, so W13 folds per half. ``block_scale`` is
    ``(E, 2I, H/16)`` fp8-e4m3, rows ``[Gate(I), Up(I)]``; ``global_scale`` is
    ``(E, 2)`` (gate, up) or ``(E,)`` if shared.
    """
    s = block_scale.float()
    g = global_scale.float()
    # (E, 2) folds per half; (E,) or (E, 1) shared global folds per expert.
    if is_gated and g.dim() == 2 and g.shape[1] == 2:
        half = s.shape[1] // 2
        s[:, :half, :] *= g[:, 0:1, None]
        s[:, half:, :] *= g[:, 1:2, None]
    else:
        g_flat = g if g.dim() == 1 else g[:, 0]
        s *= g_flat[:, None, None]
    return s


def _to_gmm_v2_scale_layout(
    scale: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``(E, OUT, num_blocks_K)`` -> gmm_v2 ``(E, num_blocks_K, 1, OUT)``.

    The stored scale ``dtype`` defaults to fp32 (the folded ``fp8_block *
    global`` product held at full precision). Passing ``torch.float8_e4m3fn``
    (or ``torch.bfloat16``) re-quantizes the folded scale to a narrower type,
    cutting the per-block scale stream 4x (fp8) / 2x (bf16) — gmm_v2 sizes its
    scale traffic from the tensor's own dtype, so no kernel change is needed
    beyond honoring ``scale.dtype``.
    """
    return scale.permute(0, 2, 1).contiguous().unsqueeze(2).to(dtype)


def prepare_tpu_nvfp4_moe_weights(
    layer: torch.nn.Module,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
    align_to: int = 128,
    scale_dtype: torch.dtype = torch.float32,
) -> None:
    """Format NVFP4 MoE weights/scales for the TPU gmm_v2 backend.

    Keeps weights packed (uint8), folds the two-level scale to one per-block
    scale stored as ``scale_dtype``, transposes to gmm_v2 layout, and pads the
    intermediate to ``align_to``. (Serving passes ``scale_dtype=bf16``; the fp32
    default here keeps the fold exact for the unit tests.)

    ``group_size`` is the per-block scale granularity along the contracting dim.
    Only native NVFP4 block-16 is supported: the kernel reaches full MXU depth by
    pre-scaling the weights before the matmul (``gmm_v2`` ``rhs_prescale``), so
    coarsening the block (e.g. 512) is unnecessary and would only lose accuracy.

    Layout in (checkpoint) -> out (gmm_v2), with ``G = group_size = 16``:
      * ``w13_weight``       (E, 2I, H/2) u8  -> (E, H/2, 2*I_pad) u8
      * ``w13_weight_scale`` (E, 2I, H/G) f8  -> (E, H/G, 1, 2*I_pad) scale_dtype
      * ``w2_weight``        (E, H, I/2) u8   -> (E, I_pad/2, H) u8
      * ``w2_weight_scale``  (E, H, I/G) f8   -> (E, I_pad/G, 1, H) scale_dtype
    """
    if group_size != NVFP4_GROUP_SIZE:
        # Native NVFP4 is block-16; the pre-scale kernel path fills the MXU
        # without coarsening, so a wider block would only degrade numerics.
        raise NotImplementedError(
            f"TPU NVFP4 MoE only supports group_size={NVFP4_GROUP_SIZE}, "
            f"got {group_size}."
        )

    if not layer.moe_runner_config.is_gated:
        # GMM1 always splits its output in half for the gated SwiGLU, so a
        # non-gated expert would be mis-activated. Reject rather than miscompute.
        raise NotImplementedError("TPU NVFP4 MoE only supports gated (SwiGLU) experts.")

    w13 = layer.w13_weight.data  # (E, 2I, H/2) uint8
    w2 = layer.w2_weight.data  # (E, H, I/2) uint8
    _E, two_I, _hidden_packed = w13.shape
    intermediate = two_I // 2

    # Fold the per-tensor global scale into the fp8 block scale (computed in fp32;
    # re-quantized to scale_dtype in the layout step below).
    w13_s = _fold_w13_global_scale(
        layer.w13_weight_scale.data, layer.w13_weight_scale_2.data, is_gated=True
    )  # (E, 2I, H/G) f32
    w2_s = (
        layer.w2_weight_scale.data.float()
        * layer.w2_weight_scale_2.data.float()[:, None, None]
    )  # (E, H, I/G) f32

    # Pad intermediate to align_to; pad_n must stay a multiple of group_size and
    # PACKING to keep scale blocks and packed bytes whole (enforced below).
    pad_n = (align_to - intermediate % align_to) % align_to
    if pad_n % group_size != 0 or pad_n % PACKING != 0:
        raise ValueError(
            f"align_to={align_to} incompatible with group_size={group_size}: "
            f"pad_n={pad_n} must be a multiple of both {group_size} and {PACKING}."
        )

    # W13: pad gate/up halves independently, then transpose to gmm_v2 (E, K, N).
    w13_w = _pad_gate_up_halves_axis1(w13, intermediate, pad_n)
    w13_s = _pad_gate_up_halves_axis1(w13_s, intermediate, pad_n)
    w13_w = w13_w.permute(0, 2, 1).contiguous()  # (E, H/2, 2*I_pad) u8
    w13_scale = _to_gmm_v2_scale_layout(w13_s, scale_dtype)  # (E, H/G, 1, 2*I_pad)

    # W2: pad packed-K and scale blocks to match, then transpose so N = hidden.
    pad_k_packed = pad_n // PACKING
    pad_k_blocks = pad_n // group_size
    w2_w = (
        _pad_last_axis(w2, pad_k_packed).permute(0, 2, 1).contiguous()
    )  # (E, I_pad/2, H)
    w2_s = _pad_last_axis(w2_s, pad_k_blocks)  # (E, H, I_pad/G)
    w2_scale = _to_gmm_v2_scale_layout(w2_s, scale_dtype)  # (E, I_pad/G, 1, H)

    copy_or_rebind_param(layer, "w13_weight", w13_w)
    copy_or_rebind_param(layer, "w2_weight", w2_w)
    copy_or_rebind_param(layer, "w13_weight_scale", w13_scale)
    copy_or_rebind_param(layer, "w2_weight_scale", w2_scale)

    for name in _UNUSED_NVFP4_MOE_PARAMS:
        if getattr(layer, name, None) is not None:
            _clear_replaced_param(layer, name)


def apply_tpu_nvfp4_moe(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    apply_router_weight_on_input: bool,
    no_combine: bool = False,
):
    """Run TPU NVFP4 MoE through the Pallas grouped-GEMM backend.

    With ``no_combine=True`` returns raw per-slot outputs ``[num_tokens, topk,
    hidden]`` (non-local EP slots zeroed) for a caller doing its own post-expert
    norm + combine.
    """
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported on TPU yet."
        )

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights, topk_ids = mask_tpu_moe_non_local_experts(topk_weights, topk_ids)

    output = fused_moe_call(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=int(topk_ids.shape[-1]),
        activation=activation,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        no_combine=no_combine,
    )
    return StandardCombineInput(hidden_states=output)
