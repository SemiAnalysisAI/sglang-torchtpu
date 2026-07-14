# SPDX-License-Identifier: Apache-2.0
"""TPU NVFP4 (``modelopt_fp4``) MoE helpers.

Serves NVFP4 MoE weights through the packed-FP4 path of the Pallas ``gmm_v2``
grouped-GEMM kernel (uint8 weights + per-block ``rhs_scale``). NVFP4 carries a
two-level scale (fp8 per-16-block + fp32 per-tensor global), but ``gmm_v2`` takes
only one per-block scale, so at load we fold the global into the block scale
(``real = fp4_code * fp8_block_scale * global``) — format conversion only.

There is no native fp4 MXU on TPU: the kernel unpacks fp4 weights to fp8 and
quantizes activations to fp8 in-kernel (W4-stored / A8-compute), so the
checkpoint's ``input_scale`` is unused.
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

# Extra params NVFP4 create_weights registers that the TPU gmm_v2 path does not
# consume (globals are folded into the block scale; input scales are unused
# because activations are quantized dynamically in-kernel; the *_swizzled and
# cutlass/cutedsl placeholders are CUDA-only).
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
    """Pre-multiply NVFP4's per-tensor global into the fp8 per-block scale.

    NVFP4 stores two scale levels (fp8 per-16-block + fp32 per-tensor global),
    but gmm_v2 accepts only one per-block scale. We fold them into a single fp32
    block scale (``real = fp4_code * fp8_block_scale * global``) so the kernel
    stays single-scale. Gate and up carry different globals, so W13 folds per
    half.

    ``block_scale`` is ``(E, 2I, H/16)`` fp8-e4m3, rows ``[Gate(I), Up(I)]``;
    ``global_scale`` is ``(E, 2)`` (gate, up) or ``(E,)`` if shared.
    """
    s = block_scale.float()
    g = global_scale.float()
    # Only the true per-half form (E, 2) gets the split fold; (E,) or a 2-D
    # single-column (E, 1) shared global takes the scalar-per-expert branch.
    if is_gated and g.dim() == 2 and g.shape[1] == 2:
        half = s.shape[1] // 2
        s[:, :half, :] *= g[:, 0:1, None]
        s[:, half:, :] *= g[:, 1:2, None]
    else:
        g_flat = g if g.dim() == 1 else g[:, 0]
        s *= g_flat[:, None, None]
    return s


def _to_gmm_v2_scale_layout(scale: torch.Tensor) -> torch.Tensor:
    """``(E, OUT, num_blocks_K)`` -> gmm_v2 ``(E, num_blocks_K, 1, OUT)`` fp32."""
    return scale.permute(0, 2, 1).contiguous().unsqueeze(2).to(torch.float32)


def prepare_tpu_nvfp4_moe_weights(
    layer: torch.nn.Module,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
    align_to: int = 128,
) -> None:
    """Format NVFP4 MoE weights/scales for the TPU Pallas gmm_v2 backend.

    Keeps weights packed (uint8), folds the two-level scales into a single fp32
    per-block scale, and transposes both into gmm_v2's ``(E, K_packed, N)`` /
    ``(E, num_blocks_K, 1, N)`` layout. Intermediate is padded to ``align_to``
    (gate/up-symmetric for W13, K-matched for W2).

    Layout in (checkpoint) -> out (gmm_v2):
      * ``w13_weight``       (E, 2I, H/2) u8  -> (E, H/2, 2*I_pad) u8
      * ``w13_weight_scale`` (E, 2I, H/16) f8 -> (E, H/16, 1, 2*I_pad) f32
      * ``w2_weight``        (E, H, I/2) u8   -> (E, I_pad/2, H) u8
      * ``w2_weight_scale``  (E, H, I/16) f8  -> (E, I_pad/16, 1, H) f32
    """
    if group_size != NVFP4_GROUP_SIZE:
        raise NotImplementedError(
            f"TPU NVFP4 MoE only supports group_size={NVFP4_GROUP_SIZE}, "
            f"got {group_size}."
        )

    if not layer.moe_runner_config.is_gated:
        # The Pallas fused-MoE GMM1 always splits its output in half for the
        # gated SwiGLU (``apply_act_fn`` on ``shape[-1] // 2``), so a non-gated
        # expert would be mis-activated downstream. Reject it here rather than
        # silently miscompute.
        raise NotImplementedError("TPU NVFP4 MoE only supports gated (SwiGLU) experts.")

    w13 = layer.w13_weight.data  # (E, 2I, H/2) uint8
    w2 = layer.w2_weight.data  # (E, H, I/2) uint8
    _E, two_I, _hidden_packed = w13.shape
    intermediate = two_I // 2

    # Fold the per-tensor global scale into the fp8 block scale (-> fp32).
    w13_s = _fold_w13_global_scale(
        layer.w13_weight_scale.data, layer.w13_weight_scale_2.data, is_gated=True
    )  # (E, 2I, H/16) f32
    w2_s = (
        layer.w2_weight_scale.data.float()
        * layer.w2_weight_scale_2.data.float()[:, None, None]
    )  # (E, H, I/16) f32

    # Pad the intermediate dim to align_to. pad_n stays a multiple of group_size
    # and PACKING (both divide align_to and intermediate), keeping scale blocks
    # and packed bytes whole; the check below enforces it.
    pad_n = (align_to - intermediate % align_to) % align_to
    if pad_n % group_size != 0 or pad_n % PACKING != 0:
        raise ValueError(
            f"align_to={align_to} incompatible with group_size={group_size}: "
            f"pad_n={pad_n} must be a multiple of both {group_size} and {PACKING}."
        )

    # W13: pad gate/up halves independently (weights along the packed axis,
    # scales along the block axis), then transpose to gmm_v2 (E, K, N).
    w13_w = _pad_gate_up_halves_axis1(w13, intermediate, pad_n)
    w13_s = _pad_gate_up_halves_axis1(w13_s, intermediate, pad_n)
    w13_w = w13_w.permute(0, 2, 1).contiguous()  # (E, H/2, 2*I_pad) u8
    w13_scale = _to_gmm_v2_scale_layout(w13_s)  # (E, H/16, 1, 2*I_pad) f32

    # W2: pad the (packed) K dim and scale blocks to match the padded
    # intermediate, then transpose so N = hidden.
    pad_k_packed = pad_n // PACKING
    pad_k_blocks = pad_n // group_size
    w2_w = (
        _pad_last_axis(w2, pad_k_packed).permute(0, 2, 1).contiguous()
    )  # (E, I_pad/2, H)
    w2_s = _pad_last_axis(w2_s, pad_k_blocks)  # (E, H, I_pad/16)
    w2_scale = _to_gmm_v2_scale_layout(w2_s)  # (E, I_pad/16, 1, H) f32

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
    """Run TPU NVFP4 MoE through the Pallas fused grouped-GEMM backend.

    With ``no_combine=True`` the kernel returns the raw per-slot expert outputs
    ``[num_tokens, topk, hidden]`` (non-local EP slots zeroed) so a caller doing
    a post-expert norm can RMS-norm each slot before the weighted reduction.
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
