# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM post-attention RMS epilogue and shared-input FP8 quantization.

Communication, residual addition, variance and rsqrt remain native Torch.
The auxiliary quantized operand is returned explicitly and belongs only to
the shared dense projection, not to the routed experts' block quantizer.
"""

from functools import partial

import torch

_EPILOGUE_OP = None


def register_glm_norm_shared_quant_op():
    """Register once during the eligible model constructor, before tracing."""
    global _EPILOGUE_OP
    if _EPILOGUE_OP is None:
        from sglang.srt.hardware_backend.tpu.glm_norm_shared_quant_kernel import (
            register_epilogue_quant_op,
        )

        _EPILOGUE_OP = register_epilogue_quant_op()


def can_use_glm_norm_shared_quant(
    decoder,
    hidden_states,
    residual,
    forward_batch,
    *,
    parallel,
    sp_active,
    input_scattered,
    lora_enabled,
    moe_a2a_none,
):
    """Read metadata only; unsupported model and communication paths fall back."""
    if (
        not getattr(decoder, "_tpu_glm_norm_shared_quant", False)
        or forward_batch is None
        or not forward_batch.forward_mode.is_decode()
        or decoder.is_nextn
        or not decoder.speculative_algorithm.is_none()
        or not decoder.is_layer_sparse
        or sp_active
        or input_scattered
        or lora_enabled
        or not moe_a2a_none
        or parallel.tp_size != 16
        or parallel.attn_tp_size != 16
        or parallel.attn_dp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.moe_ep_size != 16
        or parallel.dcp_enabled
        or parallel.dcp_replicate_q_proj
    ):
        return False

    def matches(value, shape, dtype):
        return (
            isinstance(value, torch.Tensor)
            and tuple(value.shape) == shape
            and value.dtype == dtype
            and value.device.type == "tpu"
            and value.device == hidden_states.device
            and value.is_contiguous()
        )

    if not matches(hidden_states, (1, 6144), torch.bfloat16) or not matches(
        residual, (1, 6144), torch.bfloat16
    ):
        return False

    from sglang.srt.layers.communicator import (
        CommunicateWithAllReduceAndLayerNormFn,
        LayerCommunicator,
        ScatterMode,
    )
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE

    communicator = decoder.layer_communicator
    if type(communicator) is not LayerCommunicator:
        return False
    context = communicator._context
    function = communicator._communicate_with_all_reduce_and_layer_norm_fn
    if (
        context.tp_size != 16
        or context.attn_tp_size != 16
        or context.attn_dp_size != 1
        or context.attn_cp_size != 1
        or context.cache is not None
        or not isinstance(function, partial)
        or function.func
        is not CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
        or function.keywords.get("residual_input_mode") is not ScatterMode.TP_ATTN_FULL
    ):
        return False

    norm = decoder.post_attention_layernorm
    if (
        communicator.post_attention_layernorm is not norm
        or type(norm) is not RMSNorm
        or norm.hidden_size != 6144
        or norm.variance_size_override is not None
        or norm.override_orig_dtype is not None
        or norm.cast_x_before_out_mul
        or norm.fp32_residual
        or norm.variance_epsilon != 1e-5
        or not matches(norm.weight, (6144,), torch.bfloat16)
        or not matches(getattr(norm, "_tpu_weight_fp32", None), (6144,), torch.float32)
    ):
        return False

    moe = decoder.mlp
    if (
        type(moe) is not DeepseekV2MoE
        or moe.is_nextn
        or moe._enable_a2a_moe
        or moe._fuse_shared_experts_inside_sbo
        or moe._shared_expert_tp1
        or moe.num_fused_shared_experts != 0
        or moe.alt_stream is not None
        or moe._moe_quant_once is not False
        or getattr(moe, "is_hash", False)
        or not hasattr(moe, "shared_experts")
    ):
        return False
    shared = moe.shared_experts
    if type(shared) is not DeepseekV2MLP or shared.swiglu_limit is not None:
        return False
    up, down = shared.gate_up_proj, shared.down_proj
    if (
        type(up) is not MergedColumnParallelLinear
        or type(down) is not RowParallelLinear
        or up.tp_size != 16
        or down.tp_size != 16
        or up.gather_output
        or not down.input_is_parallel
        or down.reduce_results
        or down.use_decode_attn_tp
        or down.use_dp_attention_reduce
    ):
        return False
    for layer, shape in ((up, (6144, 256)), (down, (128, 6144))):
        method = layer.quant_method
        scale = getattr(layer, "weight_scale", None)
        if (
            type(method) is not Fp8LinearMethod
            or method.use_marlin
            or method.use_mxfp8
            or (
                method.block_quant
                and getattr(layer, "weight_block_size", "missing") is not None
            )
            or getattr(layer, "input_scale", None) is not None
            or getattr(layer, "bias", None) is not None
            or getattr(layer, "set_lora", False)
            or not matches(layer.weight, shape, torch.float8_e4m3fn)
            or not any(
                matches(scale, scale_shape, torch.float32)
                for scale_shape in ((shape[1],), (shape[1], 1))
            )
        ):
            return False
    return True


def _norm_with_shared_quant(norm, hidden_states, residual):
    """Native RMS prefix with the same unrounded FP32 residual sum."""
    from sglang.srt.hardware_backend.tpu.fp8_linear import TpuFp8PrequantizedInput

    if _EPILOGUE_OP is None:
        raise RuntimeError("GLM norm/shared quant must be registered before tracing")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    z = hidden_states.to(torch.float32) + residual.to(torch.float32)
    residual_out = z.to(torch.bfloat16)
    variance = z.pow(2).mean(dim=-1, keepdim=True)
    inverse = torch.rsqrt(variance + norm.variance_epsilon)
    normalized, quantized, scale = _EPILOGUE_OP(z, inverse, norm._tpu_weight_fp32)
    prequantized = TpuFp8PrequantizedInput(
        quantized, scale.reshape(16, 1), (1, 6144), torch.bfloat16
    )
    return (normalized, prequantized), residual_out


def prepare_glm_norm_shared_quant(decoder, hidden_states, residual, forward_batch):
    """Use the existing communicator and keep the auxiliary result call-local."""
    packed_hidden, residual = decoder.layer_communicator.prepare_mlp(
        hidden_states,
        residual,
        forward_batch,
        norm_override=partial(_norm_with_shared_quant, decoder.post_attention_layernorm),
    )
    normalized, prequantized = packed_hidden
    return normalized, residual, prequantized
