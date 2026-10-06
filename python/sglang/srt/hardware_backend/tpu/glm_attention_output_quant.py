# SPDX-License-Identifier: Apache-2.0
"""Optional GLM BS1 absorbed-value expansion and dense-input quantization.

Only the prepared BF16 WV path is eligible. The output projection and its
collective retain their existing consumers and numerical boundaries.
"""

import torch

_WV_QUANT_OP = None


def register_glm_attention_output_quant_op():
    """Resolve registration during construction, before model tracing."""
    global _WV_QUANT_OP
    if _WV_QUANT_OP is None:
        from sglang.srt.hardware_backend.tpu.glm_attention_output_quant_kernel import (
            register_wv_quant_op,
        )

        _WV_QUANT_OP = register_wv_quant_op()


def can_use_glm_attention_output_quant(
    attn,
    attention,
    forward_batch,
    *,
    parallel,
    sp_active,
    input_scattered,
    lora_enabled,
    experimental_lora,
    kv_b_lora_active,
    attention_output_gate,
    llama_4_scaling,
    fusion_plan,
):
    """Read metadata only; all unqualified paths keep their original BMM."""
    if (
        not getattr(attn, "_tpu_glm_attention_output_quant", False)
        or not getattr(attn, "_tpu_glm_dense_dsa", False)
        or forward_batch is None
        or not forward_batch.forward_mode.is_decode()
        or forward_batch.spec_info is not None
        or (
            forward_batch.spec_algorithm is not None
            and not forward_batch.spec_algorithm.is_none()
        )
        or attn.is_nextn
        or attn.current_attention_backend != "pallas_tpu_mla"
        or attn.use_deep_gemm_bmm
        or getattr(attn, "_kimi_split_gguf_kv_b", False)
        or hasattr(attn, "prepare_attention_output_gate")
        or getattr(attn, "learnable_sink_param", None) is not None
        or sp_active
        or input_scattered
        or lora_enabled
        or experimental_lora
        or kv_b_lora_active
        or attention_output_gate is not None
        or llama_4_scaling is not None
        or fusion_plan is not None
        or parallel.tp_size != 16
        or parallel.attn_tp_size != 16
        or parallel.attn_dp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.dcp_enabled
        or parallel.dcp_replicate_q_proj
        or (attn.num_local_heads, attn.kv_lora_rank, attn.v_head_dim) != (4, 512, 256)
    ):
        return False

    def matches(value, shape, dtype, *, contiguous=True):
        return (
            isinstance(value, torch.Tensor)
            and tuple(value.shape) == shape
            and value.dtype == dtype
            and value.device.type == "tpu"
            and value.device == attention.device
            and (not contiguous or value.is_contiguous())
        )

    if not matches(attention, (1, 4, 512), torch.bfloat16):
        return False
    if not matches(attn.w_vc, (4, 512, 256), torch.bfloat16, contiguous=False):
        return False
    if not attn.w_vc.transpose(-1, -2).is_contiguous():
        return False
    if getattr(getattr(attn, "kv_b_proj", None), "set_lora", False):
        return False

    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    projection = attn.o_proj
    if type(projection) is not RowParallelLinear:
        return False
    method = projection.quant_method
    if (
        type(method) is not Fp8LinearMethod
        or method.use_marlin
        or method.use_mxfp8
        or (
            method.block_quant
            and getattr(projection, "weight_block_size", "missing") is not None
        )
        or getattr(projection, "input_scale", None) is not None
        or projection.tp_size != 16
        or not projection.input_is_parallel
        or projection.reduce_results
        or projection.use_decode_attn_tp
        or projection.use_dp_attention_reduce
        or projection.bias is not None
        or getattr(projection, "set_lora", False)
        or not matches(projection.weight, (1024, 6144), torch.float8_e4m3fn)
        or not any(
            matches(projection.weight_scale, shape, torch.float32)
            for shape in ((6144,), (6144, 1))
        )
    ):
        return False
    return True


def prepare_glm_attention_output_quant(attn, attention):
    """Pass one call-local carrier to the existing native output projection."""
    from sglang.srt.hardware_backend.tpu.fp8_linear import TpuFp8PrequantizedInput

    if _WV_QUANT_OP is None:
        raise RuntimeError(
            "GLM attention-output quantization must register before tracing"
        )
    quantized, scale = _WV_QUANT_OP(attention, attn.w_vc.transpose(-1, -2))
    return TpuFp8PrequantizedInput(
        quantized, scale.reshape(16, 1), (1, 1024), torch.bfloat16
    )
