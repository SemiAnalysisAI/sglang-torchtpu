# SPDX-License-Identifier: Apache-2.0
"""Metadata-only eligibility for the opt-in GLM TPU attention front."""

import torch

from sglang.srt.environ import envs


def can_use_glm_fp8_mla_front(
    attn,
    hidden_states,
    positions,
    forward_batch,
    *,
    parallel,
    input_scattered,
    lora_enabled,
    experimental_lora,
    llama_4_scaling,
):
    """Keep all unqualified shapes, modules and execution modes on native MLA.

    This function reads tensor metadata only. The Pallas module is imported by
    the caller after this predicate succeeds, never on another backend.
    """
    if (
        not getattr(attn, "_tpu_glm_dense_dsa", False)
        or attn.use_dsa
        or attn.is_nextn
        or not forward_batch.forward_mode.is_decode()
        or tuple(hidden_states.shape) != (1, 6144)
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.device.type != "tpu"
        or not hidden_states.is_contiguous()
        or tuple(positions.shape) != (1,)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != hidden_states.device
        or not positions.is_contiguous()
        or llama_4_scaling is not None
        or lora_enabled
        or experimental_lora
        or input_scattered
        or parallel.dcp_enabled
        or parallel.dcp_replicate_q_proj
        or parallel.attn_cp_size != 1
        or parallel.attn_dp_size != 1
        or attn.use_deep_gemm_bmm
        or attn.current_attention_backend != "pallas_tpu_mla"
        or hasattr(attn, "prepare_attention_output_gate")
        or getattr(attn, "learnable_sink_param", None) is not None
        or getattr(attn, "_kimi_split_gguf_kv_b", False)
        or getattr(attn, "_use_min_latency_fused_a_gemm", False)
        or getattr(attn, "_use_min_latency_q_b_gemm", False)
        or hasattr(attn, "q_a_proj")
    ):
        return False
    if (
        attn.q_lora_rank,
        attn.kv_lora_rank,
        attn.num_local_heads,
        attn.qk_nope_head_dim,
        attn.qk_rope_head_dim,
    ) != (2048, 512, 4, 192, 64):
        return False

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    def tensor_matches(value, shape, dtype, *, contiguous=True):
        return (
            value is not None
            and tuple(value.shape) == shape
            and value.dtype == dtype
            and value.device == hidden_states.device
            and (not contiguous or value.is_contiguous())
        )

    a = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    b = getattr(attn, "q_b_proj", None)
    if type(a) is not ReplicatedLinear or type(b) is not ColumnParallelLinear:
        return False
    if b.gather_output:
        return False
    for layer, shape in ((a, (6144, 2624)), (b, (2048, 1024))):
        method = getattr(layer, "quant_method", None)
        scale = getattr(layer, "weight_scale", None)
        if (
            type(method) is not Fp8LinearMethod
            or method.use_marlin
            or method.use_mxfp8
            or getattr(layer, "weight_block_size", None) is not None
            or getattr(layer, "input_scale", None) is not None
            or getattr(layer, "bias", None) is not None
            or getattr(layer, "set_lora", False)
            or not tensor_matches(layer.weight, shape, torch.float8_e4m3fn)
            or scale is None
            or tuple(scale.shape) not in ((shape[1],), (shape[1], 1))
            or scale.dtype != torch.float32
            or scale.device != hidden_states.device
            or not scale.is_contiguous()
        ):
            return False
    # These attributes are the normal RMSNorm.forward_native contract. A
    # subclass or an alternate rounding/variance rule requires separate tests.
    for norm, width in ((attn.q_a_layernorm, 2048), (attn.kv_a_layernorm, 512)):
        if (
            type(norm) is not RMSNorm
            or norm.hidden_size != width
            or norm.variance_size_override is not None
            or norm.override_orig_dtype is not None
            or norm.cast_x_before_out_mul
            or norm.fp32_residual
            or norm.variance_epsilon != 1e-5
            or not tensor_matches(norm.weight, (width,), torch.bfloat16)
        ):
            return False
    rope = attn.rotary_emb
    if (
        type(rope) is not RotaryEmbedding
        or rope.head_size != 64
        or rope.rotary_dim != 64
        or rope.is_neox_style
        or rope.position_start != 0
        or hasattr(rope, "sin_cos_cache")
        or attn.use_deepseek_yarn_rope
    ):
        return False
    cache = rope.cos_sin_cache
    if (
        cache.ndim != 2
        or cache.shape[1] != 64
        or cache.shape[0] % 8 != 0
        or cache.dtype != torch.bfloat16
        or cache.device != hidden_states.device
        or not cache.is_contiguous()
    ):
        return False
    if not tensor_matches(attn.w_kc, (4, 192, 512), torch.bfloat16, contiguous=False):
        return False
    # Both qualified orientations share this allocation. The logical-view
    # variant passes w_kc directly; neither path permits a weight repack.
    if not attn.w_kc.transpose(-1, -2).is_contiguous():
        return False
    if getattr(getattr(attn, "kv_b_proj", None), "set_lora", False):
        return False
    return True


def run_glm_fp8_mla_front(attn, hidden_states, positions):
    from sglang.srt.hardware_backend.tpu.glm_fp8_mla_front import glm_fp8_mla_front

    a = attn.fused_qkv_a_proj_with_mqa
    b = attn.q_b_proj
    kc_logical = envs.SGLANG_TPU_GLM_FP8_MLA_KC_LOGICAL.get()
    cache = attn.rotary_emb.cos_sin_cache
    context_bound = getattr(attn, "_tpu_glm_dense_dsa_max_context", None)
    if type(context_bound) is int and 0 < context_bound <= cache.shape[0]:
        # The model constructor validates this bound against the serving
        # context. Slice existing values before the custom-op layout boundary:
        # otherwise XLA copies the entire 1M-position cache each decode step.
        # Eight-row alignment is required by the kernel's cache DMA tile.
        cache = cache[: ((context_bound + 7) // 8) * 8]
    return glm_fp8_mla_front(
        hidden_states,
        a.weight,
        a.weight_scale,
        attn.q_a_layernorm.weight,
        attn.kv_a_layernorm.weight,
        b.weight,
        b.weight_scale,
        attn.w_kc if kc_logical else attn.w_kc.transpose(-1, -2),
        positions,
        cache,
        attn.q_a_layernorm.variance_epsilon,
        attn.kv_a_layernorm.variance_epsilon,
        kc_logical=kc_logical,
    )
