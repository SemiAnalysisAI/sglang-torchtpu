# SPDX-License-Identifier: Apache-2.0
"""Load-time FP32 norm-weight buffers for the opt-in GLM TPU direct path."""

import torch


def _refresh_norm_weight(module):
    """Keep the buffer object stable: compiled executables retain its identity."""
    with torch.no_grad():
        cached = getattr(module, "_tpu_weight_fp32", None)
        if cached is None:
            module.register_buffer(
                "_tpu_weight_fp32", module.weight.detach().float(), persistent=False
            )
        else:
            if (
                cached.shape != module.weight.shape
                or cached.device != module.weight.device
                or cached.dtype != torch.float32
            ):
                raise RuntimeError(
                    "A cached GLM RMSNorm weight changed shape, device or dtype; "
                    "rebuild the model before compiling it again"
                )
            cached.copy_(module.weight)


def _refresh_after_load_state_dict(module, incompatible_keys):
    _refresh_norm_weight(module)


def refresh_glm_rmsnorm_weights(model):
    """Refresh only decoder input/post-attention and final RMS6144 weights.

    The model marker is set only for the explicit TPU/GLM/TP16/direct opt-in.
    QA/KV norms retain their original BF16 contract with the fused MLA front.
    This function runs after loading, never from the token-forward hot path.
    """
    if not getattr(model, "_tpu_glm_rmsnorm_fp32_weights", False):
        return 0

    from sglang.srt.layers.layernorm import RMSNorm

    targets = []
    for layer in model.model.layers.children():
        for name in ("input_layernorm", "post_attention_layernorm"):
            targets.append(getattr(layer, name, None))
    targets.append(getattr(model.model, "norm", None))

    count = 0
    for norm in targets:
        if not (
            type(norm) is RMSNorm
            and norm.hidden_size == 6144
            and norm.has_weight
            and norm.variance_size_override is None
            and norm.override_orig_dtype is None
            and not norm.cast_x_before_out_mul
            and not norm.fp32_residual
            and tuple(norm.weight.shape) == (6144,)
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.device.type == "tpu"
            and norm.weight.is_contiguous()
        ):
            if getattr(norm, "_tpu_weight_fp32", None) is not None:
                raise RuntimeError(
                    "A cached GLM RMSNorm no longer satisfies its load-time "
                    "contract; rebuild the model before serving again"
                )
            continue
        _refresh_norm_weight(norm)
        if not hasattr(norm, "_tpu_weight_fp32_load_hook"):
            norm._tpu_weight_fp32_load_hook = norm.register_load_state_dict_post_hook(
                _refresh_after_load_state_dict
            )
        count += 1
    return count
