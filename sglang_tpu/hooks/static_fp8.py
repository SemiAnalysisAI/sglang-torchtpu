# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 post-load binding of the opt-in static-scale fp8 decode GEMVs.

Why: SGLANG_TPU_K3_STATIC_FP8 replaces RMSNorm-fed bf16 GEMVs with fp8 ones
     (sglang_tpu.kimi_k3.static_fp8); they must be bound after the weights
     load and before compilation, and core must not import the plugin to do
     it.
How: AFTER KimiK3LinearForCausalLM.post_load_weights; bind_static_fp8 sets the
     model's _static_fp8_* callables and is a no-op when the variable is unset.
Upstream PR or issue: TBD (platform post-load seam for model weights).
Remove when: core exposes a platform post-load seam for model weights.
"""


def bind_kimi_k3_static_fp8(result, model, *args, **kwargs):
    from sglang_tpu.kimi_k3.static_fp8 import bind_static_fp8

    bind_static_fp8(model)
    return result
