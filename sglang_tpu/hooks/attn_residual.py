# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 attention-residual decode kernel on TPU.

Why: on TPU, attn_residual's aggregation falls back to the generic torch
     path (cat, score RMSNorm, score proj, softmax, mix, output RMSNorm),
     about a dozen tiny XLA ops per call and two calls per Kimi-K3 layer.
How: AROUND attn_residual._aggregate and _aggregate_fused_add: decode-sized
     calls run one fused Pallas kernel (sglang_tpu/attn_residual_tpu.py),
     everything else calls the original. AFTER KimiK3LinearModel.__init__
     registers the kernel's custom ops before torch.compile traces the model.
Upstream PR or issue: not filed; a TPU dispatch branch next to the CUDA/HIP
     fast paths in attn_residual._aggregate would replace these hooks.
Remove when: attn_residual exposes a platform kernel seam for aggregation.
"""


def register_attn_residual_ops(result, *args, **kwargs):
    from sglang_tpu.attn_residual_tpu import register_attn_residual_decode_ops

    register_attn_residual_decode_ops()
    return result


def aggregate(original, *args, **kwargs):
    from sglang_tpu.attn_residual_tpu import aggregate_around

    return aggregate_around(original, *args, **kwargs)


def aggregate_fused_add(original, *args, **kwargs):
    from sglang_tpu.attn_residual_tpu import aggregate_fused_add_around

    return aggregate_fused_add_around(original, *args, **kwargs)
