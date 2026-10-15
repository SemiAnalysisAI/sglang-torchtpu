# SPDX-License-Identifier: Apache-2.0
"""Fused router top-k for small batches on TPU.

Why: topk.biased_topk_impl (sigmoid + correction bias + top-k) lowers on TPU
     to an XLA sort over every expert plus a gather and reductions, about
     7 us per MoE layer for Kimi-K3's top-16 of 896 at batch 1.
How: AROUND topk.biased_topk_impl and topk.biased_grouped_topk_impl (single
     expert group only, which Kimi-K3 uses): batches of at most 64 tokens with sigmoid
     scoring, a shared bias vector and no fused shared experts run one Pallas
     kernel (sglang_tpu/moe/router_topk_tpu.py); everything else calls the
     original. AFTER TopK.__init__ registers the kernel's custom op before
     torch.compile traces the model.
Upstream PR or issue: not filed; a TPU branch next to the JIT router in
     select_experts would replace these hooks.
Remove when: select_experts exposes a platform router seam.
"""


def register_router_topk(result, *args, **kwargs):
    from sglang_tpu.moe.router_topk_tpu import register_router_topk_op

    register_router_topk_op()
    return result


def biased_topk(original, *args, **kwargs):
    from sglang_tpu.moe.router_topk_tpu import biased_topk_around

    return biased_topk_around(original, *args, **kwargs)


def biased_grouped_topk(original, *args, **kwargs):
    from sglang_tpu.moe.router_topk_tpu import biased_grouped_topk_around

    return biased_grouped_topk_around(original, *args, **kwargs)
