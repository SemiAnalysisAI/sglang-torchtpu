# SPDX-License-Identifier: Apache-2.0
"""Static-scale fp8 for Kimi-K3's RMSNorm-fed decode GEMVs on TPU (opt-in).

Decode GEMVs are bound by the weight read. Each GEMV handled here reads an
RMSNorm output y = x / rms(x) * g, so |y_i| <= sqrt(K) * |g_i| bounds the fp8
input scale: no per-step amax, and the read halves. SGLANG_TPU_K3_STATIC_FP8
selects which ones:

- moe_front: the replicated latent down-proj and the shared experts' gate_up,
  merged into one GEMV (they read post_attention_layernorm's output);
- latent_up: the replicated latent up-proj (reads routed_expert_norm's output);
- kda_qkvg: the KDA fused [q, k, v, g] projection (reads input_layernorm's);
- mla_qkv_a: the MLA fused q_a / kv_a projection (reads input_layernorm's).

The model only holds the bound StaticFp8Weight attributes and calls them; the
bf16 weights they replace are released. Kimi-K3 on TPU already rejects online
weight updates, so nothing reads the released weights again.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.fp8_linear import StaticFp8Weight, quantize_rmsnorm_fed_weight

logger = logging.getLogger(__name__)

PARTS = frozenset({"moe_front", "latent_up", "kda_qkvg", "mla_qkv_a"})


def selected_parts() -> frozenset[str]:
    """The GEMVs SGLANG_TPU_K3_STATIC_FP8 selects; raises on unknown names."""
    parts = set(tpu_envs.SGLANG_TPU_K3_STATIC_FP8.get())
    unknown = parts - PARTS - {"all"}
    if unknown:
        raise ValueError(
            f"SGLANG_TPU_K3_STATIC_FP8: unknown {sorted(unknown)}; expected a "
            f"subset of {sorted(PARTS)} or 'all'"
        )
    return PARTS if "all" in parts else frozenset(parts)


def _pack(weight_kn: torch.Tensor, norm_weight: torch.Tensor) -> StaticFp8Weight:
    """Quantize, then pad N for the Pallas GEMV when it is registered."""
    from sglang_tpu import static_fp8_gemv_pallas

    return static_fp8_gemv_pallas.pad_n(
        quantize_rmsnorm_fed_weight(weight_kn, norm_weight)
    )


def _dense_weight_kn(linear: nn.Module, k: int) -> Optional[torch.Tensor]:
    """An unquantized linear's weight as [K, N], or None if it is quantized."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod):
        return None
    w = linear.weight.detach()
    if w.dim() != 2:  # already released by an earlier bind
        return None
    if w.shape[1] == k:
        return w.t()
    return w if w.shape[0] == k else None


def _release(*linears: nn.Module) -> int:
    """Drop the bf16 weights a static fp8 GEMV replaced; returns bytes freed.

    The Parameter object stays (its loader attributes too); only its storage
    goes.
    """
    freed = 0
    for linear in linears:
        w = linear.weight
        freed += w.numel() * w.element_size()
        w.data = torch.empty(0, dtype=w.dtype, device=w.device)
    return freed


def _bind_moe(moe, in_norm_weight: torch.Tensor, parts) -> tuple[list[str], int]:
    """Returns (parts bound on this layer, bytes freed)."""
    bound: list[str] = []
    freed = 0
    if not moe.use_latent_moe:
        return bound, freed
    hidden = in_norm_weight.shape[0]
    if (
        "moe_front" in parts
        and moe.shared_experts is not None
        and not moe._shared_experts_tp_comm
        and not moe._sbo_shared_overlap
        and not moe._eligible_for_fused_front
    ):
        down = _dense_weight_kn(moe.routed_expert_down_proj, hidden)
        gate_up = _dense_weight_kn(moe.shared_experts.gate_up_proj, hidden)
        if down is not None and gate_up is not None:
            # One GEMV for both: they read the same normed hidden state, so the
            # activation is quantized once.
            moe._static_fp8_front = _pack(
                torch.cat([down, gate_up], dim=1), in_norm_weight
            )
            moe._static_fp8_front_split = [down.shape[1], gate_up.shape[1]]
            freed += _release(
                moe.routed_expert_down_proj, moe.shared_experts.gate_up_proj
            )
            bound.append("moe_front")
    if "latent_up" in parts and moe.routed_expert_norm is not None:
        up = _dense_weight_kn(moe.routed_expert_up_proj, moe.moe_hidden_size)
        if up is not None:
            moe._static_fp8_latent_up = _pack(
                up, moe.routed_expert_norm.weight
            )
            freed += _release(moe.routed_expert_up_proj)
            bound.append("latent_up")
    return bound, freed


def _bind_linear(owner, attr: str, linear: Optional[nn.Module], norm_weight):
    w = None if linear is None else _dense_weight_kn(linear, norm_weight.shape[0])
    if w is None or w.shape[1] % 16:
        return 0, 0
    setattr(owner, attr, _pack(w, norm_weight))
    return 1, _release(linear)


def bind_static_fp8(model: nn.Module) -> None:
    """Bind the selected static fp8 GEMVs on every Kimi-K3 decoder layer.

    Runs post-load, before compilation. Idempotent: post_load_weights can run
    twice (from load_weights and from the multimodal wrapper), and the second
    call must not touch the released weights.
    """
    parts = selected_parts()
    if not parts or getattr(model, "_static_fp8_bound", False):
        return
    model._static_fp8_bound = True
    from sglang_tpu import static_fp8_gemv_pallas

    static_fp8_gemv_pallas.register()
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.models.kimi_k3 import (
        KimiK3DeltaAttention,
        KimiK3MLAAttention,
        KimiK3MoE,
    )

    counts = dict.fromkeys(sorted(parts), 0)
    freed = 0
    for layer in model.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        if isinstance(layer.mlp, KimiK3MoE):
            bound, f = _bind_moe(
                layer.mlp, layer.post_attention_layernorm.weight, parts
            )
            for part in bound:
                counts[part] += 1
            freed += f
        attn = layer.self_attn
        norm = layer.input_layernorm.weight
        if (
            "kda_qkvg" in parts
            and isinstance(attn, KimiK3DeltaAttention)
            and attn.use_full_rank_gate
        ):
            n, f = _bind_linear(
                attn, "_static_fp8_qkvg", getattr(attn, "fused_qkvg_proj", None), norm
            )
            counts["kda_qkvg"] += n
            freed += f
        if "mla_qkv_a" in parts and isinstance(attn, KimiK3MLAAttention):
            n, f = _bind_linear(
                attn,
                "_static_fp8_qkv_a",
                getattr(attn, "fused_qkv_a_proj_with_mqa", None),
                norm,
            )
            counts["mla_qkv_a"] += n
            freed += f
    if freed and not getattr(model, "_derived_weight_cache_error", None):
        model._derived_weight_cache_error = (
            "Online weight updates are not supported with SGLANG_TPU_K3_STATIC_FP8: "
            "the bf16 weights its fp8 GEMVs replace were released at load time."
        )
    logger.info(
        "Kimi-K3 static fp8 GEMVs bound per part (layers): %s; freed %.1f MiB of "
        "bf16 weights",
        counts,
        freed / 2**20,
    )


__all__ = ["PARTS", "StaticFp8Weight", "bind_static_fp8", "selected_parts"]
