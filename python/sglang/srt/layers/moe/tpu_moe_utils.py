"""TPU-specific fused MoE helpers (mirrors rocm_moe_utils.py)."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.platforms import current_platform

# SGLANG_TPU_INT4_MOE_MODE values; None = default W4A16 group products.
INT4_MOE_MODE_VMEM_BF16 = "vmem-bf16"
INT4_MOE_MODE_VMEM_FP8 = "vmem-fp8"


def resolve_int4_moe_mode() -> Optional[str]:
    """The INT4 MoE mode from the environment, validated.

    Returns:
        ``None`` for the default W4A16 path, else one of
        ``INT4_MOE_MODE_VMEM_BF16`` / ``INT4_MOE_MODE_VMEM_FP8``.

    Raises:
        ValueError: The environment names a mode that does not exist.
    """
    from sglang.srt.environ import envs

    mode = envs.SGLANG_TPU_INT4_MOE_MODE.get()
    if mode in (None, ""):
        return None
    if mode not in (INT4_MOE_MODE_VMEM_BF16, INT4_MOE_MODE_VMEM_FP8):
        raise ValueError(f"Unsupported TPU INT4 MoE mode: {mode!r}")
    return mode


def mask_tpu_moe_non_local_experts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    moe_ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate global expert IDs to this EP rank's local IDs and mask the rest.

    The dispatcher hands the TPU runner global expert IDs
    (skip_local_expert_mapping). The TPU Pallas fused-MoE kernel takes local
    expert IDs here, so shift IDs into this rank's local range and mark the
    rest -1 (dropped by the kernel) with zero routing weight.
    """
    traced = current_platform.get_traced_rank(moe_ep_rank)
    rank = moe_ep_rank if traced is None else traced
    local_ids = topk_ids - rank * num_local_experts
    valid = (local_ids >= 0) & (local_ids < num_local_experts)
    return (
        topk_weights * valid.to(topk_weights.dtype),
        torch.where(valid, local_ids, -1),
    )


def resolve_tpu_moe_ep_routing(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    moe_ep_size: int,
    moe_ep_rank: int,
    *,
    default_global_num_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Resolve TPU MoE EP routing parameters, lifting ``moe_ep_rank`` when traced.

    Outside rank-lifted ``direct_compile`` (``get_traced_rank(moe_ep_rank) is None``),
    global expert IDs and this rank's ``expert_group_offset`` are passed directly
    to ``fused_moe_call``. Inside a rank-lifted ``direct_compile`` trace,
    ``expert_group_offset`` is a Python ``int`` custom-op argument and cannot
    carry the traced rank tensor, so ``topk_ids`` is shifted into
    ``[0, num_local_experts)`` via the traced rank tensor and
    ``expert_group_offset`` is set to ``0``.
    """
    if moe_ep_size <= 1:
        return topk_weights, topk_ids, default_global_num_experts, 0
    if current_platform.get_traced_rank(moe_ep_rank) is not None:
        topk_weights, topk_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts, moe_ep_rank
        )
        return topk_weights, topk_ids, default_global_num_experts, 0
    return (
        topk_weights,
        topk_ids,
        num_local_experts * moe_ep_size,
        num_local_experts * moe_ep_rank,
    )
