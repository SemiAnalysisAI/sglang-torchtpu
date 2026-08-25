# SPDX-License-Identifier: Apache-2.0
"""Fused pre-MoE control-path primitives for narrow sequence parallelism."""

from .fused_attention_moe_prelude import (
    fused_attention_moe_prelude,
    ref_fused_attention_moe_prelude,
)
from .fused_attention_moe_prelude_replicated import (
    fused_attention_moe_prelude_router_pooler_replicated,
)
from .fused_moe_dispatch_sp import (
    fused_replicated_moe_dispatch_metadata,
    fused_sp_moe_dispatch_metadata,
)
from .fused_moe_router import fused_moe_router, ref_fused_moe_router

__all__ = [
    "fused_attention_moe_prelude",
    "fused_attention_moe_prelude_router_pooler_replicated",
    "fused_moe_router",
    "fused_replicated_moe_dispatch_metadata",
    "fused_sp_moe_dispatch_metadata",
    "ref_fused_attention_moe_prelude",
    "ref_fused_moe_router",
]
