# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 TP32 decode megakernel integration for SGLang on TPU."""

from .config import (
    is_megakernel_enabled,
    megakernel_context,
    megakernel_max_rows,
    validate_megakernel_server_args,
)
from .handoff import (
    apply_row_move,
    build_padded_token_locs_and_mask,
    extract_mamba_buffers,
    extract_mla_kv_buffers,
    handoff_request_to_megakernel_row,
)
from .program import decode_options, layer_counts, make_decode_program, state_shapes
from .runner import (
    KimiK3MegakernelRunner,
    allocate_decode_states,
    maybe_attach_decode_megakernel,
    register_megakernel_decode_op,
)
from .slots import ROW_BUCKETS, RowMove, SlotTable, row_bucket
from .weights import (
    EXPERT_FAMILIES,
    e8m0_from_f32_scales,
    expert_families_from_sglang,
    load_dense_families,
    stack_sglang_moe_experts_in_place,
)

__all__ = [
    "EXPERT_FAMILIES",
    "KimiK3MegakernelRunner",
    "ROW_BUCKETS",
    "RowMove",
    "SlotTable",
    "allocate_decode_states",
    "apply_row_move",
    "build_padded_token_locs_and_mask",
    "decode_options",
    "e8m0_from_f32_scales",
    "expert_families_from_sglang",
    "extract_mamba_buffers",
    "extract_mla_kv_buffers",
    "handoff_request_to_megakernel_row",
    "is_megakernel_enabled",
    "layer_counts",
    "load_dense_families",
    "make_decode_program",
    "maybe_attach_decode_megakernel",
    "megakernel_context",
    "megakernel_max_rows",
    "register_megakernel_decode_op",
    "row_bucket",
    "stack_sglang_moe_experts_in_place",
    "state_shapes",
    "validate_megakernel_server_args",
]
