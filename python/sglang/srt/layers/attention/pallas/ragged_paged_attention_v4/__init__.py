"""SGLang ragged-paged-attention v4 kernel."""

from .kernel import (
    DEFAULT_MASK_VALUE,
    DEFAULT_VMEM_LIMIT_BYTES,
    KVLayout,
    dynamic_validate_inputs,
    get_kv_cache_shape,
    get_smem_estimate_bytes,
    get_vmem_estimate_bytes,
    merge_kv,
    prepare_inputs,
    prepare_outputs,
    ragged_paged_attention,
    ref_ragged_paged_attention,
    static_validate_inputs,
)

# Backward-compatible aliases for unified kernel
ragged_paged_attention_hd64 = ragged_paged_attention
ref_ragged_paged_attention_hd64 = ref_ragged_paged_attention

from .tuned_block_sizes import get_tuned_block_sizes
from .util import (
    align_to,
    cdiv,
    get_dtype_bitwidth,
    get_dtype_packing,
    get_lookup_keys,
    next_power_of_2,
)

__all__ = [
    "DEFAULT_MASK_VALUE",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "KVLayout",
    "align_to",
    "cdiv",
    "dynamic_validate_inputs",
    "get_dtype_bitwidth",
    "get_dtype_packing",
    "get_lookup_keys",
    "get_smem_estimate_bytes",
    "get_tuned_block_sizes",
    "get_vmem_estimate_bytes",
    "merge_kv",
    "next_power_of_2",
    "prepare_inputs",
    "prepare_outputs",
    "get_kv_cache_shape",
    "ragged_paged_attention",
    "ref_ragged_paged_attention",
    "static_validate_inputs",
]
