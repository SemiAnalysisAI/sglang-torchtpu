"""Compatibility aliases for prod_inference's pre-NSA DSA imports.

prod_inference's sglang_meta can still import the older
sglang.srt.layers.attention.dsa utilities while the linked SGLang checkout has
renamed that implementation to attention.nsa.
"""

from sglang.srt.layers.attention.nsa.utils import (
    aiter_can_use_preshuffle_paged_mqa,
    cal_padded_tokens,
    can_nsa_cp_split,
    can_nsa_prefill_cp_round_robin_split,
    compute_nsa_seqlens,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_in_seq_split,
    is_nsa_prefill_cp_round_robin_split,
    nsa_cp_round_robin_split_data,
    nsa_cp_round_robin_split_q_seqs,
    nsa_cp_round_robin_split_q_seqs_cpu,
    nsa_cp_round_robin_split_q_seqs_kernel,
    nsa_use_prefill_cp,
    pad_nsa_cache_seqlens,
)


compute_dsa_seqlens = compute_nsa_seqlens
is_dsa_enable_prefill_cp = is_nsa_enable_prefill_cp
is_dsa_prefill_cp_in_seq_split = is_nsa_prefill_cp_in_seq_split
is_dsa_prefill_cp_round_robin_split = is_nsa_prefill_cp_round_robin_split
can_dsa_prefill_cp_round_robin_split = can_nsa_prefill_cp_round_robin_split
dsa_cp_round_robin_split_data = nsa_cp_round_robin_split_data
pad_dsa_cache_seqlens = pad_nsa_cache_seqlens
can_dsa_cp_split = can_nsa_cp_split
dsa_cp_round_robin_split_q_seqs_kernel = nsa_cp_round_robin_split_q_seqs_kernel
dsa_cp_round_robin_split_q_seqs_cpu = nsa_cp_round_robin_split_q_seqs_cpu
dsa_cp_round_robin_split_q_seqs = nsa_cp_round_robin_split_q_seqs
dsa_use_prefill_cp = nsa_use_prefill_cp

__all__ = [
    "aiter_can_use_preshuffle_paged_mqa",
    "cal_padded_tokens",
    "compute_dsa_seqlens",
    "is_dsa_enable_prefill_cp",
    "is_dsa_prefill_cp_in_seq_split",
    "is_dsa_prefill_cp_round_robin_split",
    "can_dsa_prefill_cp_round_robin_split",
    "dsa_cp_round_robin_split_data",
    "pad_dsa_cache_seqlens",
    "can_dsa_cp_split",
    "dsa_cp_round_robin_split_q_seqs_kernel",
    "dsa_cp_round_robin_split_q_seqs_cpu",
    "dsa_cp_round_robin_split_q_seqs",
    "dsa_use_prefill_cp",
]
