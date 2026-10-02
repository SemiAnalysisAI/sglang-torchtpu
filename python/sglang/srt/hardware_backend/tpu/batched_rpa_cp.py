# SPDX-License-Identifier: Apache-2.0
"""Context Parallelism (CP) extensions for the TPU Pallas Batched RPA backend.

Supports:
1. Prefill Context Parallelism (PCP) via ICI Ring Attention:
   - Cache Phase (CACHE_ONLY): KV cache is page-striped across CP ranks (1/CP memory).
     KV blocks are rotated around the TPU mesh ring in VMEM via Pallas async remote copy
     (pltpu.make_async_remote_copy), accumulating attention in a single online softmax pass.
   - Current Phase (NEW_TOKENS_ONLY): Current chunk tokens are all-gathered across the CP
     group, attended causally, and selectively written back to the rank's owned pages.
   - Softmax state merge: Combines Cache Phase and Current Phase outputs using Log-Sum-Exp.
2. Decode Context Parallelism (DCP) via position-sharded KV cache:
   - Evaluates NEW_TOKENS_ONLY and CACHE_ONLY with rank-selective writeback, followed
     by group LSE reduction.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_CP_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_batched_cp"
_OP_CACHE: dict = {}
_CP_RANK_TENSORS: dict[torch.device, torch.Tensor] = {}


def ensure_cp_rank_tensor(device: torch.device, cp_rank: int) -> torch.Tensor:
    """Create a persistent device tensor for cp_rank to avoid graph recompilations."""
    device = torch.device(device)
    tensor = _CP_RANK_TENSORS.get(device)
    if tensor is None:
        tensor = torch.tensor([cp_rank], dtype=torch.int32, device=device)
        _CP_RANK_TENSORS[device] = tensor
    return tensor


@dataclass
class PcpBatchedMetadata:
    """Metadata for Batched RPA Prefill Context Parallelism."""

    pcp_cu_q_lens: torch.Tensor
    q_positions: torch.Tensor
    new_kv_page_indices: torch.Tensor
    update_kv_cache: torch.Tensor
    kv_new_lens: torch.Tensor
    striped_page_indices: torch.Tensor
    cache_pages: int
    num_reqs: int
    pcp_cache_lens: torch.Tensor


def build_batched_rpa_cp_op():
    """Register the CP-aware custom op for batched RPA."""
    cached = _OP_CACHE.get(_CP_OP_NAME)
    if cached is not None:
        return cached

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_batched import kv_layout
    from sglang.srt.layers.attention.pallas.batched_rpa import configs, wrapper

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        q_positions: jax.Array,
        kv_new_lens: jax.Array,
        cp_rank: jax.Array,
        new_kv_page_indices: jax.Array | None = None,
        update_kv_cache: jax.Array | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        attention_scope: int = 0,
        cp_group_size: int = 1,
        pcp_ring_axis_name: str | None = None,
        return_lse: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(attention_scope, int):
            _scopes = [
                configs.AttentionScope.FULL,
                configs.AttentionScope.CACHE_ONLY,
                configs.AttentionScope.NEW_TOKENS_ONLY,
            ]
            scope_enum = _scopes[attention_scope]
        elif isinstance(attention_scope, str):
            scope_enum = configs.AttentionScope(attention_scope.lower())
        else:
            scope_enum = attention_scope
        cp_size_arg = cp_group_size if cp_group_size > 1 else None
        ring_mesh = ("pcp", "model") if pcp_ring_axis_name is not None else None
        new_pages = (
            None
            if (
                scope_enum == configs.AttentionScope.CACHE_ONLY
                or new_kv_page_indices is None
            )
            else new_kv_page_indices
        )

        res = wrapper.ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache_fused,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            use_causal_mask=True,
            attention_scope=scope_enum,
            cp_group_size=cp_size_arg,
            cp_rank=cp_rank,
            pcp_ring_axis_name=pcp_ring_axis_name,
            pcp_ring_mesh_axis_names=ring_mesh,
            q_positions=q_positions,
            kv_new_lens=kv_new_lens,
            new_kv_page_indices=new_pages,
            update_kv_cache=update_kv_cache,
            return_lse=return_lse,
            kv_layout=kv_layout(),
        )
        if return_lse:
            out, kv_updated, lse = res
            return out, kv_updated, lse
        else:
            out, kv_updated = res
            return out, kv_updated, None

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{_CP_OP_NAME}",
        # Static non-traced arguments
        static_argnums=(13, 14, 15, 16, 17, 18, 19, 20, 21),
        donate_argnums=(3,),
        input_output_aliases={3: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _CP_OP_NAME,
        mutates_args=("kv_cache_fused",),
        device_types=["tpu"],
    )
    def custom_op(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        q_positions: torch.Tensor,
        kv_new_lens: torch.Tensor,
        cp_rank: torch.Tensor,
        new_kv_page_indices: torch.Tensor | None = None,
        update_kv_cache: torch.Tensor | None = None,
        sm_scale: float = 1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        attention_scope: int = 0,
        cp_group_size: int = 1,
        pcp_ring_axis_name: str | None = None,
        return_lse: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, _, lse = jax_kernel(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            q_positions,
            kv_new_lens,
            cp_rank,
            new_kv_page_indices,
            update_kv_cache,
            sm_scale,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
            attention_scope,
            cp_group_size,
            pcp_ring_axis_name,
            return_lse,
        )
        return out, lse

    @custom_op.register_fake
    def _fake_impl(queries, keys, values, kv_cache_fused, *args, **kwargs):
        lse = queries.new_empty(queries.shape[0], queries.shape[1], dtype=torch.float32)
        return torch.empty_like(queries), lse

    _OP_CACHE[_CP_OP_NAME] = custom_op
    return custom_op


def merge_attn_states(
    context_out: torch.Tensor,
    context_lse: torch.Tensor,
    curr_out: torch.Tensor,
    curr_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention states across cache phase and current phase using Log-Sum-Exp.

    Args:
        context_out: [num_tokens, num_heads, head_dim]
        context_lse: [num_tokens, num_heads]
        curr_out: [num_tokens, num_heads, head_dim]
        curr_lse: [num_tokens, num_heads]
    Returns:
        merged_out: [num_tokens, num_heads, head_dim]
        merged_lse: [num_tokens, num_heads]
    """
    max_lse = torch.maximum(context_lse, curr_lse)
    exp_context = torch.exp(context_lse - max_lse)
    exp_curr = torch.exp(curr_lse - max_lse)
    exp_sum = exp_context + exp_curr

    w_context = (exp_context / exp_sum).unsqueeze(-1)
    w_curr = (exp_curr / exp_sum).unsqueeze(-1)

    merged_out = context_out * w_context + curr_out * w_curr
    merged_lse = max_lse + torch.log(exp_sum)
    return merged_out.to(context_out.dtype), merged_lse


def build_pcp_batched_metadata(
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> PcpBatchedMetadata:
    """Builds PCP metadata for the 2-phase batched RPA kernel."""
    num_reqs = len(seq_lens)
    two_p = 2 * cp_size

    # Extend tokens per request
    extend_lens = (seq_lens - prefix_lens).cpu().numpy()
    prefix_lens_np = prefix_lens.cpu().numpy()

    # Determine chunk size C across requests
    # In uniform chunking, C is the chunk length per zigzag block
    c_sizes = [max(1, int(l) // two_p) for l in extend_lens]
    c_per_req = c_sizes[0] if len(c_sizes) > 0 else 1

    # 1. pcp_cu_q_lens: 2 blocks per request on this rank
    # Head chunk (r) and tail chunk (2P - 1 - r)
    pcp_cu = [0]
    q_pos_offsets = []
    pcp_write = []
    run = 0

    for i in range(num_reqs):
        c_len = c_sizes[i]
        # Head chunk: rank r owns chunk r, starting at offset r * c_len
        toff_head = prefix_lens_np[i] + cp_rank * c_len
        run += c_len
        pcp_cu.append(run)
        q_pos_offsets.append(toff_head)
        pcp_write.append(False)  # Head chunk does not commit cache

        # Tail chunk: rank r owns chunk (2P - 1 - r), starting at offset (2P - 1 - r) * c_len
        toff_tail = prefix_lens_np[i] + (two_p - 1 - cp_rank) * c_len
        run += c_len
        pcp_cu.append(run)
        q_pos_offsets.append(toff_tail)
        pcp_write.append(True)  # Tail chunk commits cache

    # 2. new_kv_page_indices (pcp_table):
    # Maps all-gathered rank-major order to sequence order
    req_pages = max(1, (c_per_req + page_size - 1) // page_size)
    local_len = 2 * c_per_req * num_reqs
    table = np.zeros((2 * num_reqs, two_p * req_pages), dtype=np.int32)
    for i in range(num_reqs):
        for c in range(two_p):
            owner_rank = c if c < cp_size else two_p - 1 - c
            chunk_in_rank = 0 if c < cp_size else c_per_req
            base = (
                owner_rank * local_len + 2 * c_per_req * i + chunk_in_rank
            ) // page_size
            cols = slice(c * req_pages, (c + 1) * req_pages)
            table[2 * i, cols] = base + np.arange(req_pages)
            table[2 * i + 1, cols] = base + np.arange(req_pages)

    pcp_cu_t = torch.tensor(pcp_cu, dtype=torch.int32, device=device)
    q_pos_t = torch.tensor(q_pos_offsets, dtype=torch.int32, device=device)
    pcp_write_t = torch.tensor(pcp_write, dtype=torch.bool, device=device)
    new_kv_pages_t = torch.tensor(table.reshape(-1), dtype=torch.int32, device=device)
    kv_new_lens_t = torch.tensor(
        [c_per_req * two_p] * (2 * num_reqs), dtype=torch.int32, device=device
    )

    # Cache phase: page-striped page table
    pages_per_seq = page_indices.shape[0] // num_reqs
    pages_2d = page_indices.view(num_reqs, pages_per_seq)
    owned_pages = pages_2d[:, cp_rank::cp_size]
    pad_len = pages_per_seq - owned_pages.shape[1]
    if pad_len > 0:
        striped_page_indices = torch.nn.functional.pad(
            owned_pages, (0, pad_len), value=0
        ).reshape(-1)
    else:
        striped_page_indices = owned_pages.reshape(-1)

    total_cache_pages = int(torch.sum(prefix_lens).item()) // page_size
    pcp_cache_lens_t = prefix_lens.to(dtype=torch.int32, device=device)

    return PcpBatchedMetadata(
        pcp_cu_q_lens=pcp_cu_t,
        q_positions=q_pos_t,
        new_kv_page_indices=new_kv_pages_t,
        update_kv_cache=pcp_write_t,
        kv_new_lens=kv_new_lens_t,
        striped_page_indices=striped_page_indices,
        cache_pages=total_cache_pages,
        num_reqs=num_reqs,
        pcp_cache_lens=pcp_cache_lens_t,
    )


def batched_rpa_pcp_forward(
    cp_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    layer: "RadixAttention",
    forward_batch: "ForwardBatch",
    pcp_meta: PcpBatchedMetadata,
    base_meta,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> torch.Tensor:
    """Executes the two-phase PCP forward pass on Batched RPA.

    1. Cache Phase: Attends sharded KV cache via ICI Ring Attention (CACHE_ONLY).
    2. Current Phase: All-gathers current tokens and attends causally (NEW_TOKENS_ONLY).
    3. Merges softmax states using Log-Sum-Exp.
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    device = q.device
    cp_rank_tensor = ensure_cp_rank_tensor(device, cp_rank)

    # 1. Cache Phase (Ring Attention over ICI)
    context_out = None
    context_lse = None
    if pcp_meta.cache_pages > 0:
        cache_update_kv = torch.zeros(
            base_meta.kv_lens.shape[0], dtype=torch.int32, device=device
        )
        context_out, context_lse = cp_op(
            q,
            k,
            v,
            kv_cache,
            pcp_meta.pcp_cache_lens,
            pcp_meta.striped_page_indices,
            base_meta.cu_q_lens,
            base_meta.distribution,
            pcp_meta.q_positions,
            pcp_meta.kv_new_lens,
            cp_rank_tensor,
            base_meta.page_indices,
            cache_update_kv,
            float(layer.scaling),
            None,
            layer.logit_cap if getattr(layer, "logit_cap", 0) else None,
            k_scale,
            v_scale,
            1,  # CACHE_ONLY
            cp_size,
            "pcp",
            True,
        )

    # 2. Current Phase (All-Gather current chunk tokens)
    # Gather keys and values across the PCP group
    k_curr = get_parallel().attn_cp_group.all_gather(k, dim=0)
    v_curr = get_parallel().attn_cp_group.all_gather(v, dim=0)

    # For the 2-chunk zigzag schedule, each request consists of 2 virtual sequences
    # (head chunk and tail chunk) on this rank.
    num_reqs = base_meta.kv_lens.shape[0]
    curr_kv_lens = pcp_meta.pcp_cache_lens.repeat_interleave(2)
    pages_per_seq = base_meta.page_indices.shape[0] // num_reqs
    pages_2d = base_meta.page_indices.view(num_reqs, pages_per_seq)
    owned_pages = pages_2d[:, cp_rank::cp_size]
    pad_len = pages_per_seq - owned_pages.shape[1]
    if pad_len > 0:
        striped_curr_pages = torch.nn.functional.pad(owned_pages, (0, pad_len), value=0)
    else:
        striped_curr_pages = owned_pages
    curr_page_indices = striped_curr_pages.repeat_interleave(2, dim=0).reshape(-1)
    curr_distribution = base_meta.distribution * 2

    curr_update_kv = pcp_meta.update_kv_cache.to(dtype=torch.int32, device=device)
    curr_out, curr_lse = cp_op(
        q,
        k_curr,
        v_curr,
        kv_cache,
        curr_kv_lens,
        curr_page_indices,
        pcp_meta.pcp_cu_q_lens,
        curr_distribution,
        pcp_meta.q_positions,
        pcp_meta.kv_new_lens,
        cp_rank_tensor,
        pcp_meta.new_kv_page_indices,
        curr_update_kv,
        float(layer.scaling),
        None,
        layer.logit_cap if getattr(layer, "logit_cap", 0) else None,
        k_scale,
        v_scale,
        2,  # NEW_TOKENS_ONLY
        cp_size,
        None,
        True,
    )

    # 3. Softmax State Merge
    if context_out is not None and context_lse is not None:
        out, _ = merge_attn_states(context_out, context_lse, curr_out, curr_lse)
        return out
    return curr_out


def batched_rpa_dcp_forward(
    cp_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    layer: "RadixAttention",
    forward_batch: "ForwardBatch",
    base_meta,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> torch.Tensor:
    """Executes Decode Context Parallelism (DCP) on Batched RPA.

    Each CP rank owns a page-striped portion of the KV cache:
    1. Cache Phase (CACHE_ONLY):
       Attends query against this rank's local striped pages -> (context_out, context_lse).
    2. Current Phase (NEW_TOKENS_ONLY):
       Attends query against the new token and writes it to the owning CP rank
       -> (curr_out, curr_lse).
    3. Multi-Rank Softmax Merge:
       All-gathers context_lse, all-reduces scaled context_out, and merges with curr_out.
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    cp_group = get_parallel().attn_cp_group
    device = q.device
    cp_rank_tensor = ensure_cp_rank_tensor(device, cp_rank)

    num_reqs = base_meta.kv_lens.shape[0]
    pages_per_seq = base_meta.page_indices.shape[0] // num_reqs
    pages_2d = base_meta.page_indices.view(num_reqs, pages_per_seq)

    # Local striped pages for this rank:
    owned_pages = pages_2d[:, cp_rank::cp_size]
    pad_len = pages_per_seq - owned_pages.shape[1]
    if pad_len > 0:
        striped_page_indices = torch.nn.functional.pad(
            owned_pages, (0, pad_len), value=0
        ).reshape(-1)
    else:
        striped_page_indices = owned_pages.reshape(-1)

    seq_lens = base_meta.kv_lens
    q_positions = seq_lens - 1
    kv_new_lens = torch.ones_like(seq_lens)

    # 1. Cache Phase (CACHE_ONLY)
    cache_update_kv = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    context_out, context_lse = cp_op(
        q,
        k,
        v,
        kv_cache,
        base_meta.kv_lens,
        striped_page_indices,
        base_meta.cu_q_lens,
        base_meta.distribution,
        q_positions,
        kv_new_lens,
        cp_rank_tensor,
        None,
        cache_update_kv,
        float(layer.scaling),
        None,
        layer.logit_cap if getattr(layer, "logit_cap", 0) else None,
        k_scale,
        v_scale,
        1,  # CACHE_ONLY
        cp_size,
        None,  # no ring in decode
        True,
    )

    # 2. Current Phase (NEW_TOKENS_ONLY)
    curr_update_kv = torch.ones(num_reqs, dtype=torch.int32, device=device)
    curr_out, curr_lse = cp_op(
        q,
        k,
        v,
        kv_cache,
        base_meta.kv_lens,
        striped_page_indices,
        base_meta.cu_q_lens,
        base_meta.distribution,
        q_positions,
        kv_new_lens,
        cp_rank_tensor,
        None,
        curr_update_kv,
        float(layer.scaling),
        None,
        layer.logit_cap if getattr(layer, "logit_cap", 0) else None,
        k_scale,
        v_scale,
        2,  # NEW_TOKENS_ONLY
        cp_size,
        None,
        True,
    )

    # 3. DCP State Merge
    all_context_lse = cp_group.all_gather(context_lse.to(torch.float32), dim=0)
    all_context_lse_3d = all_context_lse.view(cp_size, -1, *context_lse.shape[1:])

    stacked_lses = torch.cat(
        [curr_lse.to(torch.float32).unsqueeze(0), all_context_lse_3d], dim=0
    )
    global_max_lse = torch.amax(stacked_lses, dim=0)
    global_max_lse_safe = torch.where(
        torch.isinf(global_max_lse), torch.zeros_like(global_max_lse), global_max_lse
    )

    exp_all = torch.exp(stacked_lses - global_max_lse_safe.unsqueeze(0))
    sum_exp = torch.sum(exp_all, dim=0)
    sum_exp_safe = torch.where(sum_exp == 0.0, torch.ones_like(sum_exp), sum_exp)

    # Mask -inf LSE and sanitize outputs to prevent 0 * NaN propagation
    context_out_safe = torch.nan_to_num(context_out.to(torch.float32), nan=0.0)
    curr_out_safe = torch.nan_to_num(curr_out.to(torch.float32), nan=0.0)

    scale_context = torch.where(
        torch.isinf(context_lse),
        torch.zeros_like(context_lse),
        torch.exp(context_lse.to(torch.float32) - global_max_lse_safe),
    ).unsqueeze(-1)
    scaled_context_out = context_out_safe * scale_context
    sum_scaled_context = cp_group.all_reduce(scaled_context_out)

    scale_curr = torch.where(
        torch.isinf(curr_lse),
        torch.zeros_like(curr_lse),
        torch.exp(curr_lse.to(torch.float32) - global_max_lse_safe),
    ).unsqueeze(-1)
    scaled_curr_out = curr_out_safe * scale_curr

    total_out = sum_scaled_context + scaled_curr_out
    merged_out = total_out / sum_exp_safe.unsqueeze(-1)
    return merged_out.to(q.dtype)
