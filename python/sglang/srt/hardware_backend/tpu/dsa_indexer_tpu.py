# SPDX-License-Identifier: Apache-2.0
"""DSA lightning indexer on TPU (bf16, plain torch ops).

Same scores as the CUDA indexer, which ranks with FP8 keys after a Hadamard
rotation: the rotation is orthogonal, so bf16 keys need neither it nor scales,
and the uniform positive factors (n_heads**-0.5, softmax_scale) do not change
the ranking. For each query token t of request r:

    score[t, s] = sum_h relu(q[t, h] . k[r, s]) * w[t, h],   s < kv_len(r)

and the indexer returns the top ``index_topk`` positions s (int32, -1 padded).
Keys are written to the pool before scoring, so a token sees its own key.
Extend batches only store keys: prefill attention stays dense (phase 1).
"""

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _project(indexer: "Indexer", x, q_lora, positions):
    rope = indexer.rope_head_dim
    query, _ = indexer.wq_b(q_lora)
    query = query.view(-1, indexer.n_heads, indexer.head_dim)
    if indexer.use_dsa_indexer_fusion:
        # One bf16 GEMM for wk and weights_proj (the loader fuses the checkpoint).
        key, weights = indexer._fused_k_weights(x)
    else:
        key, _ = indexer.wk(x)
        weights, _ = indexer.weights_proj(x)
    key = indexer.k_norm(key)
    q_rope, k_rope = indexer.rotary_emb(
        positions, query[..., :rope].contiguous(), key[..., :rope].contiguous()
    )
    query = torch.cat([q_rope.view_as(query[..., :rope]), query[..., rope:]], dim=-1)
    key = torch.cat([k_rope.view_as(key[..., :rope]), key[..., rope:]], dim=-1)
    return query, key, weights.float()


# Pages the fused kernel streams per block; 16 x 128-token pages = 2048 keys.
_KV_PAGES_PER_BLOCK = 16

_STREAMINDEX_TOPK_OP: Optional[Callable] = None


def _get_streamindex_topk_op() -> Callable:
    """Fused score + top-k Pallas kernel over the paged index-K cache."""
    global _STREAMINDEX_TOPK_OP
    if _STREAMINDEX_TOPK_OP is not None:
        return _STREAMINDEX_TOPK_OP

    import jax

    from sglang.srt.hardware_backend.tpu.kernels.dsa_streamindex_topk import (
        streamindex_topk,
    )
    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        query: jax.Array,
        weights: jax.Array,
        index_k: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        k: int,
        page_size: int,
        topk_backend: str,
        page_topk: int,
    ) -> jax.Array:
        # [rows, D] page-major bf16 is the kernel's [pages, page/2, 2, D] layout.
        cache = index_k.reshape(-1, page_size // 2, 2, index_k.shape[-1])
        return streamindex_topk(
            query,
            weights,
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            k=k,
            compression_ratio=1,
            num_kv_pages_per_block=_KV_PAGES_PER_BLOCK,
            num_queries_per_block=1,
            topk_backend=topk_backend,
            page_topk=page_topk,
        )

    _STREAMINDEX_TOPK_OP = cpu_export_jax_op(
        "sglang_tpu_pallas::dsa_streamindex_topk", _wrapped
    )
    return _STREAMINDEX_TOPK_OP


def register_dsa_indexer_ops() -> None:
    """Register the kernel schema before full-graph tracing begins."""
    _get_streamindex_topk_op()


def decode_topk(
    query: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    slots: torch.Tensor,
    kv_lens: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Top-k positions for one query per request.

    query [B, H, D], weights [B, H] fp32, index_k [rows, D], slots [B, S] (the
    request's token slots, row s = position s), kv_lens [B]. Returns [B, k]
    int32 positions, -1 where fewer than k positions are valid.
    """
    from sglang.srt.hardware_backend.tpu.top_k import tpu_top_k

    keys = index_k[slots.reshape(-1).to(torch.int64)].view(
        slots.shape[0], slots.shape[1], -1
    )
    logits = torch.einsum("bhd,bsd->bhs", query.float(), keys.float())
    scores = (torch.relu(logits) * weights.unsqueeze(-1)).sum(dim=1)
    pos = torch.arange(slots.shape[1], device=slots.device).unsqueeze(0)
    valid = pos < kv_lens.unsqueeze(1)
    scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
    k = min(k, slots.shape[1])
    values, indices = tpu_top_k(scores, k)
    return torch.where(
        torch.isfinite(values),
        indices.to(torch.int32),
        torch.full_like(indices, -1, dtype=torch.int32),
    )


def indexer_forward_tpu(
    indexer: "Indexer",
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: "ForwardBatch",
    layer_id: int,
    return_indices: bool = True,
) -> Optional[torch.Tensor]:
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    pool = get_token_to_kv_pool()
    query, key, weights = _project(indexer, x, q_lora, positions)
    pool.set_index_k(layer_id, forward_batch.out_cache_loc, key)
    if not return_indices or not forward_batch.forward_mode.is_decode():
        return None
    backend = get_attn_backend()
    if envs.SGLANG_TPU_DSA_INDEXER_KERNEL.get():
        md = backend.forward_metadata
        return _get_streamindex_topk_op()(
            query,
            weights,
            pool.get_index_k_buffer(layer_id),
            md.kv_lens,
            md.page_indices,
            md.cu_q_lens,
            md.distribution,
            indexer.index_topk,
            backend.page_size,
            envs.SGLANG_TPU_DSA_TOPK_BACKEND.get(),
            envs.SGLANG_TPU_DSA_PAGE_TOPK.get(),
        )
    slots = backend.forward_metadata.dsa_token_slots
    return decode_topk(
        query,
        weights,
        pool.get_index_k_buffer(layer_id),
        slots,
        forward_batch.seq_lens,
        indexer.index_topk,
    )
