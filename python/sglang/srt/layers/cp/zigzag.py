# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Zigzag context parallel strategy shell.

For ``cp_size = 4``, each sequence is split into ``2 * cp_size`` blocks. Each
rank owns one early block and one late block:

    cp0: block0, block7
    cp1: block1, block6
    cp2: block2, block5
    cp3: block3, block4

After all-gather, the blocks are reranged back to their original order:

    block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4
      -> block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.cp.padding import pad_local_rows
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_device, get_parallel


@dataclass
class ZigzagContextParallelMetadata(BaseContextParallelMetadata):
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    split_list: Optional[List[int]] = None
    zigzag_index: Optional[List[int]] = None
    cp_reverse_index: Optional[List[int]] = None
    reverse_split_len: Optional[List[int]] = None
    reorder_index_tensor: Optional[Any] = None
    perm: Optional[List[int]] = None
    direct_reorder_index: Optional[Any] = None
    gather_reorder_index: Optional[Any] = None
    # Host (numpy int32) owner rank / local row of each natural-order token;
    # `padding.gather_reorder_index_for` re-strides them for a gather whose
    # per-rank row count differs from `stride`.
    gather_rank: Optional[Any] = None
    gather_local: Optional[Any] = None
    local_rows: Optional[Any] = None
    cp_last_rows: Optional[Any] = None
    cp_last_owner_mask: Optional[Any] = None
    stride: int = 0

    # Per-rank aggregate lists have length cp_size.
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    per_rank_logical_token: Optional[List[int]] = None

    # Per-sequence FlashAttention tensors (shape [bs] or [bs + 1]).
    kv_len_prev_tensor: Optional[Any] = None
    kv_len_next_tensor: Optional[Any] = None
    cu_seqlens_kv_prev_tensor: Optional[Any] = None
    cu_seqlens_kv_next_tensor: Optional[Any] = None
    actual_seq_q_prev_tensor: Optional[Any] = None
    actual_seq_q_next_tensor: Optional[Any] = None
    cu_seqlens_q_prev_tensor: Optional[Any] = None
    cu_seqlens_q_next_tensor: Optional[Any] = None

    # Combined prev-then-next TRT-LLM geometry (shape [2 * bs] or [2 * bs + 1]).
    actual_seq_q_combined_tensor: Optional[Any] = None
    kv_len_combined_tensor: Optional[Any] = None
    cu_seqlens_q_combined_tensor: Optional[Any] = None
    cu_seqlens_kv_combined_tensor: Optional[Any] = None

    # Scalars derived from the per-sequence lists above.
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0
    max_seqlen_q_combined: int = 0

    # Per-sequence CPU lists, useful for indexers and diagnostics.
    kv_len_prev_list: Optional[List[int]] = None
    kv_len_next_list: Optional[List[int]] = None
    actual_seq_q_prev_list: Optional[List[int]] = None
    actual_seq_q_next_list: Optional[List[int]] = None


ContextParallelMetadata = ZigzagContextParallelMetadata


def _concat_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``range(starts[i], starts[i] + lengths[i])`` for all i."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros((0,), dtype=np.int64)
    ends = np.cumsum(lengths)
    begins = ends - lengths
    # offset of each output element to the start of its own range
    idx = np.arange(total, dtype=np.int64)
    which = np.searchsorted(ends, idx, side="right")
    return np.asarray(starts, dtype=np.int64)[which] + (idx - begins[which])


class ZigzagCPStrategy(ContextParallelStrategy):
    name = "zigzag"
    kind = ContextParallelStrategyKind.ZIGZAG

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.cp_size <= 1:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and not forward_mode.is_context_parallel_extend():
            return False

        min_tokens = self.cp_size * 2
        if num_tokens < min_tokens:
            return False

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            return True
        non_zero_lens = [int(length) for length in extend_lens if int(length) > 0]
        if not non_zero_lens:
            return False
        return all(length >= min_tokens for length in non_zero_lens)

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> ZigzagContextParallelMetadata:
        if extend_seqs_len is None:
            extend_seqs_len = seqs_len or [num_tokens]
        extend_seqs_len = [int(x) for x in extend_seqs_len]

        pad_len = int(num_tokens) - sum(extend_seqs_len)
        if pad_len > 0:
            extend_seqs_len[-1] += pad_len
            if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
                seqs_len = list(seqs_len)
                seqs_len[-1] += pad_len

        bs = len(extend_seqs_len)
        cp_segment_num = self.cp_size * 2
        if seqs_len is not None and len(seqs_len) == bs:
            prefix_offsets = [
                max(int(seqs_len[i]) - extend_seqs_len[i], 0) for i in range(bs)
            ]
        else:
            prefix_offsets = [0] * bs

        # TODO: move these per-request layout/index computations to a Triton
        # kernel if Python-side metadata construction becomes a bottleneck.
        per_seq_block_sizes: List[List[int]] = []
        split_list: List[int] = []
        for length in extend_seqs_len:
            base = length // cp_segment_num
            rem = length % cp_segment_num
            block_sizes = [
                base + 1 if block_id < rem else base
                for block_id in range(cp_segment_num)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        per_rank_actual_token = []
        for rank in range(self.cp_size):
            per_rank_actual_token.append(
                sum(
                    block_sizes[rank] + block_sizes[cp_segment_num - 1 - rank]
                    for block_sizes in per_seq_block_sizes
                )
            )
        max_rank_len = [max(per_rank_actual_token)] * self.cp_size

        cp_rank = self.cp_rank
        zigzag_index: List[int] = []
        for batch_id in range(bs):
            zigzag_index.append(batch_id * cp_segment_num + cp_rank)
            zigzag_index.append(
                batch_id * cp_segment_num + (cp_segment_num - cp_rank - 1)
            )

        reverse_split_len: List[int] = []
        gathered_chunk_ids: List[int] = []
        for r in range(self.cp_size):
            for batch_id in range(bs):
                reverse_split_len.append(per_seq_block_sizes[batch_id][r])
                gathered_chunk_ids.append(batch_id * cp_segment_num + r)
                reverse_split_len.append(
                    per_seq_block_sizes[batch_id][cp_segment_num - 1 - r]
                )
                gathered_chunk_ids.append(
                    batch_id * cp_segment_num + (cp_segment_num - 1 - r)
                )

        target_chunk_ids = list(range(bs * cp_segment_num))
        cp_reverse_index = [gathered_chunk_ids.index(c) for c in target_chunk_ids]

        kv_len_prev_list: List[int] = []
        kv_len_next_list: List[int] = []
        actual_seq_q_prev_list: List[int] = []
        actual_seq_q_next_list: List[int] = []
        for batch_id, block_sizes in enumerate(per_seq_block_sizes):
            kv_len_prev_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_rank + 1])
            )
            kv_len_next_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_segment_num - cp_rank])
            )
            actual_seq_q_prev_list.append(block_sizes[cp_rank])
            actual_seq_q_next_list.append(block_sizes[cp_segment_num - cp_rank - 1])

        try:
            device = torch.device(get_device().device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
        cu_next = [0] + list(accumulate(actual_seq_q_next_list))
        cu_kv_prev = [0] + list(accumulate(kv_len_prev_list))
        cu_kv_next = [0] + list(accumulate(kv_len_next_list))

        actual_seq_q_combined_list: List[int] = []
        kv_len_combined_list: List[int] = []
        for batch_id in range(bs):
            actual_seq_q_combined_list.append(actual_seq_q_prev_list[batch_id])
            actual_seq_q_combined_list.append(actual_seq_q_next_list[batch_id])
            kv_len_combined_list.append(kv_len_prev_list[batch_id])
            kv_len_combined_list.append(kv_len_next_list[batch_id])

        cu_q_combined = [0] + list(accumulate(actual_seq_q_combined_list))
        cu_kv_combined = [0] + list(accumulate(kv_len_combined_list))

        total_seq_lens = sum(extend_seqs_len)
        assert len(split_list) == bs * cp_segment_num
        assert sum(split_list) == total_seq_lens
        assert len(zigzag_index) == 2 * bs
        assert len(cp_reverse_index) == bs * cp_segment_num
        assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
        assert sum(per_rank_actual_token) == total_seq_lens

        # Host index construction, vectorized with numpy: every quantity below
        # is a concatenation of integer ranges, and at 16K tokens the Python
        # loops this replaces cost ~6 ms per prefill batch.
        rsl = np.asarray(reverse_split_len, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(rsl)])
        rev = np.asarray(cp_reverse_index, dtype=np.int64)
        perm = _concat_ranges(offsets[rev], rsl[rev])
        reorder_index_tensor = torch.from_numpy(perm.astype(np.int32)).to(device)

        stride = max(per_rank_actual_token) if per_rank_actual_token else int(num_tokens)
        rank_offsets = np.concatenate([[0], np.cumsum(np.asarray(per_rank_actual_token, dtype=np.int64))])
        gather_rank = np.searchsorted(rank_offsets, perm, side="right") - 1
        gather_local = perm - rank_offsets[gather_rank]
        slot = (cp_rank - gather_rank) % self.cp_size
        direct_reorder_index = torch.from_numpy(
            (slot * stride + gather_local).astype(np.int32)
        ).to(device)
        gather_reorder_index = torch.from_numpy(
            (gather_rank * stride + gather_local).astype(np.int32)
        ).to(device)

        sl = np.asarray(split_list, dtype=np.int64)
        chunk_offsets = np.concatenate([[0], np.cumsum(sl)])
        zz = np.asarray(zigzag_index, dtype=np.int64)
        local_indices = _concat_ranges(chunk_offsets[zz], sl[zz])
        n_local = len(local_indices)
        local_indices_padded = np.zeros((max(stride, n_local),), dtype=np.int32)
        local_indices_padded[:n_local] = local_indices
        local_rows = torch.from_numpy(local_indices_padded).to(device)

        # Last real token of each request -> local row (owner rank only).
        ext = np.asarray(extend_seqs_len, dtype=np.int64)
        last_tok = np.cumsum(ext) - 1
        # local_indices is sorted ascending per zigzag chunk but not globally;
        # map global token -> local row through an inverse table.
        inv = np.full((int(total_seq_lens) + 1,), -1, dtype=np.int64)
        inv[local_indices] = np.arange(n_local)
        loc = inv[np.where(ext > 0, last_tok, 0)]
        owner = (ext > 0) & (loc >= 0)
        cp_last_rows = torch.from_numpy(np.where(owner, loc, 0).astype(np.int32)).to(device)
        cp_last_owner_mask = torch.from_numpy(owner).to(device)

        return ZigzagContextParallelMetadata(
            split_list=split_list,
            zigzag_index=zigzag_index,
            cp_reverse_index=cp_reverse_index,
            reverse_split_len=reverse_split_len,
            reorder_index_tensor=reorder_index_tensor,
            perm=perm.tolist(),
            direct_reorder_index=direct_reorder_index,
            gather_reorder_index=gather_reorder_index,
            local_rows=local_rows,
            cp_last_rows=cp_last_rows,
            cp_last_owner_mask=cp_last_owner_mask,
            stride=stride,
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
            kv_len_prev_tensor=torch.tensor(
                kv_len_prev_list, device=device, dtype=torch.int32
            ),
            kv_len_next_tensor=torch.tensor(
                kv_len_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_prev_tensor=torch.tensor(
                cu_kv_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_next_tensor=torch.tensor(
                cu_kv_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_prev_tensor=torch.tensor(
                actual_seq_q_prev_list, device=device, dtype=torch.int32
            ),
            actual_seq_q_next_tensor=torch.tensor(
                actual_seq_q_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_prev_tensor=torch.tensor(
                cu_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_next_tensor=torch.tensor(
                cu_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_combined_tensor=torch.tensor(
                actual_seq_q_combined_list, device=device, dtype=torch.int32
            ),
            kv_len_combined_tensor=torch.tensor(
                kv_len_combined_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_combined_tensor=torch.tensor(
                cu_q_combined, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_combined_tensor=torch.tensor(
                cu_kv_combined, device=device, dtype=torch.int32
            ),
            total_q_prev_tokens=cu_prev[-1],
            total_q_next_tokens=cu_next[-1],
            max_seqlen_q_prev=(
                max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
            ),
            max_seqlen_q_next=(
                max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
            ),
            max_seqlen_q_combined=(
                max(actual_seq_q_combined_list) if actual_seq_q_combined_list else 0
            ),
            kv_len_prev_list=kv_len_prev_list,
            kv_len_next_list=kv_len_next_list,
            actual_seq_q_prev_list=actual_seq_q_prev_list,
            actual_seq_q_next_list=actual_seq_q_next_list,
            total_seq_lens=total_seq_lens,
            bs=bs,
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        x = x[: metadata.total_seq_lens]
        chunks = torch.split(x, metadata.split_list, dim=0)
        local_x = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=0)
        return pad_local_rows(local_x, metadata, dim=0)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        positions = positions[..., : metadata.total_seq_lens]
        chunks = torch.split(positions, metadata.split_list, dim=-1)
        local_positions = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=-1)
        return pad_local_rows(local_positions, metadata, dim=-1)

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        meta = forward_batch.attn_cp_metadata
        if x.device.type == "tpu" and (
            getattr(meta, "gather_reorder_index", None) is not None
            or getattr(meta, "direct_reorder_index", None) is not None
        ):
            from sglang.srt.hardware_backend.tpu.batched_rpa_cp import (
                cp_all_gather_rows,
            )
            minor = tuple(x.shape[1:])
            x2 = x.reshape(x.shape[0], -1).contiguous()
            out = cp_all_gather_rows(x2, forward_batch)
            return out.reshape(-1, *minor)

        gathered = self._all_gather_reorganized(x, forward_batch)
        reorder_index = getattr(meta, "reorder_index_tensor", None)
        if reorder_index is not None:
            if reorder_index.device != gathered.device:
                reorder_index = reorder_index.to(gathered.device)
            return gathered.index_select(0, reorder_index)
        chunks = torch.split(
            gathered, meta.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in meta.cp_reverse_index], dim=0
        )

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        meta = forward_batch.attn_cp_metadata
        if x.device.type == "tpu" and (
            getattr(meta, "gather_reorder_index", None) is not None
            or getattr(meta, "direct_reorder_index", None) is not None
        ):
            from sglang.srt.hardware_backend.tpu.batched_rpa_cp import (
                cp_all_gather_rows,
            )
            minor = tuple(x.shape[1:])
            x2 = x.reshape(x.shape[0], -1).contiguous()
            out = cp_all_gather_rows(x2, forward_batch)
            return out.reshape(-1, *minor)

        gathered = self._all_gather_reorganized(x, forward_batch)
        reorder_index = getattr(meta, "reorder_index_tensor", None)
        if reorder_index is not None:
            if reorder_index.device != gathered.device:
                reorder_index = reorder_index.to(gathered.device)
            return gathered.index_select(0, reorder_index)
        chunks = torch.split(
            gathered, meta.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in meta.cp_reverse_index], dim=0
        )

    def local_q_indices(self, num_tokens: int, forward_batch) -> Any:
        meta = forward_batch.attn_cp_metadata
        device = getattr(getattr(forward_batch, "input_ids", None), "device", None)
        if device is None and meta.cu_seqlens_q_prev_tensor is not None:
            device = meta.cu_seqlens_q_prev_tensor.device
        if device is None:
            device = torch.device("cpu")

        offsets = [0] + list(accumulate(meta.split_list))
        pieces = []
        for chunk_idx in meta.zigzag_index:
            start = offsets[chunk_idx]
            end = offsets[chunk_idx + 1]
            if end > start:
                pieces.append(torch.arange(start, end, device=device, dtype=torch.long))
        if not pieces:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.cat(pieces, dim=0)

    def get_supported_attention_backend(self):
        return [
            CPAttentionBackendKind.FLASH_ATTENTION,
            CPAttentionBackendKind.TRTLLM_MHA,
            CPAttentionBackendKind.PALLAS_RPA,
        ]

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        assert attention_backend in self.get_supported_attention_backend(), (
            f"{self.name} CP does not support {attention_backend=}"
        )

        meta = forward_batch.attn_cp_metadata
        q_prev = q[: meta.total_q_prev_tokens]
        logical_tokens = meta.total_q_prev_tokens + meta.total_q_next_tokens
        q_next = q[meta.total_q_prev_tokens : logical_tokens]

        prev_kwargs = {}
        next_kwargs = {}
        if attention_backend == CPAttentionBackendKind.TRTLLM_MHA:
            result = attn_fn(
                q[:logical_tokens],
                meta.cu_seqlens_q_combined_tensor,
                meta.kv_len_combined_tensor,
                meta.max_seqlen_q_combined,
                cu_seqlens_kv=meta.cu_seqlens_kv_combined_tensor,
                use_zigzag_page_table=True,
            )
        else:
            result_prev = attn_fn(
                q_prev,
                meta.cu_seqlens_q_prev_tensor,
                meta.kv_len_prev_tensor,
                meta.max_seqlen_q_prev,
                **prev_kwargs,
            )
            result_next = attn_fn(
                q_next,
                meta.cu_seqlens_q_next_tensor,
                meta.kv_len_next_tensor,
                meta.max_seqlen_q_next,
                **next_kwargs,
            )
            result = torch.cat([result_prev, result_next], dim=0)

        pad_size = q.shape[0] - logical_tokens
        assert pad_size >= 0
        if pad_size > 0:
            result = torch.cat(
                [result, result.new_zeros(pad_size, *result.shape[1:])], dim=0
            )
        return result

    def materialize_full_kv(
        self,
        forward_batch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if swa_loc is not None:
            swa_loc = swa_loc[: cache_loc.shape[0]]
        k_dim = k.shape[-1]
        v_dim = v.shape[-1]
        kv_cache = torch.cat([k, v], dim=-1).contiguous()
        key_cache_full, value_cache_full = self.gather_kv_cache(
            kv_cache, forward_batch
        ).split([k_dim, v_dim], dim=-1)
        key_cache_full = key_cache_full.contiguous()
        value_cache_full = value_cache_full.contiguous()
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc(cache_loc, swa_loc),
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )

    def materialize_full_mla_kv(
        self, forward_batch, layer: Any, k_nope: Any, k_rope: Any
    ) -> None:
        kv_lora_rank = k_nope.shape[-1]
        latent = torch.cat([k_nope, k_rope], dim=-1).contiguous()
        latent_full = self.gather_kv_cache(latent, forward_batch)
        get_token_to_kv_pool().set_mla_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            latent_full[..., :kv_lora_rank],
            latent_full[..., kv_lora_rank:],
        )

    def _all_gather_reorganized(self, x: torch.Tensor, forward_batch):
        meta = forward_batch.attn_cp_metadata
        per_rank_token = meta.per_rank_logical_token or meta.per_rank_actual_token
        max_len = max(per_rank_token)
        local_len = per_rank_token[self.cp_rank]
        assert x.shape[0] >= local_len
        if x.device.type == "tpu":
            return self._all_gather_reorganized_tpu(x, per_rank_token, max_len, local_len)
        x = x[:local_len]
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            padding = [0, 0] * (x.ndim - 1) + [0, pad_size]
            x = F.pad(x, padding, mode="constant", value=0)

        group = get_parallel().attn_cp_group
        ctx = (
            use_symmetric_memory(group, disabled=not is_allocation_symmetric())
            if x.is_cuda
            else nullcontext()
        )
        with ctx:
            gathered = torch.empty(
                max_len * self.cp_size,
                *x.shape[1:],
                device=x.device,
                dtype=x.dtype,
            )
        group.all_gather_into_tensor(gathered, x)

        chunks = torch.split(gathered, [max_len] * self.cp_size, dim=0)
        return torch.cat(
            [
                chunks[rank][:per_rank_len]
                for rank, per_rank_len in enumerate(per_rank_token)
            ],
            dim=0,
        )

    def _all_gather_reorganized_tpu(self, x, per_rank_token, max_len, local_len):
        """TPU: gather through the exported XLA collective (see
        `batched_rpa_cp.build_cp_all_gather_op`). The eager torch collective
        returned wrong rows for the remote ranks whenever the per-rank shard
        lengths differ (uneven zigzag chunks)."""
        from sglang.srt.hardware_backend.tpu.batched_rpa_cp import cp_all_gather_padded

        minor = tuple(x.shape[1:])
        x2 = x.reshape(x.shape[0], -1)
        chunks = cp_all_gather_padded(x2, local_len, max_len, self.cp_size)
        return torch.cat(
            [chunks[r][:n] for r, n in enumerate(per_rank_token)], dim=0
        ).reshape(-1, *minor)
