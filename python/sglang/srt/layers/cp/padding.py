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

"""Padding helpers shared by context-parallel strategies."""

from typing import Any

import torch

from sglang.srt.runtime_context import get_parallel


def get_cp_padding_align_size() -> int:
    """Return the token-count alignment required by the active CP strategy."""
    from sglang.srt.layers.cp.base import is_zigzag

    attn_cp_size = get_parallel().attn_cp_size
    if is_zigzag():
        return attn_cp_size * 2
    return attn_cp_size


def pad_logical_token_to_physical(metadata: Any) -> None:
    """Align each CP rank's physical token count for CP collectives."""
    logical_tokens = list(metadata.per_rank_actual_token)
    align_size = get_cp_padding_align_size()
    physical_rank_len = (
        (max(logical_tokens) + align_size - 1) // align_size * align_size
    )
    metadata.per_rank_logical_token = logical_tokens
    metadata.per_rank_actual_token = [physical_rank_len] * len(logical_tokens)
    metadata.max_rank_len = [physical_rank_len] * len(logical_tokens)

    if hasattr(metadata, "perm") and metadata.perm is not None:
        stride = physical_rank_len
        metadata.stride = stride
        parallel = get_parallel()
        cp_rank = parallel.attn_cp_rank
        cp_size = parallel.attn_cp_size
        gathered_to_rk = []
        for r, n in enumerate(metadata.per_rank_logical_token):
            for k in range(n):
                gathered_to_rk.append((r, k))
        direct_perm = []
        gather_perm = []
        for p in metadata.perm:
            r, k = gathered_to_rk[p]
            slot = (cp_rank - r) % cp_size
            direct_perm.append(slot * stride + k)
            gather_perm.append(r * stride + k)
        device = metadata.reorder_index_tensor.device
        metadata.direct_reorder_index = torch.tensor(
            direct_perm, device=device, dtype=torch.int32
        )
        metadata.gather_reorder_index = torch.tensor(
            gather_perm, device=device, dtype=torch.int32
        )

        if hasattr(metadata, "local_rows") and metadata.local_rows is not None:
            cur_rows = metadata.local_rows
            if cur_rows.shape[0] < stride:
                pad_size = stride - cur_rows.shape[0]
                metadata.local_rows = torch.nn.functional.pad(
                    cur_rows, (0, pad_size), value=0
                )


def pad_local_rows(x: torch.Tensor, metadata: Any, dim: int) -> torch.Tensor:
    """Pad a local CP tensor from its logical length to its physical length."""
    if (
        metadata.per_rank_logical_token is None
        or metadata.per_rank_logical_token == metadata.per_rank_actual_token
    ):
        return x

    target_len = metadata.per_rank_actual_token[0]
    pad_size = target_len - x.shape[dim]
    assert pad_size >= 0
    if pad_size == 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
