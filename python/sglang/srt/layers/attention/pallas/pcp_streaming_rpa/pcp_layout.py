# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Google LLC
#
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
"""PCP rank-major token layout helpers."""

import numpy as np


def pcp_query_start_offsets(
    num_scheduled_tokens_per_req: np.ndarray,
    token_start_offsets_per_req: list[int] | np.ndarray | None,
) -> np.ndarray:
    if token_start_offsets_per_req is None:
        return np.zeros(num_scheduled_tokens_per_req.size, dtype=np.int64)
    starts = np.asarray(token_start_offsets_per_req, dtype=np.int64)
    if starts.size != num_scheduled_tokens_per_req.size:
        raise ValueError(
            "token_start_offsets_per_req must have the same "
            "number of entries as num_scheduled_tokens_per_req."
        )
    if np.any(starts < 0):
        raise ValueError("token_start_offsets_per_req must be non-negative.")
    return starts


def pcp_query_chunk_ranges(
    q_len: int, q_global_base: int, pcp_rank: int, pcp_size: int, interleave_size: int
):
    if q_len <= 0:
        return
    cycle = pcp_size * interleave_size
    q_end = q_global_base + q_len
    chunk_start = (q_global_base // cycle) * cycle + pcp_rank * interleave_size
    if chunk_start + interleave_size <= q_global_base:
        chunk_start += cycle
    while chunk_start < q_end:
        overlap_start = max(chunk_start, q_global_base)
        overlap_end = min(chunk_start + interleave_size, q_end)
        if overlap_end > overlap_start:
            yield overlap_start, overlap_end
        chunk_start += cycle


def _assigned_rank_per_source_token(
    num_scheduled_tokens_per_req: np.ndarray,
    starts: np.ndarray,
    pcp_size: int,
    interleave_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (assigned_rank, global_pos) for each source token, vectorized.

    Vectorized replacement for the ``for req -> for rank -> for chunk`` triple
    loop. For source index ``i`` in the concatenated request-major batch:
      req_idx = np.searchsorted(cumsum, i, side='right')
      local_pos = i - req_starts[req_idx]
      global_pos = q_global_base[req_idx] + local_pos
      assigned_rank = (global_pos // interleave_size) % pcp_size

    The naive triple loop visits tokens in EXACTLY source order within each
    rank (chunks yielded in increasing chunk_start per rank per req, and
    within a chunk, tokens are contiguous). So np.argsort(assigned_rank,
    kind='stable') recovers the same per-rank source ordering the naive loop
    produced, letting the vectorized path reconstruct byte-identical
    token_order / inverse_order.
    """
    total = int(num_scheduled_tokens_per_req.sum())
    if total == 0:
        empty_i64 = np.zeros(0, dtype=np.int64)
        return empty_i64, empty_i64
    req_starts = np.concatenate(
        ([0], np.cumsum(num_scheduled_tokens_per_req[:-1], dtype=np.int64))
    )
    src_idx = np.arange(total, dtype=np.int64)
    # Which request each source token belongs to.
    req_idx = np.searchsorted(
        np.cumsum(num_scheduled_tokens_per_req, dtype=np.int64),
        src_idx,
        side="right",
    )
    local_pos = src_idx - req_starts[req_idx]
    global_pos = starts[req_idx] + local_pos
    assigned_rank = (global_pos // interleave_size) % pcp_size
    return assigned_rank.astype(np.int64), global_pos


def pcp_local_token_counts(
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return local token counts for each PCP rank under chunk interleave."""
    if pcp_size <= 1:
        return np.array([int(np.sum(num_scheduled_tokens_per_req))], dtype=np.int32)
    if interleave_size <= 0:
        raise ValueError("cp_kv_cache_interleave_size must be positive.")

    tokens = np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
    if tokens.size == 0:
        return np.zeros(pcp_size, dtype=np.int32)
    starts = pcp_query_start_offsets(tokens, token_start_offsets_per_req)
    assigned_rank, _ = _assigned_rank_per_source_token(
        tokens, starts, pcp_size, interleave_size
    )
    counts = np.bincount(assigned_rank, minlength=pcp_size).astype(np.int32)
    return counts


def build_pcp_rank_major_token_order(
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build token reorder indices for rank-major PCP chunk packing.

    Vectorized: replaces the ``for req -> for rank -> for chunk`` triple loop
    with a single-pass computation over source-token indices. Byte-identical
    to the previous naive implementation on all validly-shaped inputs (see
    ``test_pcp_layout_vectorization_parity_unit.py``). Measured cost drops
    from ~7.7 ms per 4096-token chunk to ~0.2 ms — ~40x speedup — which
    compounds across chunked-prefill forwards.

    Returns:
        token_order: [padded_num_tokens] source indices in original per-DP
            request-major token order. Padding entries are -1.
        inverse_order: [total_num_tokens] destination indices in packed order.
    """
    if pcp_size <= 1:
        total = int(np.sum(num_scheduled_tokens_per_req))
        token_order = np.full(padded_num_tokens, -1, dtype=np.int64)
        token_order[:total] = np.arange(total, dtype=np.int64)
        return token_order, np.arange(total, dtype=np.int64)
    if interleave_size <= 0:
        raise ValueError("cp_kv_cache_interleave_size must be positive.")
    if padded_num_tokens % pcp_size != 0:
        raise ValueError(f"{padded_num_tokens=} must be divisible by {pcp_size=}.")

    local_padded_num_tokens = padded_num_tokens // pcp_size
    token_order = np.full(padded_num_tokens, -1, dtype=np.int64)
    tokens = np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
    total_num_tokens = int(tokens.sum())
    inverse_order = np.full(total_num_tokens, -1, dtype=np.int64)

    if total_num_tokens == 0:
        return token_order, inverse_order

    starts = pcp_query_start_offsets(tokens, token_start_offsets_per_req)
    assigned_rank, _ = _assigned_rank_per_source_token(
        tokens, starts, pcp_size, interleave_size
    )

    # Per-rank source-token ordering: the naive triple loop visits source
    # indices in increasing order within each rank (chunks yielded in
    # ascending chunk_start per rank per request). ``np.argsort`` with
    # ``kind='stable'`` reproduces that ordering exactly.
    src_by_rank_order = np.argsort(assigned_rank, kind="stable")
    counts = np.bincount(assigned_rank, minlength=pcp_size)

    # Bounds check: each rank's per-rank token count must fit within its
    # padded slot allocation. The naive loop raised on overflow in the same
    # position; we hoist the check outside the loop.
    if counts.max() > local_padded_num_tokens:
        raise ValueError("PCP local token count exceeds padded local capacity.")

    # Write per-rank blocks into token_order + inverse_order.
    per_rank_starts = np.arange(pcp_size, dtype=np.int64) * local_padded_num_tokens
    cursor = 0
    for rank in range(pcp_size):
        n = int(counts[rank])
        if n == 0:
            continue
        src_slice = src_by_rank_order[cursor : cursor + n]
        dst_start = int(per_rank_starts[rank])
        dst_end = dst_start + n
        token_order[dst_start:dst_end] = src_slice
        inverse_order[src_slice] = np.arange(dst_start, dst_end, dtype=np.int64)
        cursor += n

    if total_num_tokens and np.any(inverse_order < 0):
        raise ValueError("PCP token packing did not cover all tokens.")
    return token_order, inverse_order


def apply_pcp_rank_major_token_order(
    input_ids_cpu: np.ndarray,
    positions_cpu: np.ndarray,
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    mrope_positions_cpu: np.ndarray | None = None,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Reorder per-DP token arrays into PCP rank-major chunk order."""
    total_num_tokens = int(np.sum(num_scheduled_tokens_per_req))
    token_order, inverse_order = build_pcp_rank_major_token_order(
        num_scheduled_tokens_per_req,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_start_offsets_per_req=token_start_offsets_per_req,
    )
    valid = token_order >= 0
    original_input_ids = input_ids_cpu[:total_num_tokens].copy()
    original_positions = positions_cpu[:total_num_tokens].copy()
    input_ids_cpu[:] = 0
    positions_cpu[:] = 0
    input_ids_cpu[valid] = original_input_ids[token_order[valid]]
    positions_cpu[valid] = original_positions[token_order[valid]]
    if mrope_positions_cpu is not None:
        original_mrope = mrope_positions_cpu[:, :total_num_tokens].copy()
        mrope_positions_cpu[:, :] = 0
        mrope_positions_cpu[:, valid] = original_mrope[:, token_order[valid]]
    return inverse_order


def build_pcp_logits_indices(
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    token_offset: int = 0,
    token_start_offsets_per_req: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return global packed indices for each request's last query token."""
    _, inverse_order = build_pcp_rank_major_token_order(
        num_scheduled_tokens_per_req,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_start_offsets_per_req=token_start_offsets_per_req,
    )
    if len(num_scheduled_tokens_per_req) == 0:
        return np.empty((0,), dtype=np.int64)
    local_request_ends = np.cumsum(num_scheduled_tokens_per_req, dtype=np.int64) - 1
    return inverse_order[local_request_ends] + token_offset
