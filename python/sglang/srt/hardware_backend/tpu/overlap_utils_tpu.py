"""Fused TPU stores for the overlap FutureMap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_batch_padding import (
    pad_rows,
    physical_batch_size,
)
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_future_map_store_args,
    make_future_map_store_hidden_args,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload


def _future_map_store_one(
    buf: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    values = values[: indices.shape[0]].to(buf.dtype)
    return torch.index_copy(buf, 0, indices.to(torch.int64), values)


@compiled_op(warmup=(make_future_map_store_args, make_future_map_store_hidden_args))
def fused_future_map_store_tpu(
    indices: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    bonus_tokens: torch.Tensor,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states_buf: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
):
    assert (hidden_states_buf is None) == (hidden_states is None)
    return (
        _future_map_store_one(output_tokens_buf, indices, bonus_tokens),
        _future_map_store_one(topk_p_buf, indices, topk_p),
        _future_map_store_one(topk_index_buf, indices, topk_index),
        (
            _future_map_store_one(hidden_states_buf, indices, hidden_states)
            if hidden_states_buf is not None
            else None
        ),
    )


def store_future_map_tpu(
    *, future_map: FutureMap, indices: torch.Tensor, payload: RelayPayload
):
    logical_bs = indices.shape[0]
    if logical_bs == 0:
        return (
            future_map.output_tokens_buf,
            future_map.topk_p_buf,
            future_map.topk_index_buf,
            future_map.hidden_states_buf,
        )
    assert payload.bonus_tokens.shape[0] == logical_bs
    assert payload.topk_p.shape[0] == logical_bs
    assert payload.topk_index.shape[0] == logical_bs
    physical_bs = physical_batch_size(
        logical_bs, tuple(future_map._tpu_precompiled_batch_sizes)
    )
    indices = pad_rows(indices.to(torch.int64), physical_bs, future_map.req_pool_size)
    bonus_tokens = pad_rows(
        payload.bonus_tokens.to(future_map.output_tokens_buf.dtype),
        physical_bs,
    )
    topk_p = pad_rows(payload.topk_p.to(future_map.topk_p_buf.dtype), physical_bs)
    topk_index = pad_rows(
        payload.topk_index.to(future_map.topk_index_buf.dtype), physical_bs
    )
    hidden_states_buf = None
    hidden_states = None
    if future_map.need_hidden_states:
        assert payload.hidden_states.shape[0] == logical_bs
        hidden_states_buf = future_map.hidden_states_buf
        hidden_states = pad_rows(
            payload.hidden_states.to(hidden_states_buf.dtype), physical_bs
        )
    return fused_future_map_store_tpu(
        indices,
        future_map.output_tokens_buf,
        future_map.topk_p_buf,
        future_map.topk_index_buf,
        bonus_tokens,
        topk_p,
        topk_index,
        hidden_states_buf,
        hidden_states,
    )
