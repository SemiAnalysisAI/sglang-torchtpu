"""TPU helpers for EAGLE verify trees and draft extension."""

from typing import Optional

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_batch_padding import (
    pad_rows,
    physical_batch_size,
)
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_draft_extend_prepare_args,
    make_draft_extend_select_args,
)


def build_topk1_verify_chain_torch(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """Build the fixed top-k=1 EAGLE verify tensors with pure torch ops.

    The fused TPU draft loop calls this before returning so tree construction
    stays in the same compiled program. The ordinary path also uses it as its
    torch-native fallback.
    """
    bs = seq_lens.shape[0]
    num_verify_tokens = draft_tokens.shape[1] + 1
    device = seq_lens.device

    verify_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1)
    token_offsets = torch.arange(num_verify_tokens, device=device, dtype=torch.long)
    positions = seq_lens[:, None] + token_offsets[None, :]
    retrieve_index = torch.arange(
        bs * num_verify_tokens, device=device, dtype=torch.long
    ).view(bs, num_verify_tokens)
    next_token_row = torch.cat(
        (
            torch.arange(1, num_verify_tokens, device=device, dtype=torch.long),
            torch.full((1,), -1, device=device, dtype=torch.long),
        )
    )
    retrieve_next_token = next_token_row.expand(bs, -1).contiguous()
    retrieve_next_sibling = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    tree_mask = torch.empty((0,), dtype=torch.bool, device=device)
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        verify_tokens,
    )


def _draft_extend_select_argmax_impl(
    next_token_logits: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    select_index: torch.Tensor,
):
    selected_logits = next_token_logits[select_index]
    selected_hidden = hidden_states[select_index] if hidden_states is not None else None
    topk_index = torch.argmax(selected_logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    return selected_logits, selected_hidden, topk_p, topk_index


@compiled_op(warmup=make_draft_extend_select_args)
def fused_draft_extend_select_argmax_func(
    next_token_logits: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    select_index: torch.Tensor,
):
    """Select draft-extend rows and compute their greedy next-draft tokens."""
    return _draft_extend_select_argmax_impl(
        next_token_logits, hidden_states, select_index
    )


def draft_extend_select_argmax_func(
    next_token_logits: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    select_index: torch.Tensor,
    *,
    physical_batch_buckets: tuple[int, ...],
    num_draft_tokens: int,
):
    logical_bs = select_index.numel()
    if logical_bs == 0:
        return (
            next_token_logits[:0],
            hidden_states[:0] if hidden_states is not None else None,
            next_token_logits.new_empty((0, 1), dtype=torch.float32),
            select_index.new_empty((0, 1), dtype=torch.int64),
        )
    assert next_token_logits.shape[0] % num_draft_tokens == 0
    input_bs = next_token_logits.shape[0] // num_draft_tokens
    assert logical_bs <= input_bs
    physical_bs = physical_batch_size(input_bs, physical_batch_buckets)
    if hidden_states is not None:
        assert hidden_states.shape[0] == next_token_logits.shape[0]
        hidden_states = pad_rows(hidden_states, physical_bs * num_draft_tokens)
    outputs = fused_draft_extend_select_argmax_func(
        pad_rows(next_token_logits, physical_bs * num_draft_tokens),
        hidden_states,
        pad_rows(select_index.to(torch.int64), physical_bs),
    )
    return tuple(value[:logical_bs] if value is not None else None for value in outputs)


@compiled_op(warmup=make_draft_extend_prepare_args)
def fused_draft_extend_prepare_func(
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    next_token_ids: torch.Tensor,
    num_draft_tokens: int,
):
    batch_size = seq_lens.shape[0]
    request_offsets = (
        torch.arange(batch_size, dtype=torch.int64, device=seq_lens.device)
        * num_draft_tokens
    )
    token_offsets = torch.arange(
        num_draft_tokens, dtype=torch.int64, device=seq_lens.device
    )
    positions = (
        seq_lens.to(torch.int64).unsqueeze(1) + token_offsets.unsqueeze(0)
    ).reshape(-1)
    extend_start_loc = request_offsets.to(torch.int32)
    select_index = request_offsets + accept_lens.to(torch.int64) - 1
    return (
        accept_lens - 1,
        next_token_ids.to(torch.int64),
        positions,
        extend_start_loc,
        select_index,
    )


def build_topk1_verify_tree_tpu(
    *, bonus_tokens: torch.Tensor, draft_tokens: torch.Tensor, seq_lens: torch.Tensor
):
    tree_mask, positions, retrieve_index, next_token, next_sibling, verify_tokens = (
        build_topk1_verify_chain_torch(bonus_tokens, draft_tokens, seq_lens)
    )
    return (
        tree_mask,
        positions.flatten(),
        retrieve_index,
        next_token,
        next_sibling,
        verify_tokens.flatten(),
    )


def draft_extend_prepare_func(
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    next_token_ids: torch.Tensor,
    num_draft_tokens: int,
    *,
    physical_batch_buckets: tuple[int, ...],
):
    logical_bs = seq_lens.shape[0]
    assert accept_lens.shape == seq_lens.shape
    assert next_token_ids.shape == (logical_bs * num_draft_tokens,)
    if logical_bs == 0:
        return (
            accept_lens,
            next_token_ids.to(torch.int64),
            seq_lens.to(torch.int64),
            accept_lens.new_empty((0,), dtype=torch.int32),
            seq_lens.new_empty((0,), dtype=torch.int64),
        )
    physical_bs = physical_batch_size(logical_bs, physical_batch_buckets)
    num_correct_drafts, token_ids, positions, extend_start_loc, select_index = (
        fused_draft_extend_prepare_func(
            pad_rows(seq_lens.to(torch.int64), physical_bs),
            pad_rows(accept_lens.to(torch.int32), physical_bs, 1),
            pad_rows(next_token_ids.to(torch.int64), physical_bs * num_draft_tokens),
            num_draft_tokens,
        )
    )
    logical_num_tokens = logical_bs * num_draft_tokens
    return (
        num_correct_drafts[:logical_bs],
        token_ids[:logical_num_tokens],
        positions[:logical_num_tokens],
        extend_start_loc[:logical_bs],
        select_index[:logical_bs],
    )
