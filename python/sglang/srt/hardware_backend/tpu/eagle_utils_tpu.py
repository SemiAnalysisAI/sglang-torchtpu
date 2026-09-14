"""TPU helpers for EAGLE verify trees and draft extension."""

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_op_warmup import make_draft_extend_select_args


def build_topk1_verify_chain_torch(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """Build the fixed top-k=1 EAGLE verify tensors with pure torch ops.

    The fused TPU draft loop calls this before returning so tree construction
    stays in the same compiled program. The ordinary path also uses it as its
    torch-native fallback.

    Args:
        bonus_tokens: Bonus token IDs with shape ``[bs]``.
        draft_tokens: Draft token IDs with shape ``[bs, num_draft_tokens]``.
        seq_lens: Prefix lengths with shape ``[bs]``.

    Returns:
        An empty tree mask with shape ``[0]``, followed by positions, retrieve
        indices, next-token indices, next-sibling indices, and verification
        tokens, each with shape ``[bs, num_draft_tokens + 1]``.
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


@compiled_op(warmup=make_draft_extend_select_args)
def fused_draft_extend_select_argmax_func(
    next_token_logits: torch.Tensor,
    hidden_states: torch.Tensor | None,
    num_accept_tokens: torch.Tensor,
    num_tokens_per_req: int,
):
    row_starts = (
        torch.arange(
            num_accept_tokens.shape[0],
            dtype=torch.int64,
            device=num_accept_tokens.device,
        )
        * num_tokens_per_req
    )
    # Zero-count compile-padding rows select their own first window row.
    select_index = row_starts + num_accept_tokens.to(torch.int64).clamp_min(1) - 1
    selected_logits = next_token_logits[select_index]
    selected_hidden = hidden_states[select_index] if hidden_states is not None else None
    topk_index = torch.argmax(selected_logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    return selected_hidden, topk_p, topk_index
