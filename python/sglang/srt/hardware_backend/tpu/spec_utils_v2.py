"""TPU-only helpers for SpecV2 execution."""

from __future__ import annotations

import torch

from sglang.srt.utils.common import tpu_fused_spec_op

_topk1_tree_constants: dict[tuple[str, int, int], tuple[torch.Tensor, ...]] = {}


@tpu_fused_spec_op
def _finalize_topk1_tree(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    token_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1)
    positions = seq_lens[:, None] + token_offsets[None, :]
    return tokens.flatten(), positions.flatten()


def build_topk1_tree(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_verify_tokens: int,
) -> tuple[torch.Tensor, ...]:
    """Build a top-k=1 tree while reusing its shape-only device tensors."""
    device = seq_lens.device
    bs = seq_lens.numel()
    key = (str(device), bs, num_verify_tokens)
    cached = _topk1_tree_constants.get(key)
    if cached is None:
        retrieve_index = torch.arange(bs * num_verify_tokens, dtype=torch.long).view(
            bs, num_verify_tokens
        )
        retrieve_next_token = torch.full((bs, num_verify_tokens), -1, dtype=torch.long)
        if num_verify_tokens > 1:
            retrieve_next_token[:, :-1] = torch.arange(
                1, num_verify_tokens, dtype=torch.long
            )
        cached = (
            torch.empty((0,), dtype=torch.bool).to(device),
            retrieve_index.to(device),
            retrieve_next_token.to(device),
            torch.full((bs, num_verify_tokens), -1, dtype=torch.long, device="cpu").to(
                device
            ),
            torch.arange(num_verify_tokens, dtype=torch.long).to(device),
        )
        _topk1_tree_constants[key] = cached

    tree_mask, retrieve_index, retrieve_next_token, retrieve_next_sibling, offsets = (
        cached
    )
    tokens, positions = _finalize_topk1_tree(
        bonus_tokens, draft_tokens, seq_lens, offsets
    )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        tokens,
    )
