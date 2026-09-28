# SPDX-License-Identifier: Apache-2.0
"""Target-only chain verification over compact categorical distributions.

Candidate order changes the mapping from random coins to tokens, not their
probabilities. This module consumes distributed exact-k candidates only.
"""

import torch


def sample_chain(
    values: torch.Tensor,
    indices: torch.Tensor,
    candidates: torch.Tensor,
    coins: torch.Tensor,
    final_coins: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, width = candidates.shape
    device = candidates.device
    ar = torch.arange(bs, device=device)
    num_correct_drafts = torch.zeros(bs, dtype=torch.int64, device=device)
    active = torch.ones(bs, dtype=torch.bool, device=device)
    has_rejected = torch.zeros(bs, dtype=torch.bool, device=device)
    rejected_tokens = torch.zeros(bs, dtype=candidates.dtype, device=device)
    tokens = torch.zeros((bs, width), dtype=candidates.dtype, device=device)
    coin = coins[:, 0]
    for position in range(1, width):
        candidate = candidates[:, position]
        row_values = values[ar, num_correct_drafts]
        row_indices = indices[ar, num_correct_drafts]
        probability = torch.where(row_indices == candidate[:, None], row_values, 0).sum(
            -1
        )
        accept = (coin <= probability / max(threshold_acc, 1e-9)) | (
            probability >= threshold_single
        )
        accept_now = active & accept
        reject_now = active & ~accept
        tokens = tokens.index_put(
            (ar, num_correct_drafts),
            torch.where(accept_now, candidate, tokens[ar, num_correct_drafts]),
        )
        rejected_tokens = torch.where(reject_now, candidate, rejected_tokens)
        has_rejected = has_rejected | reject_now
        num_correct_drafts = num_correct_drafts + accept_now.long()
        coin = torch.where(accept_now, coins[:, position], coin)
        active = active & ~reject_now
    residual = values[ar, num_correct_drafts]
    row_indices = indices[ar, num_correct_drafts]
    residual = torch.where(
        has_rejected[:, None] & (row_indices == rejected_tokens[:, None]), 0, residual
    )
    total = residual.sum(-1)
    cdf = residual.cumsum(-1)
    # Ignore zero-mass padding, including at the RNG's zero endpoint. Clamp
    # the upper endpoint to the final positive entry despite scan rounding.
    hits = (residual > 0) & (cdf >= (final_coins * total)[:, None])
    positions = torch.arange(residual.shape[-1], device=device)[None, :]
    last = torch.where(residual > 0, positions, 0).amax(-1)
    selected = torch.where(hits, positions, residual.shape[-1]).amin(-1)
    selected = torch.minimum(selected, last)
    bonus_tokens = row_indices[ar, selected]
    # Degenerate zero-mass input follows the old verifier's token-zero result.
    bonus_tokens = torch.where(total > 0, bonus_tokens, 0)
    tokens = tokens.index_put((ar, num_correct_drafts), bonus_tokens)
    return tokens, num_correct_drafts.to(torch.int32)


def filter_distributed_logits(
    logits: torch.Tensor,
    indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize exact-k candidates, then keep the inclusive nucleus prefix.

    Input is descending logit / ascending token ID. Temperature is positive;
    normalizing after truncation is algebraically identical to full
    softmax followed by top-k renormalization. Floating-point scans need not be
    bitwise identical to the vocabulary-wide scan.
    """
    positions = torch.arange(logits.shape[-1], device=logits.device)[None, :]
    scaled = logits.float() / temperatures.reshape(-1, 1)
    scaled = scaled.masked_fill(positions >= top_ks.reshape(-1, 1), -torch.inf)
    values = torch.softmax(scaled, dim=-1)
    keep = (values.cumsum(-1) - values < top_ps.reshape(-1, 1)) | (positions == 0)
    values = torch.where(keep, values, 0)
    values = values / values.sum(-1, keepdim=True)
    return values, indices


def sample_distributed_logits(
    logits: torch.Tensor,
    indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    candidates: torch.Tensor,
    coins: torch.Tensor,
    final_coins: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = filter_distributed_logits(
        logits, indices, temperatures, top_ks, top_ps
    )
    bs, width = candidates.shape
    return sample_chain(
        values.reshape(bs, width, -1),
        indices.reshape(bs, width, -1),
        candidates,
        coins,
        final_coins,
        threshold_single,
        threshold_acc,
    )
