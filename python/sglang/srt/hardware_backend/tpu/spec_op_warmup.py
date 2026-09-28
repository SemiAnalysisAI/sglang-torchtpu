"""Warm-up data recipes for compiled speculative-decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SpecOpWarmupContext:
    """Shape and dtype metadata used by speculative helper warm-up recipes."""

    device: torch.device
    physical_bs: int
    num_verify_tokens: int
    vocab_size: int
    draft_vocab_size: int
    hidden_size: int
    hidden_dtype: torch.dtype


def make_verify_tree_greedy_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return (
        torch.zeros(
            (num_verify_rows, context.vocab_size),
            dtype=torch.float32,
            device=context.device,
        ),
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.arange(num_verify_rows, dtype=torch.int64, device=context.device).view(
            context.physical_bs, context.num_verify_tokens
        ),
    )


def make_target_only_sampling_args(context: SpecOpWarmupContext) -> tuple:
    bs, width = context.physical_bs, context.num_verify_tokens
    device = context.device
    probs = torch.full(
        (bs, width, context.vocab_size),
        1 / context.vocab_size,
        dtype=torch.float32,
        device=device,
    )
    return (
        torch.zeros(bs * width, dtype=torch.int32, device=device),
        torch.full((bs, width), -1, dtype=torch.int32, device=device),
        torch.zeros(bs, dtype=torch.int32, device=device),
        torch.zeros((bs, width), dtype=torch.int64, device=device),
        torch.arange(bs * width, dtype=torch.int64, device=device).view(bs, width),
        torch.full((bs, width), 0.5, device=device),
        torch.full((bs,), 0.5, device=device),
        probs,
        torch.zeros_like(probs),
        1.0,
        1.0,
        1,
    )


def make_verify_bookkeeping_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    return (
        torch.zeros((context.physical_bs,), dtype=torch.int64, device=context.device),
        torch.ones((context.physical_bs,), dtype=torch.int32, device=context.device),
        torch.zeros((num_verify_rows,), dtype=torch.int32, device=context.device),
        torch.arange(num_verify_rows, dtype=torch.int32, device=context.device).view(
            context.physical_bs, context.num_verify_tokens
        ),
    )


def make_draft_extend_select_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    num_verify_rows = context.physical_bs * context.num_verify_tokens
    select_index = (
        torch.arange(context.physical_bs, dtype=torch.int64, device=context.device)
        * context.num_verify_tokens
        + context.num_verify_tokens
        - 1
    )
    return (
        torch.zeros(
            (num_verify_rows, context.draft_vocab_size),
            dtype=torch.float32,
            device=context.device,
        ),
        torch.zeros(
            (num_verify_rows, context.hidden_size),
            dtype=context.hidden_dtype,
            device=context.device,
        ),
        select_index,
    )


def make_dflash_verify_argmax_args(
    context: SpecOpWarmupContext,
) -> tuple[Any, ...]:
    return (
        torch.zeros(
            (context.physical_bs * context.num_verify_tokens, context.vocab_size),
            dtype=torch.float32,
            device=context.device,
        ),
        context.num_verify_tokens,
    )


def make_dflash_verify_bookkeeping_args(
    context: SpecOpWarmupContext,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.zeros(
            (context.physical_bs, context.num_verify_tokens),
            dtype=torch.int64,
            device=context.device,
        ),
        torch.zeros(
            (context.physical_bs,),
            dtype=torch.int64,
            device=context.device,
        ),
    )
