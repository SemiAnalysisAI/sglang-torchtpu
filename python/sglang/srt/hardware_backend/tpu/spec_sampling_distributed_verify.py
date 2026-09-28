# SPDX-License-Identifier: Apache-2.0
"""Compiled distributed speculative-verify filtering and chain sampling."""

from dataclasses import replace

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    make_target_only_sampling_args,
)
from sglang.srt.hardware_backend.tpu.spec_sampling_verify import (
    sample_distributed_logits,
)


def _warmup(context):
    from sglang.srt.hardware_backend.tpu.sampler import distributed_top_k_min_batch_size

    if (
        context.vocab_size < 64
        or context.physical_bs < distributed_top_k_min_batch_size()
    ):
        return None
    args = make_target_only_sampling_args(replace(context, vocab_size=64))
    rows = context.physical_bs * context.num_verify_tokens
    return (
        *args[:8],
        torch.arange(64, dtype=torch.int32, device=context.device)
        .expand(rows, -1)
        .contiguous(),
        torch.full((rows, 1), 0.7, device=context.device),
        torch.full((rows,), 50, dtype=torch.int32, device=context.device),
        torch.full((rows,), 0.9, device=context.device),
        1.0,
        1.0,
    )


@compiled_op(warmup=_warmup)
def distributed_verify(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    coins,
    final_coins,
    target_probs,
    indices,
    temperatures,
    top_ks,
    top_ps,
    threshold_single,
    threshold_acc,
):
    tokens, counts = sample_distributed_logits(
        target_probs.reshape(-1, target_probs.shape[-1]),
        indices.to(torch.int32),
        temperatures.float(),
        top_ks.to(torch.int32).reshape(-1),
        top_ps.float().reshape(-1),
        candidates.to(torch.int32),
        coins,
        final_coins,
        threshold_single,
        threshold_acc,
    )
    valid = (
        torch.arange(candidates.shape[1], device=candidates.device)[None, :]
        <= counts[:, None]
    )
    flat_indices = retrieve_index.reshape(-1).long()
    predicts[flat_indices] = torch.where(
        valid.reshape(-1), tokens.reshape(-1).to(predicts.dtype), predicts[flat_indices]
    )
    accept_index[:] = torch.where(valid, retrieve_index, -1).to(accept_index.dtype)
    accept_token_num[:] = counts.to(accept_token_num.dtype)
