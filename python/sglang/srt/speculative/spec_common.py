"""Stateless speculative-decoding features shared by every spec worker.

Each function takes explicit tensors and small inputs, never a worker,
scheduler or model runner, so an out-of-tree worker gets the feature by
calling it. Each docstring names its trace class:

- traceable: static shapes, no host sync or data-dependent output size; safe
  inside a compiled region.
- host: runs between forwards on host metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def adjust_verify_logits(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    draft_token_num: int,
) -> None:
    """Apply accumulated penalties and logit_bias to verify logits in place.

    Traceable. Each request's per-request row is repeated over its
    ``draft_token_num`` verify rows, a relaxed form of penalties for
    speculative decoding. Rows past ``len(sampling_info) * draft_token_num``
    (bucket padding) are left untouched.
    """
    # Looked up at call time: a platform may rebind this module attribute at startup.
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )

    if sampling_info.acc_additive_penalties is not None:
        additive_penalties = torch.repeat_interleave(
            sampling_info.acc_additive_penalties,
            draft_token_num,
            dim=0,
        )
        next_token_logits[: additive_penalties.shape[0]].add_(additive_penalties)
    if sampling_info.acc_scaling_penalties is not None:
        scaling_penalties = torch.repeat_interleave(
            sampling_info.acc_scaling_penalties,
            draft_token_num,
            dim=0,
        )
        apply_scaling_penalties(
            next_token_logits[: scaling_penalties.shape[0]],
            scaling_penalties,
        )
    if sampling_info.logit_bias is not None:
        logit_bias = torch.repeat_interleave(
            sampling_info.logit_bias, draft_token_num, dim=0
        )
        next_token_logits[: logit_bias.shape[0]].add_(logit_bias)
