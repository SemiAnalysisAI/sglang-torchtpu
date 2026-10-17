"""Stateless speculative-decoding features shared by every spec worker.

Each function takes explicit tensors and small inputs, never a worker,
scheduler or model runner, so an out-of-tree worker gets the feature by
calling it. Each docstring names its trace class:

- traceable: static shapes, no host sync or data-dependent output size; safe
  inside a compiled region.
- host: runs between forwards on host metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

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


def seeded_verify_coins(
    *,
    sampling_seed: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive deterministic verify-side coins from per-request sampling seeds.

    Traceable.
    Mirrors the main seeded-sampling path: murmur_hash32(seed, seq_lens,
    column) mapped to [0, 1). Columns [0, draft_token_num) drive the
    per-draft rejection coins; column draft_token_num drives the final
    fallback-sampling coin.

    Scope: this seeds only the verify-side RNG. With rejection sampling the
    draft workers still pick candidates via unseeded multinomial
    (fast_sample in eagle_worker_v2), so that mode stays non-deterministic
    until the draft RNG is seeded in a follow-up; top-k/greedy draft
    selection is already deterministic.
    """
    from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

    cols = torch.arange(draft_token_num + 1, device=device, dtype=torch.int64)
    hashed = murmur_hash32(
        sampling_seed.to(torch.uint64), seq_lens.to(torch.uint64), cols
    )
    uniforms = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
    # The float32 cast rounds the top 129 uint32 hashes to exactly 1.0, but
    # the sampling kernels expect half-open [0, 1) coins: a 1.0 coin walks
    # past the last CDF bucket and can return a zero-probability token.
    # Clamp to the largest float32 below one; every other coin value is
    # untouched, so previously verified bitwise baselines stay intact.
    max_coin = 1.0 - 2**-24
    coins = (
        uniforms[:, :draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    coins_for_final_sampling = (
        uniforms[:, draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    return coins, coins_for_final_sampling


def verify_coins(
    *,
    sampling_info: SamplingBatchInfo,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    candidates: torch.Tensor,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rejection and final-sampling coins for verify: deterministic seeded
    coins when sampling_seed is set (see seeded_verify_coins), torch.rand
    otherwise. Traceable.
    """
    if sampling_info.sampling_seed is not None:
        return seeded_verify_coins(
            sampling_seed=sampling_info.sampling_seed,
            seq_lens=seq_lens,
            draft_token_num=draft_token_num,
            device=device,
        )
    # coins for rejection sampling
    coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
    # coins for final sampling
    coins_for_final_sampling = torch.rand(
        (candidates.shape[0],), dtype=torch.float32, device=device
    )
    return coins, coins_for_final_sampling
