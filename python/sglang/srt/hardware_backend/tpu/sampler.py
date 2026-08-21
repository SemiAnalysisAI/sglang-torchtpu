# SPDX-License-Identifier: Apache-2.0
"""TPU sampler with exact top-k reduction before vocabulary communication.
Unsupported sampling features retain the full-vocabulary PyTorch path."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.distributed import get_attn_tp_group, get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.sampler import (
    Sampler,
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

MAX_DISTRIBUTED_TOP_K = 64
_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE = max(
    1, envs.SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE.get()
)


def distributed_top_k_min_batch_size() -> int:
    """Return the process-static physical batch cutoff for candidate mode."""

    return _DISTRIBUTED_TOP_K_MIN_BATCH_SIZE


def distributed_top_k_for_batch(
    forward_batch,
    *,
    sampling_backend: str,
) -> Optional[int]:
    """Return the uniform requested k when candidate reduction is exact.

    The returned value is an eligibility bound, not the model output width.
    The eager reducer always emits a model-static candidate width, so changing
    the uniform request top-k does not create another model specialization.
    """

    if sampling_backend != "tpu":
        return None
    if forward_batch is None or getattr(forward_batch, "sampling_info", None) is None:
        return None
    spec_algorithm = getattr(forward_batch, "spec_algorithm", None)
    has_speculative_decode = spec_algorithm is not None and not spec_algorithm.is_none()
    # Compiled TPU batches are padded before mode selection. Use the physical
    # size because that is the work performed by the selected executable, and
    # never guard a graph on the logical size stored in ``_tpu_real_b``.
    batch_size = getattr(forward_batch, "batch_size", 0)
    if (
        getattr(forward_batch, "return_logprob", False)
        or has_speculative_decode
        or batch_size < _DISTRIBUTED_TOP_K_MIN_BATCH_SIZE
    ):
        return None

    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_extend_without_speculative()):
        return None

    info = forward_batch.sampling_info
    top_k = info.uniform_top_k
    if (
        top_k is None
        or not 1 < top_k <= MAX_DISTRIBUTED_TOP_K
        or info.is_any_greedy
        or info.sampling_seed is not None
        or not info.need_top_k_sampling
        or info.need_top_p_sampling
        or info.need_min_p_sampling
    ):
        return None
    if info.has_custom_logit_processor or info.logit_bias is not None:
        return None
    if any(info.grammars or ()) or getattr(info, "sampling_mask_max_top_k", 0) > 0:
        return None
    if (
        info.acc_additive_penalties is not None
        or info.acc_scaling_penalties is not None
    ):
        return None
    if (
        info.penalizer_orchestrator is not None
        and info.penalizer_orchestrator.is_required
    ):
        return None
    return top_k


def sample_from_global_top_k_logits(
    global_top_k_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize temperature-scaled global TopK logits and sample a position.
    Returned positions index sorted TopK slots, not the candidate-ID table.

    The eager reducer always returns one static candidate width. Masking the
    sorted candidates here recovers the batch's requested top-k distribution
    without specializing the model graph; the caller then uses the sampled
    slot to read the token ID aligned by the eager global merge.
    """

    if top_ks is not None:
        candidate_positions = torch.arange(
            global_top_k_logits.shape[-1],
            dtype=top_ks.dtype,
            device=global_top_k_logits.device,
        )
        keep = candidate_positions.view(1, -1) < top_ks.view(-1, 1)
        global_top_k_logits = global_top_k_logits.masked_fill(~keep, -torch.inf)

    global_top_k_probs = torch.softmax(global_top_k_logits / temperatures, dim=-1)
    return sampling_from_probs_torch(global_top_k_probs)


def _distributed_top_k_from_sharded_logits(
    logits_output,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the shard-local model output, then run distributed TopK.

    The explicit synchronization is a correctness boundary, not a timing
    optimization. It prevents TPU's lazy executor from composing either TopK
    custom op or the candidate collectives back into the complete 480B model
    program. Nine exact reproductions isolated that composition as unsafe.
    """

    sharded_logits = logits_output.next_token_sharded_logits
    vocab_shard = logits_output.tpu_top_k_vocab_shard
    is_tpu_output = sharded_logits.device.type == "tpu"
    if is_tpu_output:
        from torch_tpu._internal.sync.sync import synchronize as tpu_sync

        # This is the hard lazy-execution boundary. Do not fold it into the
        # helper below: the contiguous FP32 LM-head result must finish before
        # local TopK.
        tpu_sync([sharded_logits], wait=True)

    from sglang.srt.hardware_backend.tpu.top_k import distributed_tpu_top_k

    group = get_attn_tp_group() if get_parallel().enable_dp_lm_head else get_tp_group()
    # The width is model/layout static. Requests with K=2..64 share this one
    # eager program; their actual K is applied later as a tensor mask.
    candidate_width = min(
        MAX_DISTRIBUTED_TOP_K,
        vocab_shard.global_vocab_size,
        vocab_shard.width,
    )
    global_top_k = distributed_tpu_top_k(
        sharded_logits,
        vocab_shard,
        candidate_width,
        group=group,
    )
    if is_tpu_output:
        # Materialize the complete small TopK/collective program too. Startup
        # prewarms this exact boundary, and serving must not fuse it with later
        # sampling work into a first-use program after Dynamo has frozen.
        tpu_sync(list(global_top_k), wait=True)
    return global_top_k


class TpuSampler(Sampler):
    """Consume shard-local logits for eligible TPU sampling batches.
    Every other batch delegates to the full-vocabulary sampler implementation."""

    def forward(
        self,
        logits_output,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums,
        token_ids_logprobs,
        positions: torch.Tensor,
    ):
        """Sample from exact global TopK when full logits gathering was skipped.
        Full-logit outputs retain the base sampler's processing and return contract."""

        # No sharded output: sample from full-vocabulary logits instead.
        if not logits_output.has_next_token_sharded_logits:
            return super().forward(
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
            )

        if return_logprob:
            raise ValueError("distributed TPU top-k cannot return full logprobs")
        # All local/global TopK work and bounded TP communication run only after
        # the FP32 shard-local output has been materialized. This keeps
        # distributed TopK enabled without any TopK custom op in the 480B
        # executable.
        global_top_k_logits, global_top_k_token_ids = (
            _distributed_top_k_from_sharded_logits(logits_output)
        )
        sampled_top_k_slot = sample_from_global_top_k_logits(
            global_top_k_logits,
            sampling_info.temperatures,
            sampling_info.top_ks,
        )
        # The eager merge already aligned exact global token IDs with sorted
        # logits, so sampling needs only this final bounded lookup.
        batch_next_token_ids = torch.gather(
            global_top_k_token_ids,
            1,
            sampled_top_k_slot.to(torch.int64).view(-1, 1),
        ).view(-1)
        batch_next_token_ids = batch_next_token_ids.to(torch.int64)
        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
        return batch_next_token_ids

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> torch.Tensor:
        """Use native XLA TopK after gathering uniform small-k probability rows.
        This avoids a full sort when pre-gather reduction is not selected."""

        if simple_sampling_case:
            return sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
            )
        top_k = sampling_info.uniform_top_k
        if (
            top_k is not None
            and 1 < top_k <= MAX_DISTRIBUTED_TOP_K
            and sampling_info.sampling_seed is None
            and sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_min_p_sampling
        ):
            from sglang.srt.hardware_backend.tpu.top_k import tpu_top_k

            global_top_k_probs, global_top_k_token_ids = tpu_top_k(probs, top_k)
            sampled_top_k_position = sampling_from_probs_torch(global_top_k_probs)
            # This lookup is an eager sampler operation over K<=64, not the
            # post-collective lookup that must stay out of the model graph.
            return torch.gather(
                global_top_k_token_ids,
                1,
                sampled_top_k_position.to(torch.int64).view(-1, 1),
            ).view(-1)

        return top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            sampling_info.top_ks,
            sampling_info.top_ps,
            sampling_info.min_ps,
            sampling_info.need_min_p_sampling,
            sampling_info.sampling_seed,
            positions,
        )
