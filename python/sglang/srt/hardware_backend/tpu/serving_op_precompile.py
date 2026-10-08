# SPDX-License-Identifier: Apache-2.0
"""Warm ordinary decoder operations outside the compiled model forward."""

import logging
import time

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang.srt.model_executor.forward_batch_info import (
    compute_local_num_token_non_padded,
)
from sglang.srt.runtime_context import get_disagg, get_flags

logger = logging.getLogger(__name__)


def precompile_serving_ops(
    *,
    compile_policy,
    future_map,
    spec_algorithm,
    max_running_requests,
    vocab_size,
    enable_mixed_chunk=False,
):
    """Cover logical batch sizes, including partially occupied model buckets.

    Model precompile stops before sampling and the cross-iteration token relay.
    These operations see logical batch sizes, not just the model's bucket sizes.
    Scratch buffers preserve live relay ownership and request allocation state.
    """
    policy = compile_policy
    if (
        not get_flags().capture.enable_torch_compile
        or not spec_algorithm.is_none()
        or not isinstance(policy, DecoderTpuCompilePolicy)
        or policy.b_bucketer is None
    ):
        return
    start = time.perf_counter()
    future = future_map
    device = future.output_tokens_buf.device
    mode = get_disagg().disaggregation_mode
    max_bs = min(max_running_requests, policy.b_bucketer.max_val)
    physical_sizes = {}
    for bs in range(1, max_bs + 1):
        sizes = set()
        if mode != "prefill":
            sizes.add(policy.b_bucketer.next_bucket(bs))
        if mode != "decode" and bs <= policy._extend_compile_cap():
            sizes.add(policy._extend_compile_cap())
        physical_sizes[bs] = sizes

    # LogitsProcessor returns float32 full-vocabulary logits. A trimmed view of
    # a padded result has a different native program from a compact allocation.
    logits = {
        size: torch.zeros(size, vocab_size, dtype=torch.float32).to(device)
        for size in {size for sizes in physical_sizes.values() for size in sizes}
    }
    with torch.no_grad():
        count = torch.tensor(1, dtype=torch.int32, device=device)
        compute_local_num_token_non_padded(count, max_bs, sharded=False).cpu()
        for bs, sizes in physical_sizes.items():
            indices = torch.arange(bs, dtype=torch.int64).to(device)
            values = torch.ones(bs, dtype=torch.int64).to(device)
            # Cover both a fresh (lazy-empty) buffer's first write and subsequent
            # read/modify/write operations. Never touch the live relay buffers.
            scratch = torch.empty_like(future.output_tokens_buf)
            for _ in range(2):
                future._store(scratch, indices, values)
                scratch.cpu()
            scratch[indices].cpu()
            if enable_mixed_chunk and mode == "null":
                # Same padded destination and tail width as mixed serving; vary
                # offsets as data so exact prompt lengths never specialize it.
                for tokens in policy.t_bucketer.buckets:
                    if tokens >= bs:
                        mixed = torch.zeros(tokens, dtype=torch.int64).to(device)
                        mixed.index_copy_(0, indices, values)
                        mixed.cpu()
            if mode != "prefill":
                padded_bs = policy.b_bucketer.next_bucket(bs)
                F.pad(values, (0, padded_bs - bs)).cpu()
            for size in sizes:
                torch.argmax(logits[size][:bs], dim=-1).cpu()
    if device.type == "tpu":
        torch.get_device_module(device).synchronize()
        snapshot_recompilation_baseline()
    logger.info(
        "[serving op precompile] warmed relay, padding and greedy sampling "
        "for logical batch sizes 1..%d in %.1fs",
        max_bs,
        time.perf_counter() - start,
    )
