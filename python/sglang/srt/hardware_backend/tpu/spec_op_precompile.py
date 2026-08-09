"""Startup precompile for fixed-shape TPU Spec V2 helper ops."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable, Sequence

import torch

from sglang.srt.utils.common import get_tpu_fused_spec_ops

logger = logging.getLogger(__name__)

_SPEC_V2_HELPER_MODULES = (
    "sglang.srt.speculative.eagle_utils",
    "sglang.srt.hardware_backend.tpu.spec_utils_v2",
    "sglang.srt.managers.overlap_utils",
)


def _find_fused_ops(op_names: Sequence[str]) -> dict[str, Callable]:
    """Import helper owners and resolve the requested registered wrappers."""
    for module_name in _SPEC_V2_HELPER_MODULES:
        importlib.import_module(module_name)

    wanted = set(op_names)
    found: dict[str, Callable] = {}
    for op in get_tpu_fused_spec_ops():
        if op.__name__ not in wanted:
            continue
        if op.__name__ in found:
            raise ValueError(f"Duplicate fused spec op name: {op.__name__}")
        found[op.__name__] = op

    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Fused spec ops were not registered: {sorted(missing)}")
    return found


def _build_op_calls(
    *,
    device: torch.device,
    physical_bs: int,
    logical_bs: int | None = None,
    num_verify_tokens: int,
    num_spec_steps: int,
    verify_vocab_size: int,
    draft_vocab_size: int,
    verify_logits_dtype: torch.dtype,
    draft_logits_dtype: torch.dtype,
    hidden_size: int,
    hidden_dtype: torch.dtype,
    req_pool_size: int,
    warm_device_accept: bool,
    include_bucket_helpers: bool = True,
    include_logical_helpers: bool = True,
) -> list[tuple[str, tuple]]:
    """Build serving-equivalent calls for one physical or logical batch."""
    logical_bs = physical_bs if logical_bs is None else logical_bs
    num_verify_rows = physical_bs * num_verify_tokens
    logical_verify_rows = logical_bs * num_verify_tokens
    indices = torch.arange(logical_bs, dtype=torch.int64, device=device)
    seq_lens = torch.zeros((logical_bs,), dtype=torch.int64, device=device)
    calls = []

    if include_bucket_helpers:
        token_offsets = torch.arange(
            num_verify_tokens, dtype=torch.int64, device=device
        )
        draft_tokens = torch.zeros(
            (logical_bs, num_verify_tokens - 1), dtype=torch.int64, device=device
        )
        candidates = torch.zeros(
            (physical_bs, num_verify_tokens), dtype=torch.int64, device=device
        )
        verify_logits = torch.zeros(
            (num_verify_rows, verify_vocab_size),
            dtype=verify_logits_dtype,
            device=device,
        )
        retrieve_index = torch.arange(
            num_verify_rows, dtype=torch.int64, device=device
        ).view(physical_bs, num_verify_tokens)
        predict = torch.zeros(
            (logical_verify_rows,), dtype=torch.int32, device=device
        )
        accept_index = torch.arange(
            logical_verify_rows, dtype=torch.int32, device=device
        ).view(logical_bs, num_verify_tokens)
        accept_lens = torch.ones((logical_bs,), dtype=torch.int32, device=device)
        draft_logits = torch.zeros(
            (num_verify_rows, draft_vocab_size),
            dtype=draft_logits_dtype,
            device=device,
        )
        hidden_states = torch.zeros(
            (num_verify_rows, hidden_size), dtype=hidden_dtype, device=device
        )
        select_index = (
            torch.arange(logical_bs, dtype=torch.int64, device=device)
            * num_verify_tokens
            + num_verify_tokens
            - 1
        )
        calls.extend(
            [
                (
                    "_finalize_topk1_tree",
                    (
                        torch.zeros(
                            (logical_bs,), dtype=torch.int64, device=device
                        ),
                        draft_tokens,
                        seq_lens,
                        token_offsets,
                    ),
                ),
                (
                    "greedy_verify_fused_func",
                    (verify_logits, candidates, retrieve_index),
                ),
                (
                    "verify_post_fused_func",
                    (
                        seq_lens,
                        accept_lens,
                        predict,
                        accept_index,
                        num_verify_tokens,
                    ),
                ),
                (
                    "draft_extend_select_argmax_fused_func",
                    (draft_logits, hidden_states, select_index),
                ),
            ]
        )

    if not include_logical_helpers:
        return calls

    accept_lens = torch.ones((logical_bs,), dtype=torch.int32, device=device)
    topk_p = torch.ones((logical_bs, 1), dtype=torch.float32, device=device)
    topk_index = torch.zeros((logical_bs, 1), dtype=torch.int64, device=device)
    selected_hidden = torch.zeros(
        (logical_bs, hidden_size), dtype=hidden_dtype, device=device
    )
    output_tokens_buf = torch.empty(
        (req_pool_size,), dtype=torch.int64, device=device
    )
    topk_p_buf = torch.empty((req_pool_size, 1), dtype=torch.float32, device=device)
    topk_index_buf = torch.empty(
        (req_pool_size, 1), dtype=torch.int64, device=device
    )
    hidden_states_buf = torch.empty(
        (req_pool_size, hidden_size), dtype=hidden_dtype, device=device
    )

    # Prefill relays model token ids (int64), while greedy verify produces
    # int32 bonus tokens. Dynamo guards dtype, so precompile both real variants.
    for bonus_dtype in (torch.int64, torch.int32):
        calls.append(
            (
                "store_future_map_spec_hidden",
                (
                    indices,
                    output_tokens_buf,
                    topk_p_buf,
                    topk_index_buf,
                    hidden_states_buf,
                    torch.zeros(
                        (logical_bs,), dtype=bonus_dtype, device=device
                    ),
                    topk_p,
                    topk_index,
                    selected_hidden,
                ),
            )
        )

    if warm_device_accept:
        shared_width = max(num_spec_steps, num_verify_tokens)
        candidate_width = num_verify_tokens + shared_width - 1
        cache_loc_width = (candidate_width + 1) // 2
        candidate_locs = torch.zeros(
            (logical_bs, candidate_width), dtype=torch.int64, device=device
        )
        calls.append(
            (
                "_gather_spec_extras_hidden_and_cache_locs_tpu",
                (
                    indices,
                    seq_lens.new_empty((req_pool_size,)),
                    topk_p_buf,
                    topk_index_buf,
                    output_tokens_buf,
                    hidden_states_buf,
                    candidate_locs,
                    accept_lens,
                    cache_loc_width,
                ),
            )
        )
    return calls


def precompile_spec_v2_ops(
    *,
    device: torch.device,
    physical_batch_buckets: Sequence[int],
    num_verify_tokens: int,
    num_spec_steps: int,
    verify_vocab_size: int,
    draft_vocab_size: int,
    verify_logits_dtype: torch.dtype,
    draft_logits_dtype: torch.dtype,
    hidden_size: int,
    hidden_dtype: torch.dtype,
    req_pool_size: int,
    warm_device_accept: bool,
) -> tuple[tuple[int, int], ...]:
    """Precompile top-k=1 EAGLE3 helpers for every supported logical batch."""
    shapes = tuple(
        (physical_bs, physical_bs * num_verify_tokens)
        for physical_bs in physical_batch_buckets
    )
    if not shapes:
        return shapes
    max_physical_bs = max(physical_bs for physical_bs, _ in shapes)
    if req_pool_size < max_physical_bs:
        raise ValueError(
            f"FutureMap capacity {req_pool_size} cannot warm batch {max_physical_bs}."
        )

    op_names = (
        "_finalize_topk1_tree",
        "greedy_verify_fused_func",
        "verify_post_fused_func",
        "draft_extend_select_argmax_fused_func",
        "store_future_map_spec_hidden",
    )
    if warm_device_accept:
        op_names += ("_gather_spec_extras_hidden_and_cache_locs_tpu",)
    ops = _find_fused_ops(op_names)

    start = time.perf_counter()
    with torch.no_grad():
        # Spec helpers receive the exact, unpadded scheduler batch size even
        # when model outputs retain their padded physical token rows.
        for logical_bs in range(1, max_physical_bs + 1):
            physical_bs = min(
                bucket for bucket in physical_batch_buckets if bucket >= logical_bs
            )
            calls = _build_op_calls(
                device=device,
                physical_bs=physical_bs,
                logical_bs=logical_bs,
                num_verify_tokens=num_verify_tokens,
                num_spec_steps=num_spec_steps,
                verify_vocab_size=verify_vocab_size,
                draft_vocab_size=draft_vocab_size,
                verify_logits_dtype=verify_logits_dtype,
                draft_logits_dtype=draft_logits_dtype,
                hidden_size=hidden_size,
                hidden_dtype=hidden_dtype,
                req_pool_size=req_pool_size,
                warm_device_accept=warm_device_accept,
            )
            for op_name, args in calls:
                ops[op_name](*args)
    if device.type == "tpu":
        torch.get_device_module(device).synchronize()
    logger.info(
        "[spec op precompile] precompiled %d helpers for logical batch sizes "
        "1..%d (configured bucket shapes: %s) in %.1fs",
        len(ops),
        max_physical_bs,
        shapes,
        time.perf_counter() - start,
    )
    return shapes
