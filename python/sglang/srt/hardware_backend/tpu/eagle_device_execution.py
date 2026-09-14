"""Experimental EAGLE execution metadata with device lengths and a host allocator.

The host keeps the authoritative slot table. Small, value-checked page-table
snapshots stay on TPU; dynamic sequence lengths never index the CPU table.
This prototype supports single-pipeline, DP=1, top-1 paged EAGLE only.
"""

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.host_metadata_cache import HostMetadataCache
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len
from sglang.srt.runtime_context import get_parallel, get_spec


def enabled():
    return envs.SGLANG_TPU_EAGLE_DEVICE_EXECUTION.get()


def validate_configuration(model_runner):
    if not enabled():
        return False
    spec = get_spec()
    parallel = get_parallel()
    if (
        not model_runner.spec_algorithm.is_eagle3()
        or parallel.dp_size != 1
        or parallel.pp_size != 1
        or spec.speculative_eagle_topk != 1
        or not spec.speculative_num_steps
        or spec.speculative_adaptive
        or model_runner.page_size <= 1
        or getattr(getattr(model_runner, "model_config", None), "model_is_mrope", False)
    ):
        raise ValueError(
            "Device EAGLE execution requires DP=1, PP=1, fixed top-1 paged EAGLE3 "
            "without adaptive speculation or multimodal RoPE"
        )
    return True


def capacity_for_batch(batch):
    page_size = batch.token_to_kv_pool_allocator.page_size
    context = batch.model_config.context_len + get_req_to_token_extra_context_len()
    return ((context + page_size - 1) // page_size) * page_size


def allocation_upper_bound(batch):
    return max((req.kv.kv_allocated_len for req in batch.reqs), default=0)


def requires_cpu_capacity_mirror(batch):
    # Keep the old exact-length check for unusual allocation states near the
    # capacity boundary. Normal execution is bounded by its reserved KV pages.
    upper = allocation_upper_bound(batch)
    return bool(batch.reqs) and (
        any(req.kv.kv_allocated_len <= 0 for req in batch.reqs)
        or upper > capacity_for_batch(batch)
    )


def _slots(pages, seq_lens, width, page_size):
    columns = (
        seq_lens.to(torch.int32)[:, None]
        + torch.arange(width, dtype=torch.int32, device=seq_lens.device)[None, :]
    )
    page_ids = torch.gather(pages, 1, (columns // page_size).to(torch.int64))
    return (page_ids * page_size + columns % page_size).reshape(-1).to(torch.int64)


def _positions(prefix_lens, width):
    offsets = torch.arange(width, dtype=torch.int32, device=prefix_lens.device)
    positions = prefix_lens.to(torch.int32)[:, None] + offsets[None, :]
    starts = (
        torch.arange(prefix_lens.shape[0], dtype=torch.int32, device=prefix_lens.device)
        * width
    )
    return positions.reshape(-1).to(torch.int64), starts


def _key(*values):
    return tuple(
        (
            (tuple(v.shape), v.dtype, v.device, tuple(v.stride()))
            if isinstance(v, torch.Tensor)
            else v
        )
        for v in values
    )


_compiled_slots = None
_compiled_positions = None


def _functions():
    global _compiled_slots, _compiled_positions
    if _compiled_slots is None:
        from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
            direct_compile,
        )

        _compiled_slots = direct_compile(_slots, specialization_key=_key)
        _compiled_positions = direct_compile(_positions, specialization_key=_key)
    return _compiled_slots, _compiled_positions


def slot_locations(batch, req_to_token, width):
    assert batch.seq_lens_cpu is None
    assert req_to_token.device.type == "cpu"
    page_size = batch.token_to_kv_pool_allocator.page_size
    capacity = capacity_for_batch(batch)
    assert 0 < allocation_upper_bound(batch) <= capacity
    assert width <= get_spec().speculative_num_draft_tokens
    rows = batch.req_pool_indices_cpu
    assert rows is not None and rows.device.type == "cpu"
    assert rows.shape == batch.seq_lens.shape
    state = getattr(req_to_token, "_eagle_execution_page_cache", None)
    if state is None or state[0] != (capacity, page_size):
        columns = torch.arange(0, capacity, page_size, dtype=torch.int64)
        state = ((capacity, page_size), columns, HostMetadataCache())
        req_to_token._eagle_execution_page_cache = state
    _, columns, cache = state
    # Advanced indexing owns its result; later allocation/retraction/reuse
    # cannot mutate a snapshot already queued for an earlier model call.
    pages_cpu = (
        req_to_token[rows.long()[:, None], columns[None, :]] // page_size
    ).int()
    pages = cache.get("pages", pages_cpu, batch.seq_lens.device)
    slots_fn, _ = _functions()
    return slots_fn(pages, batch.seq_lens, width, page_size)


def uniform_positions(prefix_lens, width):
    _, positions_fn = _functions()
    return positions_fn(prefix_lens, width)


def precompile(device, max_bs, context_len, page_size, draft_steps, verify_width):
    if not enabled():
        return
    capacity = context_len + get_req_to_token_extra_context_len()
    pages_per_seq = (capacity + page_size - 1) // page_size
    slots_fn, positions_fn = _functions()
    for bs in range(max_bs + 1):
        pages = torch.zeros((bs, pages_per_seq), dtype=torch.int32, device=device)
        for dtype in (torch.int32, torch.int64):
            lengths = torch.zeros(bs, dtype=dtype, device=device)
            for width in sorted({draft_steps, verify_width}):
                slots_fn(pages, lengths, width, page_size)
            positions_fn(lengths, verify_width)
    torch.get_device_module(device).synchronize()
