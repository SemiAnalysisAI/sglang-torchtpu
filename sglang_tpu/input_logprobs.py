# SPDX-License-Identifier: Apache-2.0
"""Bound prompt TopK compilation by row and output-width buckets."""

from __future__ import annotations

import functools

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    _append_top_logprobs_chunk,
)
from sglang.utils import LazyImport
from sglang_tpu.compilation.compiled_forward_shim import _pad_tensor
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.output_logprobs import _compiled_sort, _sort_logprobs_impl

direct_compile = LazyImport("sglang_tpu.compilation.direct_compile", "direct_compile")
_SORT_ROWS = 128


def _gather_rows(logprobs, indices):
    return logprobs.index_select(0, indices)


def _select_columns(values, indices, width):
    return values[:, :width].contiguous(), indices[:, :width].contiguous()


@functools.cache
def _compiled_gather():
    return direct_compile(
        _gather_rows,
        specialization_key=lambda values, indices: (
            tuple(values.shape),
            values.dtype,
            values.device,
        ),
    )


@functools.cache
def _compiled_select():
    return direct_compile(
        _select_columns,
        specialization_key=lambda values, indices, width: (
            tuple(values.shape),
            values.dtype,
            values.device,
            width,
        ),
    )


def _copy_top_logprobs(logprobs, max_k, log_normalizer, precomputed_topk):
    rows, vocab_size = logprobs.shape
    if not 0 <= max_k <= vocab_size:
        raise RuntimeError("selected index k out of range")
    if not rows or not max_k:
        return [[] for _ in range(rows)], [[] for _ in range(rows)]
    if precomputed_topk is not None and log_normalizer is not None:
        values, indices = (tensor.cpu() for tensor in precomputed_topk)
        row_max, row_log_sum = (tensor.cpu() for tensor in log_normalizer)
        values = (values.float() - row_max[:, None]) - row_log_sum[:, None]
        return values.tolist(), indices.tolist()

    width = min(1 << (max_k - 1).bit_length(), vocab_size)
    logprobs = _pad_tensor(logprobs, 1 << (rows - 1).bit_length())
    native = logprobs.device.type == "tpu"
    gather = _compiled_gather() if native else _gather_rows
    sort = _compiled_sort() if native else _sort_logprobs_impl
    select = _compiled_select() if native else _select_columns
    chunks = []
    for start in range(0, rows, _SORT_ROWS):
        count = min(_SORT_ROWS, rows - start)
        # Offset is tensor data, so later chunks reuse the first chunk's gather.
        row_indices = torch.arange(
            start, start + _SORT_ROWS, dtype=torch.int64, device="cpu"
        ).clamp_max(rows - 1)
        chunk = gather(logprobs, row_indices.to(logprobs.device))
        values, indices = sort(chunk)
        values, indices = select(values, indices, width)
        chunks.append((start, count, values, indices))

    normalizer = (
        tuple(tensor.cpu() for tensor in log_normalizer)
        if log_normalizer is not None
        else None
    )
    all_values, all_indices = [], []
    for start, count, values, indices in chunks:
        values = values.cpu()[:count, :max_k]
        indices = indices.cpu()[:count, :max_k]
        if normalizer is not None:
            row_max, row_log_sum = normalizer
            values = (
                values.float() - row_max[start : start + count, None]
            ) - row_log_sum[start : start + count, None]
        all_values.extend(values.tolist())
        all_indices.extend(indices.tolist())
    return all_values, all_indices


def get_top_logprobs_chunk(
    logprobs,
    top_k_nums,
    pruned_lens,
    top_logprobs_val,
    top_logprobs_idx,
    split_pruned_len,
    log_normalizer=None,
    precomputed_topk=None,
    copy_to_pinned_cpu=False,
):
    values, indices = _copy_top_logprobs(
        logprobs, max(top_k_nums), log_normalizer, precomputed_topk
    )
    if copy_to_pinned_cpu:
        values = torch.tensor(
            values, dtype=torch.float32, device="cpu", pin_memory=True
        )
        indices = torch.tensor(
            indices, dtype=torch.int64, device="cpu", pin_memory=True
        )
    return _append_top_logprobs_chunk(
        values,
        indices,
        len(values),
        top_k_nums,
        pruned_lens,
        top_logprobs_val,
        top_logprobs_idx,
        split_pruned_len,
        copy_to_pinned_cpu,
    )


class TpuInputLogprobProcessor(InputLogprobProcessor):
    def __init__(self, vocab_size, chunking_group=None, *, skip_chunking=False):
        super().__init__(
            vocab_size,
            chunking_group,
            get_top_logprobs_chunk_fn=get_top_logprobs_chunk,
        )
        self.skip_chunking = skip_chunking

    def precompile(self, max_rows, device):
        """Queue shape-only examples; no full-vocabulary output is copied to CPU."""
        if self.enable_logprobs_chunk and not self.skip_chunking:
            max_rows = min(max_rows, self.logprobs_chunk_size)
        max_rows = 1 << (max_rows - 1).bit_length()
        async_compile = envs.SGLANG_DIRECT_COMPILE_ASYNC.get()
        pending = []
        indices = torch.empty(_SORT_ROWS, dtype=torch.int64, device=device)
        rows = 1
        while rows <= max_rows:
            example = torch.empty(
                rows, self.vocab_size, dtype=torch.float32, device=device
            )
            pending.append(
                _compiled_gather().precompile(
                    example, indices, async_compile=async_compile
                )
            )
            rows *= 2
        values = torch.empty(
            _SORT_ROWS, self.vocab_size, dtype=torch.float32, device=device
        )
        indices = torch.empty(
            _SORT_ROWS, self.vocab_size, dtype=torch.int64, device=device
        )
        pending.append(_compiled_sort().precompile(values, async_compile=async_compile))
        width = 1
        while width < self.vocab_size:
            pending.append(
                _compiled_select().precompile(
                    values, indices, width, async_compile=async_compile
                )
            )
            width *= 2
        pending.append(
            _compiled_select().precompile(
                values, indices, self.vocab_size, async_compile=async_compile
            )
        )
        # Exercise first-use host/runtime setup after compilation has joined.
        pending.append(
            PendingCall(
                lambda: _copy_top_logprobs(
                    torch.zeros(
                        min(max_rows, _SORT_ROWS),
                        self.vocab_size,
                        dtype=torch.float32,
                        device="cpu",
                    ).to(device),
                    1,
                    None,
                    None,
                )
            )
        )
        return pending
