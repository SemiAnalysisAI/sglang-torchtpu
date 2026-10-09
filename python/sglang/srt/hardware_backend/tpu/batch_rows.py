# SPDX-License-Identifier: Apache-2.0
"""Merge and filter per-request tensors without a TPU program per length pair.

The scheduler merges a prefill batch into the running batch and drops finished
requests with ``torch.cat`` and ``tensor[keep]`` on the exact batch lengths of
per-request tensors (``seq_lens``, ``req_pool_indices``, sampling parameters,
...). torch_tpu compiles one XLA program per input shape, so every
(running, joining) or (before, after) length pair the batch dynamics happen to
produce is a new program: hundreds per serving sweep at high concurrency, and
the set keeps growing for as long as new pairs turn up.

On TPU every operand is first zero-padded to a fixed capacity, which costs one
program per operand length, and the rows are picked with ``index_select`` from
an index built on the host, which costs one program per result length. Both
sets are bounded by the capacity, so a server that has seen each batch size
once compiles nothing more. Other devices keep the plain ops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.utils.common import next_power_of_2

logger = get_logger(__name__)

_LAYOUTS = (
    (torch.int64, ()),
    (torch.int32, ()),
    (torch.float32, ()),
    (torch.float32, (1,)),
)


@dataclass(frozen=True)
class _MergeIndex:
    key: tuple[int, int, int, str]
    tensor: torch.Tensor


@dataclass
class _BatchRowsState:
    """Each scheduler process owns one capacity and one read-only merge index."""

    capacity: int | None = None
    layouts: list[tuple[torch.dtype, tuple[int, ...]]] = field(
        default_factory=lambda: list(_LAYOUTS)
    )
    merge_index: _MergeIndex | None = None
    warned_capacity: int = 0
    layouts_frozen: bool = False


_state = _BatchRowsState()


def configure_capacity(max_running_requests: int) -> None:
    """Publish the worker's resolved per-DP request limit before warmup."""
    if max_running_requests < 1:
        raise ValueError("batch row capacity requires a positive request limit")
    if _state.layouts_frozen:
        raise RuntimeError("configure batch row capacity before precompile")
    _state.capacity = next_power_of_2(max_running_requests)
    _state.merge_index = None
    _state.warned_capacity = _state.capacity


def register_layout(dtype: torch.dtype, trailing_shape: tuple[int, ...]) -> None:
    """Register an optional per-request tensor layout before scheduler warmup."""
    layout = (dtype, trailing_shape)
    if layout in _state.layouts:
        return
    if _state.layouts_frozen:
        raise RuntimeError("register batch row layouts before precompile")
    _state.layouts.append(layout)


def _capacity(*lengths: int) -> int:
    """The worker request limit, rounded to a power of two (larger if needed)."""
    capacity = _state.capacity
    if capacity is None:
        # Standalone users have no scheduler warmup to publish the resolved cap.
        from sglang.srt.runtime_context import get_parallel, get_schedule

        configured = int(get_schedule().max_running_requests or 0)
        capacity = configured // get_parallel().attn_dp_size
    required = next_power_of_2(max(capacity, *lengths, 1))
    if _state.capacity is not None and required > _state.warned_capacity:
        logger.warning(
            "TPU batch rows exceed warmed capacity %d: lengths=%s, capacity=%d. "
            "These shapes may compile while serving.",
            _state.capacity,
            lengths,
            required,
        )
        _state.warned_capacity = required
    return required


def _pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    if t.shape[0] > rows:
        raise ValueError(f"tensor with {t.shape[0]} rows exceeds capacity {rows}")
    if t.shape[0] == rows:
        return t
    return F.pad(t, (0, 0) * (t.dim() - 1) + (0, rows - t.shape[0]))


def padded_filter_rows(
    t: torch.Tensor, keep_device: torch.Tensor, capacity: int
) -> torch.Tensor:
    """``t[keep_device]`` through a ``capacity``-row copy of ``t``."""
    return _pad_rows(t, capacity).index_select(0, keep_device)


def padded_merge_rows(a: torch.Tensor, b: torch.Tensor, capacity: int) -> torch.Tensor:
    """``torch.cat([a, b])`` through ``capacity``-row copies of both."""
    na, nb = a.shape[0], b.shape[0]
    both = torch.cat([_pad_rows(a, capacity), _pad_rows(b, capacity)])
    key = (na, nb, capacity, str(a.device))
    cached = _state.merge_index
    if cached is None or cached.key != key:
        index = torch.cat(
            [
                torch.arange(na, dtype=torch.int64, device="cpu"),
                torch.arange(nb, dtype=torch.int64, device="cpu") + capacity,
            ]
        ).to(a.device)
        # Retain only the latest read-only index, shared by every merged field.
        cached = _MergeIndex(key, index)
        _state.merge_index = cached
    return both.index_select(0, cached.tensor)


def filter_rows(t: torch.Tensor, keep_device: torch.Tensor) -> torch.Tensor:
    """``t[keep_device]`` for an int64 index tensor on ``t``'s device."""
    if t.device.type != "tpu":
        return t[keep_device]
    return padded_filter_rows(
        t, keep_device, _capacity(t.shape[0], keep_device.shape[0])
    )


def merge_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.cat([a, b])`` along the request dimension."""
    if a.device.type != "tpu":
        return torch.cat([a, b])
    return padded_merge_rows(a, b, _capacity(a.shape[0] + b.shape[0]))


def merge_optional_rows(
    left_value: torch.Tensor | None,
    left_valid: torch.Tensor | None,
    left_size: int,
    right_value: torch.Tensor | None,
    right_valid: torch.Tensor | None,
    right_size: int,
    *,
    validity_device: torch.device | str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Merge optional row state, marking zero-filled missing rows invalid."""
    if left_value is None and right_value is None:
        return None, None

    exemplar = left_value if left_value is not None else right_value
    assert exemplar is not None
    valid_device = validity_device or exemplar.device
    # Host construction avoids a device zeros/ones program for each batch size.
    if left_value is None:
        left_value = torch.zeros(
            (left_size, *exemplar.shape[1:]), dtype=exemplar.dtype, device="cpu"
        ).to(exemplar.device)
        left_valid = torch.zeros(left_size, dtype=torch.bool, device="cpu").to(
            valid_device
        )
    elif left_valid is None:
        left_valid = torch.ones(left_size, dtype=torch.bool, device="cpu").to(
            valid_device
        )
    if right_value is None:
        right_value = torch.zeros(
            (right_size, *exemplar.shape[1:]), dtype=exemplar.dtype, device="cpu"
        ).to(exemplar.device)
        right_valid = torch.zeros(right_size, dtype=torch.bool, device="cpu").to(
            valid_device
        )
    elif right_valid is None:
        right_valid = torch.ones(right_size, dtype=torch.bool, device="cpu").to(
            valid_device
        )

    return merge_rows(left_value, right_value), merge_rows(left_valid, right_valid)


def precompile(device: torch.device) -> None:
    """Compile the merge/filter programs for every batch size at startup.

    Each program depends on one length up to the capacity, so this is a fixed
    set (capacity x layouts x 3), written to the persistent compile cache once
    and loaded by later launches. Serving then compiles none of them, even for
    a batch size the traffic reaches for the first time.
    """
    capacity = _state.capacity
    if capacity is None:
        raise ValueError("configure_capacity must run before batch row warmup")
    _state.layouts_frozen = True
    start = time.perf_counter()
    layouts = tuple(_state.layouts)
    logger.info(
        "TPU batch rows warmup: capacity=%d, lengths=%d, layouts=%d",
        capacity,
        capacity + 1,
        len(layouts),
    )
    # Built on the host: device zeros() or slices would compile per length.
    empty_rows = [
        torch.zeros((0, *trailing), dtype=dtype, device="cpu").to(device)
        for dtype, trailing in layouts
    ]
    for n in range(capacity + 1):
        keep = torch.arange(n, dtype=torch.int64, device="cpu").to(device)
        for (dtype, trailing), empty in zip(layouts, empty_rows):
            rows = torch.zeros((n, *trailing), dtype=dtype, device="cpu").to(device)
            padded_filter_rows(rows, keep, capacity)
            padded_merge_rows(rows, empty, capacity)
    # Merging request state can promote int32 to int64 or float32 to float64.
    # These cat variants have fixed operand shapes and need warming only once.
    fixed_rows = {
        (dtype, trailing): torch.zeros(
            (capacity, *trailing), dtype=dtype, device="cpu"
        ).to(device)
        for dtype, trailing in layouts
    }
    for dtype_a, dtype_b in (
        (torch.int32, torch.int64),
        (torch.float32, torch.float64),
    ):
        for dtype, trailing in layouts:
            if dtype == dtype_a and (dtype_b, trailing) in fixed_rows:
                a = fixed_rows[(dtype_a, trailing)]
                b = fixed_rows[(dtype_b, trailing)]
                torch.cat([a, b])
                torch.cat([b, a])
    torch.get_device_module(device).synchronize()
    if device.type == "tpu":
        from sglang.srt.hardware_backend.tpu.compilation.recompilation_checker import (
            snapshot_recompilation_baseline,
        )

        snapshot_recompilation_baseline()
    logger.info("TPU batch rows warmup finished in %.3fs", time.perf_counter() - start)
