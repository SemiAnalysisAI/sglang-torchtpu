"""Merge and filter the per-request tensors of a batch through a backend.

``ScheduleBatch`` and ``SamplingBatchInfo`` merge a prefill batch into the
running batch and drop finished requests with ``merge_rows`` and
``filter_rows``. The default backend runs ``torch.cat`` and ``tensor[keep]``.
A platform that compiles one program per eager-op input shape can register a
backend that bounds those shapes (``register_batch_rows_backend``, typically
from ``init_backend()``). The scheduler calls ``warmup_batch_rows`` once at
startup, so such a backend can compile ahead of serving.
"""

from __future__ import annotations

import torch


class BatchRowsBackend:
    """The plain row operations; platform backends subclass and override."""

    def merge_rows(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``torch.cat([a, b])`` along the request dimension."""
        return torch.cat([a, b])

    def filter_rows(self, t: torch.Tensor, keep_device: torch.Tensor) -> torch.Tensor:
        """``t[keep_device]`` for an int64 index tensor on ``t``'s device."""
        return t[keep_device]

    def warmup(self, max_running_requests: int, device: torch.device) -> None:
        """Prepare for serving, given the worker's resolved request limit."""


_backend = BatchRowsBackend()


def register_batch_rows_backend(backend: BatchRowsBackend) -> None:
    """Route every later merge, filter and warmup through ``backend``."""
    if not isinstance(backend, BatchRowsBackend):
        raise TypeError(f"expected a BatchRowsBackend, got {type(backend).__name__}")
    global _backend
    _backend = backend


def merge_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _backend.merge_rows(a, b)


def filter_rows(t: torch.Tensor, keep_device: torch.Tensor) -> torch.Tensor:
    return _backend.filter_rows(t, keep_device)


def warmup_batch_rows(max_running_requests: int, device: torch.device) -> None:
    """Called once by the scheduler after its workers' graphs are initialized."""
    _backend.warmup(max_running_requests, device)
