"""Warm eager TPU programs without waiting after each shape."""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


@contextmanager
def eager_warmup(name: str) -> Iterator[Callable[[torch.Tensor], None]]:
    """Submit the usual D2H boundaries, then join before releasing their buffers."""
    stream = torch.tpu.current_stream()
    host_outputs: list[torch.Tensor] = []
    start = time.perf_counter()

    def materialize(value: torch.Tensor) -> None:
        # D2H materializes views as well as their base buffers, like .cpu().
        # TorchTPU retains the destination pointer; keep its storage until join.
        host_outputs.append(value.to("cpu", non_blocking=True))

    try:
        yield materialize
    finally:
        submitted = time.perf_counter()
        stream.synchronize()
        finished = time.perf_counter()
        logger.info(
            "TPU eager warmup %s: outputs=%d submit=%.3fs wait=%.3fs total=%.3fs",
            name,
            len(host_outputs),
            submitted - start,
            finished - submitted,
            finished - start,
        )
        host_outputs.clear()
