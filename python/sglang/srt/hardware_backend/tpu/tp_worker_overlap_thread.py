"""Thread-based overlap worker for TPU.

PJRT dispatch from torch_tpu is already async at the C++ level — calling
a JIT-compiled function returns an unmaterialized tensor immediately.
But materializing the result on host (``.tolist()`` / ``.cpu()`` /
``.item()``) blocks the calling thread. By running the forward and the
host-side resolution in a *separate* Python thread, the main scheduler
thread is free to keep building / submitting the next batch while the
TPU executes the current one.

This mirrors the pattern in ``sglang-jax``'s ``ModelWorkerClient`` and
the historical SGLang ``TpModelWorkerClient`` — specialized to TPU
because torch_tpu's ``torch.tpu.Stream`` is a stub and SGLang's
CUDA-stream-based ``event_loop_overlap`` does not apply.

Cross-step token dependency is handled with the same negative-placeholder
mechanism as the CUDA path (:class:`FutureMap`): the scheduler hands the
worker thread a ``FutureIndices`` along with each batch; the worker runs
``resolve_future`` (device op, no host sync) to rewrite input_ids in
place from the buffer, then stores its own ``next_token_ids`` back into
the buffer for the next step. Both reads and writes stay on device.
"""

from __future__ import annotations

import logging
import signal
import threading
from queue import Queue
from typing import TYPE_CHECKING, Any, Optional, Tuple

import psutil
import torch

from sglang.srt.managers.tp_worker import TpModelWorker

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureIndices, FutureMap
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


# NOTE: we deliberately do NOT call ``.cpu()`` on the result inside the
# worker thread. ``.cpu()`` on torch_tpu blocks until the device→host
# transfer completes — if we did that here, the worker thread would not
# be free to dispatch the next batch's forward until the current one is
# fully materialized, defeating the entire point of the threaded path.
# Instead we hand back unmaterialized device tensors; the scheduler
# thread's existing ``.tolist()`` calls inside ``process_batch_result``
# block on host, and during that block PJRT is free to execute the next
# (already-dispatched) forward on device.


class TpuTpModelWorkerClient:
    """Async wrapper around :class:`TpModelWorker` for TPU overlap.

    Attribute access falls through to the wrapped worker via
    ``__getattr__`` so we satisfy whatever surface the rest of the
    scheduler/code expects without re-listing every method.
    """

    def __init__(self, **worker_kwargs):
        self.worker = TpModelWorker(**worker_kwargs)
        self.device = self.worker.device
        self.max_running_requests = self.worker.max_running_requests

        # Set by the scheduler once it knows the running-batch capacity.
        self._future_map: Optional["FutureMap"] = None

        self._input_queue: "Queue[Optional[Tuple[Any, Any]]]" = Queue()
        self._output_queue: "Queue[Any]" = Queue()

        self._parent_process = psutil.Process().parent()
        self._forward_thread = threading.Thread(
            target=self._forward_thread_entry,
            daemon=True,
            name="sglang-tpu-forward",
        )
        self._forward_thread.start()

    def __getattr__(self, name: str):
        # Pass-through proxy for anything we don't override. ``worker``
        # is set via ``object.__setattr__`` in ``__init__`` so it's in
        # ``__dict__`` and won't recurse.
        return getattr(self.worker, name)

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def set_future_map(self, future_map: "FutureMap") -> None:
        self._future_map = future_map

    # ------------------------------------------------------------------
    # Forward thread
    # ------------------------------------------------------------------

    def _forward_thread_entry(self):
        try:
            self._forward_thread_loop()
        except Exception:
            logger.exception(
                "TPU forward thread crashed; signalling parent to exit."
            )
            if self._parent_process is not None:
                self._parent_process.send_signal(signal.SIGQUIT)

    def _forward_thread_loop(self):
        import os
        import time as _time

        _PROFILE = os.environ.get("SGLANG_TPU_OVERLAP_PROFILE") == "1"
        _phases = {"wait_input": 0.0, "resolve": 0.0, "forward": 0.0, "store": 0.0, "put": 0.0}
        _count = 0
        _next_dump = _time.perf_counter() + 5.0

        while True:
            t_in_start = _time.perf_counter() if _PROFILE else 0.0
            payload = self._input_queue.get()
            t_in_end = _time.perf_counter() if _PROFILE else 0.0
            if payload is None:
                break
            model_worker_batch, future_indices = payload

            # Resolve negative placeholders in input_ids using the
            # buffer that the previous step's ``store_to_map`` wrote to.
            # Pure device op — no host sync.
            if self._future_map is not None:
                self._future_map.resolve_future(model_worker_batch)
            t_resolve = _time.perf_counter() if _PROFILE else 0.0

            batch_result = self.worker.forward_batch_generation(model_worker_batch)
            t_fwd = _time.perf_counter() if _PROFILE else 0.0

            # Publish our tokens into the buffer for the NEXT step's
            # resolve_future to pick up.
            if self._future_map is not None and future_indices is not None:
                self._future_map.store_to_map(future_indices, batch_result)
            t_store = _time.perf_counter() if _PROFILE else 0.0

            # Hand back unmaterialized device tensors — the scheduler
            # thread blocks on host inside ``process_batch_result``'s
            # ``.tolist()`` calls. That blocking overlaps with the next
            # forward, which is already queued on PJRT.
            self._output_queue.put(batch_result)
            t_put = _time.perf_counter() if _PROFILE else 0.0

            if _PROFILE:
                _phases["wait_input"] += t_in_end - t_in_start
                _phases["resolve"] += t_resolve - t_in_end
                _phases["forward"] += t_fwd - t_resolve
                _phases["store"] += t_store - t_fwd
                _phases["put"] += t_put - t_store
                _count += 1
                if t_put >= _next_dump and _count > 0:
                    total = sum(_phases.values())
                    logger.info(
                        "[TPU worker profile] %d iters in %.3fs: "
                        "wait_input=%.1f%% resolve=%.1f%% forward=%.1f%% "
                        "store=%.1f%% put=%.1f%% | mean_iter=%.3fms",
                        _count, total,
                        100 * _phases["wait_input"] / total,
                        100 * _phases["resolve"] / total,
                        100 * _phases["forward"] / total,
                        100 * _phases["store"] / total,
                        100 * _phases["put"] / total,
                        1000 * total / _count,
                    )
                    for k in _phases:
                        _phases[k] = 0.0
                    _count = 0
                    _next_dump = t_put + 5.0

    # ------------------------------------------------------------------
    # Scheduler-facing API
    # ------------------------------------------------------------------

    def submit_forward(
        self,
        model_worker_batch: "ModelWorkerBatch",
        future_indices: Optional["FutureIndices"],
    ) -> None:
        """Enqueue a batch for asynchronous forward + host resolution."""
        self._input_queue.put((model_worker_batch, future_indices))

    def resolve_last_batch_result(self) -> "GenerationBatchResult":
        """Block until the next pending result is fully on host."""
        return self._output_queue.get()

    def shutdown(self) -> None:
        self._input_queue.put(None)
