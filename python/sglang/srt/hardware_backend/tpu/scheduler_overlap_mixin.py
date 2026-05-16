"""TPU overlap scheduling mixin.

Provides ``event_loop_overlap_tpu``, a single-producer / single-consumer
overlap loop that uses :class:`TpuTpModelWorkerClient` to run each
forward + host-resolution in a dedicated thread. While that thread is
blocked materializing batch N's tokens on host, the scheduler thread
builds and submits batch N+1, whose forward is dispatched into PJRT's
async queue and runs back-to-back on the device.

Why not reuse the CUDA-stream-based ``event_loop_overlap``?
torch_tpu's ``torch.tpu.Stream`` is a stub (see
``torch_tpu._internal.stream.streams``): ``wait_event`` is a no-op,
``wait_stream`` reduces to it, and the C++ runtime serializes work on a
single PJRT queue regardless of which "stream" the host code is on.
Combined with SGLang's ``FutureMap`` (which assumes per-stream
dependency tracking), the CUDA path either hangs or races on TPU.
The thread approach side-steps all of that — PJRT's async dispatch
already provides device-side concurrency, we just need to ensure the
blocking ``.cpu()`` call doesn't sit on the scheduler's critical path.

Cross-step token dependency uses :class:`FutureMap` exactly as the CUDA
path does: each iteration allocates a fresh ``FutureIndices``, sets
``batch.output_ids = -future_indices.indices`` (negative placeholders
that ``prepare_for_decode`` then carries into the next ``input_ids``),
and hands ``future_indices`` to the worker thread so it can call
``store_to_map`` once the real tokens are computed. All buffer reads
and writes stay on device.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Deque, Tuple

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


class SchedulerTpuOverlapMixin:
    """Mixin that adds TPU thread-based overlap scheduling to
    :class:`Scheduler`.
    """

    @DynamicGradMode()
    def event_loop_overlap_tpu(self: "Scheduler"):
        """Scheduler loop with TPU thread overlap.

        Each iteration:
          1. Receive new requests + bookkeeping.
          2. Build the next batch.
          3. Submit it to the worker thread (non-blocking).
          4. Block on the previous batch's resolved result and run
             ``process_batch_result``.
        """
        self.result_queue: Deque[Tuple[ScheduleBatch, "GenerationBatchResult"]] = (
            deque()
        )

        # Per-phase wall-time accumulation, for the host-vs-overlap audit.
        # Toggle via env var to keep the steady-state path branch-light.
        import os
        import time as _time

        _PROFILE = os.environ.get("SGLANG_TPU_OVERLAP_PROFILE") == "1"
        _phases = {
            "recv": 0.0,
            "process_input": 0.0,
            "get_next_batch": 0.0,
            "submit": 0.0,
            "queue_wait": 0.0,
            "d2h_wait": 0.0,
            "post_d2h": 0.0,
            "on_idle": 0.0,
            "iter_total": 0.0,
        }
        _iter_count = 0
        _next_dump = _time.perf_counter() + 5.0

        while True:
            t0 = _time.perf_counter() if _PROFILE else 0.0

            recv_reqs = self.recv_requests()
            t1 = _time.perf_counter() if _PROFILE else 0.0

            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            t2 = _time.perf_counter() if _PROFILE else 0.0

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            t3 = _time.perf_counter() if _PROFILE else 0.0

            if batch:
                self._submit_batch_tpu(batch)
            t4 = _time.perf_counter() if _PROFILE else 0.0

            if self.last_batch:
                # Split wait vs process so we can see how much of the
                # scheduler's host time is genuinely blocked on the
                # worker (good — that's where the device-side overlap
                # lives) vs spent in post-processing (Python overhead).
                if _PROFILE:
                    _qw, _d2h, _post = self._pop_and_process_tpu_timed()
                    _phases["queue_wait"] += _qw
                    _phases["d2h_wait"] += _d2h
                    _phases["post_d2h"] += _post
                else:
                    self._pop_and_process_tpu()
            elif batch is None:
                self.on_idle()
            t5 = _time.perf_counter() if _PROFILE else 0.0

            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

            if _PROFILE:
                _phases["recv"] += t1 - t0
                _phases["process_input"] += t2 - t1
                _phases["get_next_batch"] += t3 - t2
                _phases["submit"] += t4 - t3
                _phases["iter_total"] += t5 - t0
                if batch is None and not self.last_batch:
                    _phases["on_idle"] += t5 - t4
                _iter_count += 1
                if t5 >= _next_dump and _iter_count > 0:
                    total = _phases["iter_total"]
                    logger.info(
                        "[TPU overlap profile] %d iters in %.3fs: "
                        "recv=%.1f%% process_input=%.1f%% get_next=%.1f%% "
                        "submit=%.1f%% queue_wait=%.1f%% d2h_wait=%.1f%% "
                        "post_d2h=%.1f%% on_idle=%.1f%% | mean_iter=%.3fms",
                        _iter_count, total,
                        100 * _phases["recv"] / total,
                        100 * _phases["process_input"] / total,
                        100 * _phases["get_next_batch"] / total,
                        100 * _phases["submit"] / total,
                        100 * _phases["queue_wait"] / total,
                        100 * _phases["d2h_wait"] / total,
                        100 * _phases["post_d2h"] / total,
                        100 * _phases["on_idle"] / total,
                        1000 * total / _iter_count,
                    )
                    for k in _phases:
                        _phases[k] = 0.0
                    _iter_count = 0
                    _next_dump = t5 + 5.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _submit_batch_tpu(self: "Scheduler", batch: ScheduleBatch):
        """Build the model worker batch, set up future placeholders, and
        send to the worker thread. Mirrors the relevant subset of
        ``Scheduler.run_batch`` but without any stream-context machinery.
        """
        self.forward_ct += 1
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(
                "Scheduler.run_batch sleep %ss (tpu-overlap)", self.forward_sleep_time
            )
            time.sleep(self.forward_sleep_time)

        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_start_time")

        mwb = batch.get_model_worker_batch()
        self.record_batch_in_overlap(mwb)

        # Sampling info is mutated in-place during forward; the scheduler
        # may also modify the live ``mwb.sampling_info`` before the worker
        # thread reads it. Snapshot it for the forward.
        mwb.sampling_info = mwb.sampling_info.copy_for_forward()

        bs = len(mwb.seq_lens)
        future_indices = self.future_map.alloc_future_indices(bs)

        # output_ids on the live batch is what prepare_for_decode will
        # consume in the next iteration as input_ids. Stash placeholders
        # so the NEXT iter's input_ids contain ``-future_indices.indices``,
        # which the worker thread will resolve_future() against the
        # buffer this batch writes via store_to_map().
        batch.output_ids = -future_indices.indices

        # Save the per-req extend bookkeeping the output processor needs
        # while the request objects are still in their pre-result state.
        if batch.return_logprob:
            extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
            extend_logprob_start_len_per_req = [
                req.extend_logprob_start_len for req in batch.reqs
            ]
        else:
            extend_input_len_per_req = None
            extend_logprob_start_len_per_req = None

        batch_copy = batch.copy()

        self.tp_worker.submit_forward(mwb, future_indices)

        # Stash extend-time per-req info on a slot in the result queue;
        # the resolved GenerationBatchResult coming back from the worker
        # won't have it yet so we patch it in post-pop.
        self.result_queue.append(
            (batch_copy, (extend_input_len_per_req, extend_logprob_start_len_per_req))
        )

    def _pop_and_process_tpu(self: "Scheduler"):
        last_batch_copy, (
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = self.result_queue.popleft()
        result = self.tp_worker.resolve_last_batch_result()
        result.extend_input_len_per_req = extend_input_len_per_req
        result.extend_logprob_start_len_per_req = extend_logprob_start_len_per_req
        self.process_batch_result(last_batch_copy, result)

    def _pop_and_process_tpu_timed(self: "Scheduler"):
        """Like ``_pop_and_process_tpu`` but returns
        ``(queue_wait, d2h_wait, post_d2h_python)`` split. Used when
        ``SGLANG_TPU_OVERLAP_PROFILE=1``.

        Forces materialization of ``next_token_ids`` first via ``.cpu()``
        — that's the device-sync wait. Everything after is pure Python
        post-processing (.tolist() on a CPU tensor is microseconds).
        """
        import time as _time
        import torch

        last_batch_copy, (
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = self.result_queue.popleft()

        t0 = _time.perf_counter()
        result = self.tp_worker.resolve_last_batch_result()
        t1 = _time.perf_counter()

        # Force the D2H sync explicitly so we can measure the device-wait
        # separately from the Python post-processing.
        if (
            result.next_token_ids is not None
            and isinstance(result.next_token_ids, torch.Tensor)
            and result.next_token_ids.device.type == "tpu"
        ):
            result.next_token_ids = result.next_token_ids.cpu()
        t2 = _time.perf_counter()

        result.extend_input_len_per_req = extend_input_len_per_req
        result.extend_logprob_start_len_per_req = extend_logprob_start_len_per_req
        self.process_batch_result(last_batch_copy, result)
        t3 = _time.perf_counter()

        # Returned: (queue_wait, d2h_wait, post_d2h_python)
        # Callers map this into wait_worker / process_result phases.
        return t1 - t0, t2 - t1, t3 - t2
