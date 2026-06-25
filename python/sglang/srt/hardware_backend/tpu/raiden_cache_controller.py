"""HiCache controller for TPU, transferring KV via raiden instead of CUDA.

``HiCacheController`` drives KV transfers on device streams and CUDA events,
neither of which exist on torch_tpu. This subclass overrides only the
transfer-issuing methods (``start_writing`` / ``start_loading`` /
``move_indices``) to route through raiden's DMA; queues, ack draining, and write
policy are inherited. D2H write-back is async (polled by
``HiRadixCache.writing_check``). H2D load-back needs a gate because raiden's H2D
bypasses XLA dependency tracking; ``TPU_HICACHE_LOAD_MODE`` selects it:
``"poll"`` (default) blocks on the future in ``start_loading``; ``"deferral"``
issues the H2D async and holds the consuming request out of the running batch
(``HiRadixCache.async_load_back_ready``) until the future is ready, overlapping
the DMA with other requests.
"""

from __future__ import annotations

import os

from sglang.srt.hardware_backend.tpu.raiden_transfer import NOOP_EVENT
from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheAck,
    HiCacheController,
)

_LOAD_MODE = os.environ.get("TPU_HICACHE_LOAD_MODE", "poll")


class TpuRaidenHiCacheController(HiCacheController):
    """HiCache controller whose transfers go through raiden DMA."""

    # "deferral" defers the request at the scheduler; "poll" blocks in
    # start_loading. Both gate the H2D, which is untracked by XLA.
    async_load_back = _LOAD_MODE == "deferral"

    def move_indices(self, host_indices, device_indices):
        # raiden addresses whole pages; the token->page conversion happens in
        # the host pool. Keep both index tensors as-is: no stream moves, no
        # ``.is_cuda`` branching, no host/device relocation.
        return host_indices, device_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return
        if self.has_draft:
            raise NotImplementedError(
                "Draft KV offload is not supported on the TPU raiden path."
            )

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.write_queue.clear()

        # D2H for all layers in one raiden call; the returned future-event is
        # polled by HiRadixCache.writing_check before the device slot is freed.
        finish_event = self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device, host_indices, device_indices, self.io_backend
        )
        self.ack_write_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=finish_event,
                node_ids=op.node_ids,
            )
        )

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1
        if self.has_draft:
            raise NotImplementedError(
                "Draft KV offload is not supported on the TPU raiden path."
            )

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()

        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        # H2D for all layers in one raiden call.
        finish_event = self.mem_pool_host.load_all_layer(
            self.mem_pool_device, host_indices, device_indices
        )
        if not self.async_load_back:
            # poll mode: the consuming forward runs this same step, so block on
            # the DMA before it reads KV (poll-only; never Await).
            finish_event.synchronize()

        # Mark every layer complete so the LayerDoneCounter contract (which
        # asserts the prior producer's finish_event is ready before reuse)
        # holds; the per-layer gate itself is decorative on TPU.
        for i in range(self.layer_num):
            producer_event.complete(i)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id
