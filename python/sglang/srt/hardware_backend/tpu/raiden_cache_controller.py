"""HiCache controller for TPU, transferring KV via raiden instead of CUDA.

``HiCacheController`` drives KV transfers on device streams and CUDA events,
neither of which exist on torch_tpu. This subclass overrides only the
transfer-issuing methods (``start_writing`` / ``start_loading`` /
``move_indices``) to route through raiden's DMA; queues, ack draining, and write
policy are inherited. D2H write-back is async (polled by
``HiRadixCache.writing_check``); H2D load-back is poll-to-complete --
``start_loading`` blocks on the future before the consuming forward reads KV,
since raiden's H2D bypasses XLA dependency tracking.
"""

from __future__ import annotations

from sglang.srt.hardware_backend.tpu.raiden_transfer import NOOP_EVENT
from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheAck,
    HiCacheController,
)


class TpuRaidenHiCacheController(HiCacheController):
    """HiCache controller whose transfers go through raiden DMA."""

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
        # Block until the DMA has landed (poll-only; never Await). TPU's
        # wait_event is a no-op and the consuming forward runs this same step,
        # so we cannot defer this without a request-level gate.
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
