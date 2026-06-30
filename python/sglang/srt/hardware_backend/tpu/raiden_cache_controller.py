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

from sglang.srt.hardware_backend.tpu import offload_trace as _ot
from sglang.srt.hardware_backend.tpu.raiden_transfer import NOOP_EVENT
from sglang.srt.managers.cache_controller import (
    BatchDoneCounter,
    CacheOperation,
    HiCacheAck,
    HiCacheController,
)


class TpuRaidenHiCacheController(HiCacheController):
    """HiCache controller whose transfers go through raiden DMA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # raiden gates the whole forward at once (no per-layer device streams),
        # like the torch_tpu path. Swap the base's per-layer LayerDoneCounter for
        # a BatchDoneCounter and re-register it, so CompiledForwardShim's
        # whole-batch ``wait()`` resolves against the reload event.
        self.layer_done_counter = BatchDoneCounter()
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

    def reset(self):
        self.layer_done_counter.reset()
        super().reset()

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

        with _ot.trace("offload", "start_writing"):
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

        with _ot.trace("offload", "start_loading"):
            producer_id = self.layer_done_counter.update_producer()
            op = CacheOperation.merge_ops(self.load_queue)
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
            self.load_queue.clear()

            # H2D for all layers in one raiden call, then block until the DMA has
            # landed (poll-only; never Await). TPU's wait_event is a no-op and the
            # consuming forward runs this same step, so the load must complete here.
            finish_event = self.mem_pool_host.load_all_layer(
                self.mem_pool_device, host_indices, device_indices
            )
            with _ot.trace("offload", "h2d_wait"):
                finish_event.synchronize()

            # Hand the already-landed event to the whole-batch gate; the consuming
            # forward's CompiledForwardShim wait() then resolves immediately.
            self.layer_done_counter.set_event(producer_id, finish_event)
            self.ack_load_queue.append(
                HiCacheAck(
                    start_event=NOOP_EVENT,
                    finish_event=finish_event,
                    node_ids=op.node_ids,
                )
            )
            return producer_id
