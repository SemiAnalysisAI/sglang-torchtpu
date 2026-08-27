"""Native torch_tpu data plane for unified HiCache pool groups."""

from __future__ import annotations

from sglang.srt.managers.cache_controller import BatchDoneCounter, HiCacheAck
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)


class TpuHybridCacheController(HybridCacheController):
    """Transfer every unified-cache pool with torch_tpu batch operations.

    torch_tpu has no usable per-layer transfer streams. Device-to-host copies
    therefore gather each pool on the scheduler thread and drain the staged
    tensors on a dedicated worker. Host-to-device copies stage each pool on a
    worker and commit every scatter together at the next scheduler-thread gate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sglang.srt.hardware_backend.tpu.kv_copy_worker import TpuKvCopyWorker

        self.layer_done_counter = BatchDoneCounter()
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)
        self._load_worker = TpuKvCopyWorker("hybrid-kv-offload-h2d")
        self._backup_worker = TpuKvCopyWorker("hybrid-kv-offload-d2h")
        self._backup_inflight = None

    def register_host_pool_entry(self, entry) -> None:
        super().register_host_pool_entry(entry)
        if entry.name in (PoolName.DRAFT, PoolName.DRAFT_SWA):
            entry.device_pool.register_layer_transfer_counter(self.layer_done_counter)

    def reset(self):
        if self._backup_inflight is not None:
            self._backup_inflight.synchronize()
            self._backup_inflight = None
        self._load_worker.submit(lambda: None).synchronize()
        self._backup_worker.submit(lambda: None).synchronize()
        self.layer_done_counter.reset()
        super().reset()

    def evict_device(self, device_indices):
        self.layer_done_counter.drain_commits()
        return super().evict_device(device_indices)

    def start_writing(self) -> None:
        if not self.write_queue:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices, pool_transfers = self._move_write_operation(op)
        self.write_queue.clear()

        if self._backup_inflight is not None:
            self._backup_inflight.synchronize()
        self.layer_done_counter.drain_commits()

        transfers = self._l2_transfers(host_indices, device_indices, pool_transfers)
        staged = [
            (
                transfer,
                transfer.host_pool.backup_gather(
                    transfer.device_pool, transfer.device_indices
                ),
            )
            for transfer in transfers
        ]

        def _drain():
            for transfer, gathered in staged:
                if gathered is not None:
                    transfer.host_pool.backup_copy_d2h(gathered, transfer.host_indices)

        done = self._backup_worker.submit(_drain)
        self._backup_inflight = done
        self.ack_write_queue.append(
            HiCacheAck(
                start_event=done,
                finish_event=done,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
            )
        )

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, pool_transfers = self._move_op_indices(op)
        self.load_queue.clear()
        transfers = self._l2_load_transfers(
            host_indices, device_indices, pool_transfers
        )
        staged = []

        def _stage():
            for transfer in transfers:
                staged.append(
                    (
                        transfer,
                        transfer.host_pool.load_stage(
                            transfer.device_pool,
                            transfer.host_indices,
                            transfer.device_indices,
                        ),
                    )
                )

        def _commit():
            for transfer, loaded in staged:
                transfer.host_pool.load_commit(transfer.device_pool, loaded)

        worker_done = self._load_worker.submit(_stage)
        self.layer_done_counter.set_event(producer_id, worker_done)
        self.layer_done_counter.add_commit(worker_done, _commit)
        self.ack_load_queue.append(
            HiCacheAck(
                start_event=worker_done,
                finish_event=worker_done,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
            )
        )
        return producer_id
