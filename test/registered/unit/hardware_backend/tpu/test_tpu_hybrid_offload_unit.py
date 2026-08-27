"""CPU-only tests for the native torch_tpu unified HiCache controller."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.test.test_utils import CustomTestCase


class _FakeDevicePool:
    def __init__(self):
        self.layer_transfer_counter = None

    def register_layer_transfer_counter(self, counter):
        self.layer_transfer_counter = counter


class _FakeHostPool:
    layout = "page_first_fused"
    page_size = 4
    device = "cpu"
    size = 64
    logical_size = 64
    can_use_write_back_jit = False
    size_per_token = 2

    def __init__(self):
        self.gathers = []
        self.backups = []
        self.stages = []
        self.commits = []

    def backup_gather(self, device_pool, device_indices):
        staged = object()
        self.gathers.append((device_pool, device_indices, staged))
        return staged

    def backup_copy_d2h(self, staged, host_indices):
        self.backups.append((staged, host_indices))

    def load_stage(self, device_pool, host_indices, device_indices):
        staged = object()
        self.stages.append((device_pool, host_indices, device_indices, staged))
        return staged

    def load_commit(self, device_pool, staged):
        self.commits.append((device_pool, staged))


def _entry(name):
    return PoolEntry(
        name=name,
        host_pool=_FakeHostPool(),
        device_pool=_FakeDevicePool(),
        layer_mapper=lambda layer_id: layer_id,
        is_primary_index_anchor=name == PoolName.KV,
    )


def _make_controller():
    from sglang.srt.hardware_backend.tpu.hybrid_cache_controller import (
        TpuHybridCacheController,
    )
    from sglang.srt.hardware_backend.tpu.kv_copy_worker import TpuKvCopyWorker

    ctrl = object.__new__(TpuHybridCacheController)
    anchor = _entry(PoolName.KV)
    sidecar = _entry(PoolName.DRAFT)
    ctrl.mem_pool_host = HostPoolGroup([anchor, sidecar])
    ctrl.mem_pool_device = anchor.device_pool
    ctrl.io_backend = "torch_tpu"
    ctrl.device = "cpu"
    ctrl.write_queue = []
    ctrl.load_queue = []
    ctrl.ack_write_queue = []
    ctrl.ack_load_queue = []
    ctrl.layer_done_counter = BatchDoneCounter()
    ctrl._load_worker = TpuKvCopyWorker("test-hybrid-tpu-h2d")
    ctrl._backup_worker = TpuKvCopyWorker("test-hybrid-tpu-d2h")
    ctrl._backup_inflight = None
    ctrl.enable_storage = False
    ctrl.storage_backend = None
    ctrl.storage_backend_type = None
    ctrl.enable_storage_metrics = False
    ctrl.extra_host_mem_release_queues = {
        PoolName.DRAFT: Queue(),
    }
    return ctrl


def _operation():
    return CacheOperation(
        host_indices=torch.arange(8, dtype=torch.int64),
        device_indices=torch.arange(8, dtype=torch.int64),
        node_id=1,
        pool_transfers=[
            PoolTransfer(
                name=PoolName.DRAFT,
                host_indices=torch.arange(8, dtype=torch.int64),
                device_indices=torch.arange(8, dtype=torch.int64),
                indices_from_pool=PoolName.KV,
            )
        ],
    )


class TestNativeTpuHybridTransfers(CustomTestCase):
    def setUp(self):
        self.ctrl = _make_controller()
        self.addCleanup(self.ctrl._load_worker.shutdown)
        self.addCleanup(self.ctrl._backup_worker.shutdown)

    def test_write_gathers_and_copies_every_pool(self):
        self.ctrl.write_queue.append(_operation())
        self.ctrl.start_writing()
        self.ctrl._backup_inflight.synchronize()

        for entry in self.ctrl.mem_pool_host.entries:
            self.assertEqual(len(entry.host_pool.gathers), 1)
            self.assertEqual(len(entry.host_pool.backups), 1)
        (ack,) = self.ctrl.ack_write_queue
        self.assertTrue(ack.finish_event.query())

    def test_load_stages_every_pool_and_commits_on_scheduler_thread(self):
        self.ctrl.load_queue.append(_operation())
        producer_id = self.ctrl.start_loading()
        self.ctrl.layer_done_counter.events[producer_id].synchronize()

        for entry in self.ctrl.mem_pool_host.entries:
            self.assertEqual(len(entry.host_pool.stages), 1)
            self.assertEqual(entry.host_pool.commits, [])

        self.ctrl.layer_done_counter.drain_commits()
        for entry in self.ctrl.mem_pool_host.entries:
            self.assertEqual(len(entry.host_pool.commits), 1)

    def test_draft_sidecar_uses_the_shared_batch_counter(self):
        entry = _entry(PoolName.DRAFT_SWA)
        self.ctrl.register_host_pool_entry(entry)
        self.assertIs(
            entry.device_pool.layer_transfer_counter,
            self.ctrl.layer_done_counter,
        )


class TestNativeTpuHybridDispatch(CustomTestCase):
    def test_torch_tpu_backend_selects_native_hybrid_controller(self):
        from sglang.srt.hardware_backend.tpu.hybrid_cache_controller import (
            TpuHybridCacheController,
        )

        with patch.object(
            TpuHybridCacheController, "__init__", return_value=None
        ) as init:
            ctrl = HybridCacheController.create_for_backend(io_backend="torch_tpu")
        self.assertIsInstance(ctrl, TpuHybridCacheController)
        init.assert_called_once()

    def test_flat_kv_builder_uses_backend_factory(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

        params = SimpleNamespace(
            page_size=16,
            token_to_kv_pool_allocator=object(),
            tp_cache_group=object(),
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
            mtp_draft_device_pools=(),
        )
        memory = SimpleNamespace(
            hicache_write_policy="write_through",
            hicache_io_backend="torch_tpu",
        )
        server_args = SimpleNamespace(hicache_host_memory_mode="cache")
        host_group = object()
        controller = object()

        with (
            patch.object(
                hybrid_pool_assembler,
                "build_kv_only_group",
                return_value=host_group,
            ),
            patch.object(hybrid_pool_assembler, "get_memory", return_value=memory),
            patch.object(
                HybridCacheController,
                "create_for_backend",
                return_value=controller,
            ) as create,
        ):
            built_group, built_controller = hybrid_pool_assembler.build_kv_only_stack(
                params=params,
                server_args=server_args,
                kv_pool=object(),
                full_layer_mapping={0: 0},
                load_cache_event=object(),
                storage_backend=None,
                use_mla=False,
            )

        self.assertIs(built_group, host_group)
        self.assertIs(built_controller, controller)
        self.assertEqual(create.call_args.kwargs["io_backend"], "torch_tpu")


if __name__ == "__main__":
    unittest.main()
