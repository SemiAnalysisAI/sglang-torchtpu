"""Unit tests for submitting HiCache host backups after the next forward launch."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _op(node_id: int) -> CacheOperation:
    indices = torch.tensor([node_id], dtype=torch.int64)
    return CacheOperation(indices, indices, node_id)


def _controller(submitted: list) -> HiCacheController:
    """A controller whose start_writing records the node ids it submits."""
    cc = object.__new__(HiCacheController)
    cc.write_queue = []

    def start_writing():
        if cc.write_queue:
            submitted.append(CacheOperation.merge_ops(cc.write_queue).node_ids)
            cc.write_queue.clear()

    cc.start_writing = start_writing
    return cc


class TestSealedWriting(CustomTestCase):
    def test_submits_only_the_sealed_prefix(self):
        submitted = []
        cc = _controller(submitted)
        cc.write_queue += [_op(1), _op(2)]
        cc.seal_writes()
        cc.write_queue.append(_op(3))

        cc.start_sealed_writing()

        self.assertEqual(submitted, [[1, 2]])
        self.assertEqual([op.node_ids for op in cc.write_queue], [[3]])

    def test_nothing_sealed_submits_nothing(self):
        submitted = []
        cc = _controller(submitted)
        cc.write_queue.append(_op(1))

        cc.start_sealed_writing()

        self.assertEqual(submitted, [])
        self.assertEqual(len(cc.write_queue), 1)

    def test_seal_does_not_outlive_a_full_submit(self):
        submitted = []
        cc = _controller(submitted)
        cc.write_queue.append(_op(1))
        cc.seal_writes()
        cc.start_writing()  # e.g. write-back eviction or idle
        cc.write_queue.append(_op(2))

        cc.start_sealed_writing()

        self.assertEqual(submitted, [[1]])
        self.assertEqual([op.node_ids for op in cc.write_queue], [[2]])

    def test_failed_submit_keeps_unsealed_writes_queued(self):
        """An unsealed write dropped by a failed submit would keep its node locked."""
        cc = _controller([])
        cc.write_queue += [_op(1), _op(2)]
        cc.seal_writes()
        cc.write_queue.append(_op(3))
        cc.start_writing = mock.Mock(side_effect=RuntimeError("transfer failed"))

        with self.assertRaises(RuntimeError):
            cc.start_sealed_writing()

        self.assertEqual(cc.write_queue[-1].node_ids, [3])


def _unified_cache(write_policy="write_through", **overrides) -> UnifiedRadixCache:
    cache = object.__new__(UnifiedRadixCache)
    cache.linker = None
    cache.buffer_pipeline = None
    cache.defer_backup_flush = False
    cache.cache_controller = mock.Mock(write_policy=write_policy)
    for name, value in overrides.items():
        setattr(cache, name, value)
    return cache


class TestUnifiedCacheDeferral(CustomTestCase):
    def test_write_through_opts_in(self):
        cache = _unified_cache()
        self.assertTrue(cache.enable_deferred_backup_flush())
        self.assertTrue(cache.defer_backup_flush)

    def test_ineligible_caches_stay_eager(self):
        for cache in (
            _unified_cache(write_policy="write_back"),
            _unified_cache(buffer_pipeline=object()),
            _unified_cache(linker=object()),
            _unified_cache(cache_controller=None),
        ):
            self.assertFalse(cache.enable_deferred_backup_flush())
            self.assertFalse(cache.defer_backup_flush)

    def test_seal_and_flush_are_noops_until_enabled(self):
        cache = _unified_cache()
        cache.seal_pending_backups()
        cache.flush_sealed_backups()
        cache.cache_controller.seal_writes.assert_not_called()
        cache.cache_controller.start_sealed_writing.assert_not_called()

        cache.enable_deferred_backup_flush()
        cache.seal_pending_backups()
        cache.flush_sealed_backups()
        cache.cache_controller.seal_writes.assert_called_once()
        cache.cache_controller.start_sealed_writing.assert_called_once()

    def test_check_hicache_events_leaves_the_queue_to_the_scheduler(self):
        for deferred in (False, True):
            cache = _unified_cache(enable_storage_metrics=False)
            cache.defer_backup_flush = deferred
            cache._drain_async_work = mock.Mock()
            cache._sync_hicache_ready_counts = mock.Mock(return_value=(0, 0, [], []))
            cache.writing_check = mock.Mock()
            cache.loading_check = mock.Mock()

            with mock.patch.object(
                UnifiedRadixCache,
                "enable_storage",
                new_callable=mock.PropertyMock,
                return_value=False,
            ):
                cache.check_hicache_events()

            self.assertEqual(cache.cache_controller.start_writing.called, not deferred)


class _FakeScheduler:
    """Drives Scheduler.event_loop_overlap over a fixed batch sequence."""

    def __init__(self, batches, defer=True, paused_steps=()):
        self.batches = deque(batches)
        self.paused_steps = deque(paused_steps)
        self.events = []
        self.defer = defer
        self.gracefully_exit = False
        self._engine_paused = False
        self.enable_unified_memory = False
        self.is_generation = False
        self.running_batch = None
        self.last_batch = None
        self.tree_cache = SimpleNamespace(
            flush_sealed_backups=lambda: self.events.append("flush_sealed"),
            flush_pending_backups=lambda: self.events.append("flush_all"),
        )

    def _enable_deferred_backup_flush(self):
        return self.defer

    def ingest_requests(self):
        step = self.paused_steps.popleft() if self.paused_steps else False
        if step == "retract":
            # pause_generation(mode="retract") processes the in-flight batch.
            self.process_batch_result(*self.result_queue.popleft())
            self.last_batch = None
        self._engine_paused = bool(step)

    def _record_scheduler_state_for_paused_engine(self):
        self.events.append("paused")

    def get_next_batch_to_run(self, running_batch, last_batch):
        if not self.batches:
            self.gracefully_exit = True
            batch = None
        else:
            batch = self.batches.popleft()
        return SimpleNamespace(running_batch=None, batch_to_run=batch)

    def is_disable_overlap_for_batch(self, batch, last_batch):
        return batch is not None and batch.disable_overlap

    def run_batch(self, batch):
        self.events.append(f"launch {batch.name}")
        return batch.name

    def _apply_war_barrier(self):
        pass

    def process_batch_result(self, batch, result):
        self.events.append(f"process {batch.name}")

    def on_idle(self):
        self.events.append("idle")


def _batch(name, disable_overlap=False):
    return SimpleNamespace(
        name=name,
        disable_overlap=disable_overlap,
        copy=lambda: SimpleNamespace(name=name),
    )


class TestOverlapLoopPlacement(CustomTestCase):
    def test_sealed_backups_submit_after_each_launch(self):
        sched = _FakeScheduler([_batch("A"), _batch("B"), _batch("C")])
        sched.events.clear()

        Scheduler.event_loop_overlap(sched)

        self.assertEqual(
            sched.events,
            [
                "launch A",
                "flush_sealed",
                "launch B",
                "flush_sealed",  # nothing sealed yet: A is unprocessed
                "process A",
                "launch C",
                "flush_sealed",  # A's backups queue behind C's inputs
                "process B",
                "process C",  # no next batch: idle, submit everything
                "flush_all",
            ],
        )

    def test_pause_submits_sealed_backups(self):
        sched = _FakeScheduler([_batch("A"), _batch("B")], paused_steps=[False, True])

        Scheduler.event_loop_overlap(sched)

        self.assertEqual(
            sched.events,
            [
                "launch A",
                "flush_sealed",
                "flush_sealed",  # paused: KV may be released before resume
                "paused",
                "launch B",
                "flush_sealed",
                "process A",
                "process B",
                "flush_all",
            ],
        )

    def test_retract_pause_submits_every_backup(self):
        """A retract-mode pause must not hold backups until resume: flush_cache
        refuses to run while any backup is outstanding."""
        sched = _FakeScheduler(
            [_batch("A"), _batch("B")], paused_steps=[False, "retract"]
        )

        Scheduler.event_loop_overlap(sched)

        self.assertEqual(
            sched.events[:5],
            ["launch A", "flush_sealed", "process A", "flush_all", "paused"],
        )

    def test_non_overlapped_batch_submits_after_its_launch(self):
        sched = _FakeScheduler([_batch("A"), _batch("B", disable_overlap=True)])

        Scheduler.event_loop_overlap(sched)

        self.assertEqual(
            sched.events,
            [
                "launch A",
                "flush_sealed",
                "process A",  # processed before B launches
                "launch B",
                "flush_sealed",  # A's backups still wait for B's inputs
                "process B",
                "flush_all",
            ],
        )

    def test_eager_mode_never_touches_the_queue(self):
        sched = _FakeScheduler([_batch("A"), _batch("B")], defer=False)

        Scheduler.event_loop_overlap(sched)

        self.assertNotIn("flush_sealed", sched.events)
        self.assertNotIn("flush_all", sched.events)


class _StopLoop(Exception):
    pass


class _FakeDisaggPrefillScheduler(_FakeScheduler):
    """Drives Scheduler.event_loop_overlap_disagg_prefill, which never exits.

    After the batches run out, one idle step runs and then the loop is stopped.
    """

    def __init__(self, batches, **kwargs):
        super().__init__(batches, **kwargs)
        self.waiting_queue = []
        self.chunked_req = None
        self.enable_staging = False
        self.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=lambda: []
        )
        self.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        self.idle_steps = 0

    def get_next_disagg_prefill_batch_to_run(self, running_batch, last_batch):
        if self.batches:
            batch = self.batches.popleft()
        elif self.idle_steps == 0:
            self.idle_steps += 1
            batch = None
        else:
            raise _StopLoop
        return SimpleNamespace(running_batch=None, batch_to_run=batch)

    def process_disagg_prefill_inflight_queue(self):
        self.events.append("release")

    def launch_batch_sample_if_needed(self, batch_result, batch):
        pass


class TestDisaggPrefillLoopPlacement(CustomTestCase):
    def _run(self, sched):
        with self.assertRaises(_StopLoop):
            Scheduler.event_loop_overlap_disagg_prefill(sched)
        return sched.events

    def test_sealed_backups_submit_after_each_launch(self):
        sched = _FakeDisaggPrefillScheduler([_batch("A"), _batch("B")])

        self.assertEqual(
            self._run(sched),
            [
                "launch A",
                "flush_sealed",
                "release",
                "launch B",
                "flush_sealed",
                "process A",
                "release",
                "process B",  # no next batch: idle
                "release",
                "flush_all",  # after the releases, which may queue backups
            ],
        )

    def test_pause_submits_sealed_backups(self):
        sched = _FakeDisaggPrefillScheduler([_batch("A")], paused_steps=[False, True])

        self.assertEqual(
            self._run(sched),
            [
                "launch A",
                "flush_sealed",
                "release",
                "flush_sealed",  # paused: KV may be released before resume
                "paused",
                "process A",
                "release",
                "flush_all",
            ],
        )

    def test_retract_pause_submits_every_backup(self):
        sched = _FakeDisaggPrefillScheduler(
            [_batch("A")], paused_steps=[False, "retract"]
        )

        self.assertEqual(
            self._run(sched)[:5],
            ["launch A", "flush_sealed", "release", "process A", "flush_all"],
        )


class TestRoleSwitchRedispatch(CustomTestCase):
    def test_decode_loop_after_prefill_resumes_eager_backups(self):
        """After a prefill->decode switch, the decode loop never submits deferred
        backups, so queued backups must go out and the cache must flush eagerly."""
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers import scheduler as scheduler_module

        cache = _unified_cache()
        cache.enable_deferred_backup_flush()
        seen = {}
        sched = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            enable_overlap=True,
            defer_backup_flush=True,
            tree_cache=cache,
        )

        def decode_loop():
            seen["sched"] = sched.defer_backup_flush
            seen["cache"] = cache.defer_backup_flush
            seen["flushed"] = cache.cache_controller.start_writing.called

        sched.event_loop_overlap_disagg_decode = decode_loop
        with mock.patch.object(
            scheduler_module, "get_parallel", return_value=SimpleNamespace(pp_size=1)
        ):
            scheduler_module._dispatch_event_loop_once(sched)

        self.assertEqual(seen, {"sched": False, "cache": False, "flushed": True})


class TestEnableDeferredBackupFlush(CustomTestCase):
    def _sched(self, **overrides):
        fields = dict(
            enable_hierarchical_cache=True,
            spec_algorithm=mock.Mock(is_none=mock.Mock(return_value=True)),
            tree_cache=mock.Mock(
                enable_deferred_backup_flush=mock.Mock(return_value=True)
            ),
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_requires_the_env_flag(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_OPT_HICACHE_SUBMIT_BACKUPS_AFTER_LAUNCH.override(False):
            sched = self._sched()
            self.assertFalse(Scheduler._enable_deferred_backup_flush(sched))
            sched.tree_cache.enable_deferred_backup_flush.assert_not_called()
        with envs.SGLANG_OPT_HICACHE_SUBMIT_BACKUPS_AFTER_LAUNCH.override(True):
            self.assertTrue(Scheduler._enable_deferred_backup_flush(self._sched()))

    def test_spec_decoding_and_unsupported_caches_stay_eager(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_OPT_HICACHE_SUBMIT_BACKUPS_AFTER_LAUNCH.override(True):
            spec = self._sched(
                spec_algorithm=mock.Mock(is_none=mock.Mock(return_value=False))
            )
            self.assertFalse(Scheduler._enable_deferred_backup_flush(spec))
            unsupported = self._sched(
                tree_cache=mock.Mock(
                    enable_deferred_backup_flush=mock.Mock(return_value=False)
                )
            )
            with self.assertLogs("sglang.srt.managers.scheduler", level="WARNING"):
                self.assertFalse(Scheduler._enable_deferred_backup_flush(unsupported))


if __name__ == "__main__":
    unittest.main()
