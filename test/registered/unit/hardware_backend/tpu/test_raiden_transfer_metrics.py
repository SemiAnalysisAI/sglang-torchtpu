"""CPU tests for Raiden KV transfer metrics."""

import threading
import unittest
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.raiden.conn import (
    RaidenKVManager,
    RaidenKVSender,
    _TransferIntervalTracker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTransferIntervalTracker(CustomTestCase):
    def test_duration_unions_overlaps_and_excludes_chunk_gaps(self):
        tracker = _TransferIntervalTracker()

        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=1.0,
        ):
            tracker.start(7, ["7#0"])
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=3.0,
        ):
            tracker.finish(["7#0"])

        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=10.0,
        ):
            tracker.start(7, ["7#1"])
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=10.5,
        ):
            tracker.start(7, ["swa:7"])
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=12.0,
        ):
            tracker.finish(["swa:7"])
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=13.0,
        ):
            tracker.finish(["7#1"])

        self.assertEqual(tracker.duration(7), 5.0)

    def test_duplicate_start_preserves_original_interval_without_raising(self):
        tracker = _TransferIntervalTracker()

        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=1.0,
        ):
            tracker.start(7, ["7#0"])
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=2.0,
        ):
            with self.assertLogs(
                "sglang.srt.disaggregation.raiden.conn", level="ERROR"
            ):
                tracker.start(7, ["7#0"])
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=3.0,
        ):
            tracker.finish(["7#0"])

        self.assertEqual(tracker.duration(7), 2.0)

    def test_sender_read_settlement_records_the_interval(self):
        manager = object.__new__(RaidenKVManager)
        manager._state_lock = threading.Lock()
        manager._aborted = {}
        manager._send_reads = {}
        manager._send_intervals = _TransferIntervalTracker()
        manager._completed_send = set()
        manager._native_failed = set()
        engine = Mock()
        engine.register_read.return_value = True

        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=1.0,
        ):
            manager._register_sender_read(7, engine, "7#0", 11, [3])

        manager._completed_send.add("7#0")
        with patch(
            "sglang.srt.disaggregation.raiden.conn.time.perf_counter",
            return_value=3.0,
        ):
            with manager._state_lock:
                manager._settle_sender_reads_locked()

        self.assertNotIn(7, manager._send_reads)
        self.assertEqual(manager.send_active_latency_s(7), 2.0)


class TestRaidenKVTransferMetrics(CustomTestCase):
    def test_sender_reports_backend_active_transfer_duration(self):
        manager = Mock()
        manager.is_dummy_cp_rank = False
        manager.kv_item_lens_sum = 1 << 30
        manager.state_item_lens_sum = 0
        manager.get_kv_replica_factor.return_value = 1
        manager.is_failed.return_value = False
        manager.is_send_done.return_value = True
        manager.check_status.return_value = KVPoll.Transferring
        manager.send_active_latency_s.return_value = 5.0

        sender = object.__new__(RaidenKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7
        sender.conclude_state = None
        sender._transfer_metric = KVTransferMetric()
        sender._transfer_num_kv_indices = 2
        sender._transfer_num_state_indices = 0

        self.assertEqual(sender.poll(), KVPoll.Success)

        metric = sender.get_transfer_metric()
        self.assertEqual(metric.transfer_total_bytes, 2 << 30)
        self.assertEqual(metric.transfer_latency_s, 5.0)
        self.assertEqual(
            metric.transfer_total_bytes / (1 << 30) / metric.transfer_latency_s,
            0.4,
        )


if __name__ == "__main__":
    unittest.main()
