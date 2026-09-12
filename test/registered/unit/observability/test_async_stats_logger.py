"""A blocked statistics sink must not stop scheduler progress."""

import logging
import threading
import time
import unittest
from unittest.mock import Mock, patch

from sglang.srt.utils.async_stats_logger import AsyncStatsLogger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class BlockingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release_write = threading.Event()
        self.records = []

    def emit(self, record):
        self.entered.set()
        self.release_write.wait()
        self.records.append(record)


class TestAsyncStatsLogger(CustomTestCase):
    def setUp(self):
        self.logger = logging.Logger("blocked-stats", level=logging.INFO)
        self.handler = BlockingHandler()
        self.logger.addHandler(self.handler)
        self.stats = AsyncStatsLogger(self.logger, capacity=2)

    def tearDown(self):
        self.handler.release_write.set()
        self.stats.queue.join()
        self.stats.listener.stop()

    def test_blocked_sink_bounds_backlog_and_preserves_producer_record(self):
        """A slow destination cannot block submission, even after backlog fills."""
        returned = threading.Event()

        def produce():
            self.stats.info("first")
            returned.set()

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            self.assertTrue(self.handler.entered.wait(2))
            self.assertTrue(returned.wait(1), "log write blocked the producer")
            self.stats.info("second")
            self.stats.info("third")
            for _ in range(100):
                self.stats.info("overflow")
            before_release = time.time()
            self.handler.release_write.set()
            self.stats.queue.join()
            self.stats.info("recovered")
            self.stats.queue.join()
            records = self.handler.records
            self.assertEqual(
                [r.getMessage() for r in records],
                ["first", "second", "third", "recovered, omitted statistics logs: 100"],
            )
            self.assertEqual(records[0].thread, producer.ident)
            self.assertEqual(records[0].funcName, "produce")
            self.assertLessEqual(records[2].created, before_release)
        finally:
            self.handler.release_write.set()
            producer.join(2)

    def test_decode_stats_do_not_wait_for_sink(self):
        """Regression: report_decode_stats previously blocked all TP ranks."""
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )

        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.scheduler = Mock()
        reporter.scheduler.running_batch.reqs = []
        reporter.scheduler.waiting_queue = []
        reporter.scheduler.spec_algorithm.is_none.return_value = True
        reporter.scheduler.pool_stats_observer.get_pool_stats.return_value.get_decode_usage_msg_parts.return_value = (
            []
        )
        reporter.current_scheduler_metrics_enabled = False
        reporter.is_stats_logging_rank = True
        reporter.forward_ct_decode = 10
        reporter.decode_log_interval = 10
        reporter.last_decode_stats_tic = time.perf_counter() - 1
        reporter.num_generated_tokens = 16
        reporter.enable_mfu_metrics = False
        reporter._graph_backend_label = "Torch compile"
        reporter.stats_logger = self.stats
        returned = threading.Event()
        errors = []

        def report():
            try:
                reporter.report_decode_stats(False)
                returned.set()
            except Exception as exc:
                errors.append(exc)

        module = "sglang.srt.managers.scheduler_components.metrics_reporter"
        with (
            patch(module + ".logger", self.logger),
            patch(module + ".get_disagg", return_value=Mock(language_only=False)),
            patch(module + ".RECORD_STEP_TIME", False),
            patch(module + ".ENABLE_METRICS_DEVICE_TIMER", False),
        ):
            producer = threading.Thread(target=report, daemon=True)
            producer.start()
            try:
                self.assertTrue(self.handler.entered.wait(2), errors)
                self.assertTrue(returned.wait(1), "scheduler waited for log sink")
                self.assertFalse(errors)
                reporter.scheduler.kv_events_publisher.publish_kv_events.assert_called_once()
            finally:
                self.handler.release_write.set()
                producer.join(2)


if __name__ == "__main__":
    unittest.main()
