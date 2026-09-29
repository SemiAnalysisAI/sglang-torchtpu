"""A decode room that waits too long for a slot is shed, and the prefill is
told to release its copy."""

import threading
import unittest
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.raiden import conn as conn_module
from sglang.srt.disaggregation.raiden import receiver as receiver_module
from sglang.srt.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang.srt.disaggregation.raiden.host_pool import StagingAdmission
from sglang.srt.disaggregation.raiden.state import DecodeState
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOM = 71
T0 = 1_000_000.0


def _decode_manager(prealloc_timeout_s=60.0):
    """A decode manager with no BLOCKS for the room yet, no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr.attn_tp_rank = 0
    mgr.chunked_transfer = False
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._engine = Mock()
    mgr._engine.poll_stats.return_value = ([], [], [])
    mgr._swa_engine = None
    mgr._dynamic_host_staging = True
    mgr._pulling = StagingAdmission(64, None)
    mgr._pulling_cleared = set()
    mgr._full_engine_max_blocks = 16
    mgr._swa_engine_max_blocks = 4
    mgr._aborted = {}
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._send_reads = {}
    mgr._cleared_plans = {}
    mgr._stale_plans = []
    mgr._engine_poll_failing_since = {}
    mgr._timeout_s = 120.0
    mgr.waiting_timeout = 300
    mgr._prealloc_timeout_s = prealloc_timeout_s
    return mgr


def _receiver(mgr, enqueued_at=T0):
    recv = RaidenKVReceiver.__new__(RaidenKVReceiver)
    recv.kv_mgr = mgr
    recv.bootstrap_room = ROOM
    recv.conclude_state = None
    recv.abort_notified = False
    recv.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 4444}]
    recv._started = False
    recv.init_time = None
    recv.enqueue_time = enqueued_at
    return recv


def _poll_at(recv, now):
    with patch.object(receiver_module.time, "time", return_value=now):
        return recv.poll()


class TestPreallocDeadline(unittest.TestCase):
    def test_room_is_shed_after_the_deadline_and_the_prefill_told_once(self):
        recv = _receiver(_decode_manager())

        with patch.object(RaidenKVReceiver, "_send_abort_notification") as abort:
            self.assertNotEqual(_poll_at(recv, T0 + 59), KVPoll.Failed)
            self.assertEqual(_poll_at(recv, T0 + 61), KVPoll.Failed)
            self.assertEqual(_poll_at(recv, T0 + 62), KVPoll.Failed)

        abort.assert_called_once()
        with self.assertRaisesRegex(Exception, "waited over 60s for a decode slot"):
            recv.failure_exception()

    def test_no_prealloc_deadline_once_metadata_is_sent(self):
        recv = _receiver(_decode_manager())
        recv.init_time = T0 + 50  # send_metadata ran: the waiting timeout owns it

        with patch.object(RaidenKVReceiver, "_send_abort_notification") as abort:
            self.assertNotEqual(_poll_at(recv, T0 + 200), KVPoll.Failed)

        abort.assert_not_called()

    def test_zero_timeout_disables_the_deadline(self):
        recv = _receiver(_decode_manager(prealloc_timeout_s=0.0))

        self.assertNotEqual(_poll_at(recv, T0 + 10_000), KVPoll.Failed)


class TestPreallocTimeoutKnob(unittest.TestCase):
    def _resolve(self, value, bootstrap=300):
        with (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.override(bootstrap),
            envs.SGLANG_RAIDEN_PREALLOC_TIMEOUT_S.override(value),
        ):
            return conn_module._prealloc_timeout_s()

    def test_clamped_below_the_prefill_silence_timeout(self):
        self.assertEqual(self._resolve(400), 270.0)
        self.assertEqual(self._resolve(60, bootstrap=60), 30.0)
        self.assertEqual(self._resolve(60, bootstrap=10), 1.0)

    def test_non_positive_disables(self):
        self.assertEqual(self._resolve(0), 0.0)
        self.assertEqual(self._resolve(-5), 0.0)


if __name__ == "__main__":
    unittest.main()
