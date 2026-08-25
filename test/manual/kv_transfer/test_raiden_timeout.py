"""Unit tests for the raiden stall->Failed timeout.

These cover the timeout wiring in ``RaidenKVSender.poll`` /
``RaidenKVReceiver.poll`` with a stub kv-manager, so no TPU, torch_tpu or zmq is
required and they run anywhere. The on-device transfer path is validated
separately on a TPU VM.

Why these exist. Preallocated KV is released only by
``DecodeTransferQueue.pop_transferred`` on ``KVPoll.Failed``; the transition into
Failed for a stalled transfer is ``_check_waiting_timeout`` /
``_check_bootstrap_timeout``. Raiden was the only backend not calling them, so a
transfer whose peer disappeared pinned its blocks for the life of the process and
a production deployment wedged after ~22h of serving. Each test below locks
one way that fix was got wrong
before it was got right -- all four failure modes were silently inert or actively
harmful rather than loud, which is exactly why they need tests:

  * the helper is called but ``init_time`` is never stamped -> no-op
  * the sender calls the receiver's helper -> AttributeError on first poll
  * the call is gated on a state raiden never occupies -> dead code
  * the clock starts at registration, not transfer start -> healthy queued
    requests get failed (measured 57% false failures)

Run: python -m pytest test/manual/kv_transfer/test_raiden_timeout.py
"""

import time
import unittest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver, CommonKVSender
from sglang.srt.disaggregation.raiden.conn import RaidenKVReceiver, RaidenKVSender

BUDGET = 300.0


class StubKVManager:
    """Minimal stand-in for RaidenKVManager covering only what poll() touches."""

    def __init__(self, status=KVPoll.WaitingForInput, pull_starts=False):
        self.waiting_timeout = BUDGET
        self.bootstrap_timeout = BUDGET
        self.is_dummy_cp_rank = False
        self._status = status
        self._pull_starts = pull_starts
        self.failures = {}
        self.status_writes = []
        self.drained = 0

    # -- surface used by RaidenKV{Sender,Receiver}.poll -------------------
    def _drain_engine(self):
        self.drained += 1

    def is_failed(self, room):
        return False

    def is_recv_done(self, room):
        return False

    def is_send_done(self, room):
        return False

    def check_status(self, room):
        return self._status

    def maybe_start_pull(self, room):
        return self._pull_starts

    def update_status(self, room, status):
        self._status = status
        self.status_writes.append(status)

    def record_failure(self, room, reason):
        self.failures[room] = reason


def _receiver(mgr, init_time, started=False):
    """Build a receiver without running __init__ (which needs zmq/bootstrap)."""
    r = object.__new__(RaidenKVReceiver)
    r.kv_mgr = mgr
    r.bootstrap_room = 1234
    r.conclude_state = None
    r.init_time = init_time
    r._started = started
    r.abort_notified = True  # skip the peer notification path
    r.bootstrap_infos = None
    return r


def _sender(mgr, init_time):
    s = object.__new__(RaidenKVSender)
    s.kv_mgr = mgr
    s.bootstrap_room = 5678
    s.conclude_state = None
    s.init_time = init_time
    s._started = True
    s._last_chunk = False
    return s


class TestRaidenReceiverTimeout(unittest.TestCase):
    """The decode side, where the consequence of not firing is a PERMANENT leak."""

    def test_stalled_pull_times_out(self):
        """A pull that started and then stalled must reach Failed.

        This is the whole point of the fix. It also locks out gating the check on
        WaitingForInput: a stalled pull sits in Transferring, so such a gate
        excludes exactly the population that leaks.
        """
        mgr = StubKVManager(status=KVPoll.Transferring)
        rx = _receiver(mgr, init_time=time.time() - (BUDGET + 1), started=True)

        self.assertEqual(rx.poll(), KVPoll.Failed)
        self.assertIn(KVPoll.Failed, mgr.status_writes)
        self.assertIn("timed out", mgr.failures[rx.bootstrap_room])

    def test_pull_that_never_starts_times_out(self):
        """Producer gone: the pull never starts, so the registration stamp applies."""
        mgr = StubKVManager(status=KVPoll.WaitingForInput, pull_starts=False)
        rx = _receiver(mgr, init_time=time.time() - (BUDGET + 1), started=False)

        self.assertEqual(rx.poll(), KVPoll.Failed)

    def test_queued_request_is_not_killed(self):
        """A long wait BEFORE the transfer starts must not consume the budget.

        Regression: init_time was stamped at registration, but the gap to pull
        start is dominated by the KV allocation wait (kv_transfer_alloc_ms
        averaged ~228s against a 300s budget). Timing from registration killed
        requests that were merely queued -- 57% false failures observed in a
        production deployment.
        """
        mgr = StubKVManager(status=KVPoll.WaitingForInput, pull_starts=True)
        rx = _receiver(mgr, init_time=time.time() - (BUDGET + 1), started=False)

        # The pull starts on this poll, which must re-stamp the clock.
        self.assertNotEqual(rx.poll(), KVPoll.Failed)
        self.assertAlmostEqual(rx.init_time, time.time(), delta=5)
        self.assertNotIn(rx.bootstrap_room, mgr.failures)

    def test_healthy_transfer_is_not_killed(self):
        mgr = StubKVManager(status=KVPoll.Transferring)
        rx = _receiver(mgr, init_time=time.time() - 1.0, started=True)

        self.assertNotEqual(rx.poll(), KVPoll.Failed)
        self.assertNotIn(rx.bootstrap_room, mgr.failures)

    def test_unstamped_clock_is_inert(self):
        """Guards the no-op failure mode: the helper returns None while init_time
        is None, so forgetting to stamp it makes the whole fix dead code."""
        mgr = StubKVManager(status=KVPoll.Transferring)
        rx = _receiver(mgr, init_time=None, started=True)

        self.assertNotEqual(rx.poll(), KVPoll.Failed)


class TestRaidenSenderTimeout(unittest.TestCase):
    """The prefill side, where the consequence is raiden send-slot exhaustion."""

    def test_stalled_send_times_out(self):
        """Locks out copying MooncakeKVSender's `status == Bootstrapping` gate.

        Raiden is consumer-pull and RaidenKVSender.__init__ leaves Bootstrapping
        immediately, so that gate would make this timeout unreachable. A stalled
        send is observed in Transferring.
        """
        mgr = StubKVManager(status=KVPoll.Transferring)
        tx = _sender(mgr, init_time=time.time() - (BUDGET + 1))

        self.assertEqual(tx.poll(), KVPoll.Failed)
        self.assertIn("timed out", mgr.failures[tx.bootstrap_room])

    def test_healthy_send_is_not_killed(self):
        mgr = StubKVManager(status=KVPoll.Transferring)
        tx = _sender(mgr, init_time=time.time() - 1.0)

        self.assertNotEqual(tx.poll(), KVPoll.Failed)


class TestTimeoutHelperOwnership(unittest.TestCase):
    """Structural guards -- these fail at import/attribute level, not behaviour."""

    def test_sender_and_receiver_use_different_helpers(self):
        """Locks the AttributeError regression.

        `_check_waiting_timeout` is defined on CommonKVReceiver only;
        CommonKVSender exposes `_check_bootstrap_timeout`. Calling the receiver's
        helper from the sender raised AttributeError on the first poll and cycled
        the prefill schedulers.
        """
        self.assertTrue(hasattr(CommonKVReceiver, "_check_waiting_timeout"))
        self.assertFalse(hasattr(CommonKVSender, "_check_waiting_timeout"))
        self.assertTrue(hasattr(CommonKVSender, "_check_bootstrap_timeout"))

    def test_raiden_poll_overrides_call_a_timeout_helper(self):
        """Raiden was the only backend whose poll() never called one at all."""
        import inspect

        rx_src = inspect.getsource(RaidenKVReceiver.poll)
        tx_src = inspect.getsource(RaidenKVSender.poll)
        self.assertIn("_check_waiting_timeout", rx_src)
        self.assertIn("_check_bootstrap_timeout", tx_src)

    def test_receiver_restamps_clock_at_pull_start(self):
        """The re-stamp is what makes the deadline measure the transfer."""
        import inspect

        src = inspect.getsource(RaidenKVReceiver.poll)
        start = src.index("maybe_start_pull")
        self.assertIn(
            "init_time", src[start:], "clock must be re-stamped at pull start"
        )


if __name__ == "__main__":
    unittest.main()
