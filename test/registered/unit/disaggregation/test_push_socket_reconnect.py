"""Unit tests for srt/disaggregation/common/conn — PUSH socket reconnect options."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import threading
import time
import unittest

import zmq

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.test_utils import CustomTestCase


def _free_port(ctx: zmq.Context) -> int:
    """Bind and release a port, so connecting to it finds nothing listening."""
    probe = ctx.socket(zmq.PULL)
    try:
        return probe.bind_to_random_port("tcp://127.0.0.1")
    finally:
        probe.close(linger=0)


class TestPushSocketReconnect(CustomTestCase):
    """A PUSH whose first connect fails must still deliver once the peer is up.

    Regression guard for a TPU disagg CI flake. `_connect` rebuilds a socket
    when its monitor reports EVENT_DISCONNECTED, but libzmq only raises that for
    a connection that was established and then dropped. When the very first
    connect fails there is no event, so with reconnection disabled
    (RECONNECT_IVL=-1) nothing ever revived the socket.

    The failure was silent, which is what made it expensive to diagnose: PUSH
    creates its pipe at connect() time, so send() queued the message and
    returned success. The sender logged nothing, the receiver waited out its
    full timeout, and the bytes were stranded permanently.
    """

    def setUp(self):
        self.ctx = zmq.Context()
        self.addCleanup(self.ctx.destroy, 0)
        # _connect only touches these four attributes plus _zmq_ctx.
        self.manager = CommonKVManager.__new__(CommonKVManager)
        self.manager._zmq_ctx = self.ctx
        self.manager._socket_cache = {}
        self.manager._monitor_cache = {}
        self.manager._socket_lock = threading.Lock()
        self.manager._socket_send_locks = {}
        self.addCleanup(self._close_manager_sockets)

    def _close_manager_sockets(self):
        for sock in [
            *self.manager._socket_cache.values(),
            *self.manager._monitor_cache.values(),
        ]:
            if not sock.closed:
                sock.close(linger=0)

    def test_send_before_peer_exists_is_delivered_once_peer_binds(self):
        port = _free_port(self.ctx)
        endpoint = f"tcp://127.0.0.1:{port}"

        # Nothing is listening yet. This send does not raise -- it queues.
        sock = self.manager._connect(endpoint)
        sock.send_multipart([b"queued-before-peer-existed"])

        receiver = self.ctx.socket(zmq.PULL)
        self.addCleanup(receiver.close, 0)
        # Generous: the retry backoff is 100ms growing to a 5s ceiling, and CI
        # hosts are slow. A stranded socket never delivers no matter the wait,
        # so this bounds the pass case without weakening the assertion.
        receiver.setsockopt(zmq.RCVTIMEO, 30000)
        receiver.bind(endpoint)

        started = time.perf_counter()
        try:
            received = receiver.recv_multipart()
        except zmq.Again:
            self.fail(
                "PUSH never delivered after the peer came up: the socket was "
                "stranded by its failed initial connect. Check that _connect "
                "leaves reconnection enabled."
            )
        self.assertEqual(received, [b"queued-before-peer-existed"])
        self.assertLess(time.perf_counter() - started, 30)

    def test_connect_enables_bounded_reconnect_backoff(self):
        """Retries must be on, and capped so a long outage does not busy-loop."""
        sock = self.manager._connect(f"tcp://127.0.0.1:{_free_port(self.ctx)}")

        interval = sock.getsockopt(zmq.RECONNECT_IVL)
        self.assertGreater(
            interval, 0, "RECONNECT_IVL <= 0 disables retries and strands sockets"
        )
        self.assertGreaterEqual(sock.getsockopt(zmq.RECONNECT_IVL_MAX), interval)


if __name__ == "__main__":
    unittest.main()
