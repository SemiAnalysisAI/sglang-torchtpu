"""Fault windows armed through /raiden_faults disarm on their own and keep their hits."""

import threading
import time
import unittest

from sglang.srt.disaggregation.raiden.fault_control import RaidenFaultController
from sglang.srt.managers.io_struct import RaidenFaultReqInput, RaidenFaultReqOutput
from sglang.srt.managers.tokenizer_control_mixin import _merge_raiden_fault_results
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOOK = "block_transport.recv.progress"
RULES = [{"hook": HOOK, "action": "fail", "probability": 1.0}]
# Long enough that the real timer never fires inside a test that does not wait for it.
NEVER_S = 3600.0


class _FakeFaultInjection:
    """tpu-sync semantics: inject replaces rules but keeps hit counts; reset clears both."""

    def __init__(self):
        self._lock = threading.Lock()
        self.rules = []
        self.counts = {}

    def inject_faults(self, rules):
        with self._lock:
            self.rules = list(rules)

    def reset_faults(self):
        with self._lock:
            self.rules = []
            self.counts = {}

    def has_active_injections(self):
        with self._lock:
            return bool(self.rules)

    def get_fault_status(self):
        with self._lock:
            return dict(self.counts)

    def hit(self, hook):
        with self._lock:
            self.counts[hook] = self.counts.get(hook, 0) + 1


def _inject(ctrl, duration_s=NEVER_S):
    return ctrl.handle(
        RaidenFaultReqInput(op="inject", rules=RULES, duration_s=duration_s)
    )


class TestRaidenFaultControl(CustomTestCase):
    def setUp(self):
        self.fi = _FakeFaultInjection()
        self.ctrl = RaidenFaultController(fault_injection=self.fi)

    def tearDown(self):
        self.ctrl.handle(RaidenFaultReqInput(op="reset"))

    def test_superseded_expiry_does_not_disarm_newer_window(self):
        """An expiry left over from an earlier inject must not clear the rules armed after it."""
        _inject(self.ctrl)
        first_window = self.ctrl._window
        _inject(self.ctrl)
        self.ctrl._expire(window=first_window)
        self.assertTrue(self.fi.has_active_injections())
        self.ctrl._expire(window=self.ctrl._window)
        self.assertFalse(self.fi.has_active_injections())

    def test_expiry_disarms_without_the_scheduler(self):
        """Rules disarm after duration_s with no further request reaching the controller."""
        _inject(self.ctrl, duration_s=0.05)
        deadline = time.monotonic() + 5.0
        while self.fi.has_active_injections() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.fi.has_active_injections())

    def test_window_hits_survive_reset_and_restart_on_inject(self):
        """tpu-sync zeroes counts on reset; the window's hits must still be reported after it."""
        _inject(self.ctrl)
        self.fi.hit(HOOK)
        self.fi.hit(HOOK)
        out = self.ctrl.handle(RaidenFaultReqInput(op="reset"))
        self.assertFalse(out.armed)
        self.assertEqual(out.hits, {HOOK: 2})
        out = _inject(self.ctrl)
        self.assertTrue(out.armed)
        self.assertEqual(out.hits, {})

    def test_rejected_rules_report_failure_and_stay_disarmed(self):
        """A rule tpu-sync rejects must surface as a failed reply, not an armed server."""

        def reject(rules):
            raise ValueError("unknown hook")

        self.fi.inject_faults = reject
        out = _inject(self.ctrl)
        self.assertFalse(out.success)
        self.assertIn("unknown hook", out.message)
        self.assertFalse(out.armed)

    def test_merge_fails_when_any_worker_fails(self):
        merged = _merge_raiden_fault_results(
            [
                RaidenFaultReqOutput(success=True, armed=True, hits={HOOK: 1}),
                RaidenFaultReqOutput(success=False, message="boom", armed=False),
            ]
        )
        self.assertFalse(merged["success"])
        self.assertEqual(merged["errors"], ["boom"])
        self.assertEqual((merged["workers"], merged["armed_workers"]), (2, 1))
        self.assertEqual(merged["hits"], {HOOK: 1})


if __name__ == "__main__":
    unittest.main()
