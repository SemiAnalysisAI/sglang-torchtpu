"""Tests for the deferred TPU compilation call primitive."""

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall


def test_pending_call_separates_compile_wait_from_execution() -> None:
    events = []
    pending = PendingCall(
        lambda: events.append("compiled"),
        lambda: events.append("executed") or "result",
    ).then(lambda result: events.append(f"callback:{result}") or "final")

    pending.wait_compiled()
    assert events == ["compiled"]
    assert pending.wait() == "final"
    assert events == ["compiled", "executed", "callback:result"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
