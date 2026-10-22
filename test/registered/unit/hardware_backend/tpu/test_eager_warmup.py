"""Eager warmup must retain asynchronous D2H destinations through its join."""

import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation.eager_warmup import eager_warmup

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("fail", [False, True])
def test_eager_warmup_joins_before_releasing_destinations(monkeypatch, fail):
    events = []
    destinations = []

    class Source:
        def to(self, device, *, non_blocking):
            assert device == "cpu" and non_blocking
            events.append("submit")
            destination = torch.empty(1)
            destinations.append(weakref.ref(destination))
            return destination

    def synchronize():
        assert events == ["submit", "submit"]
        assert all(reference() is not None for reference in destinations)
        events.append("join")

    stream = SimpleNamespace(synchronize=synchronize)
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(current_stream=lambda: stream), raising=False
    )

    def warmup():
        with eager_warmup("test") as materialize:
            materialize(Source())
            materialize(Source())
            assert events == ["submit", "submit"]
            if fail:
                raise ValueError("failed after submission")
        assert all(reference() is None for reference in destinations)

    if fail:
        with pytest.raises(ValueError, match="failed after submission"):
            warmup()
    else:
        warmup()
    assert events == ["submit", "submit", "join"]
    assert all(reference() is None for reference in destinations)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
