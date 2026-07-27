# Emit one ``trace-event`` line to stdout per KV transfer phase (gather / push /
# receive / scatter).
"""Lightweight Perfetto trace-event instrumentation for TPU P/D KV transfer."""

import contextlib
import threading
import time
from typing import List, Optional

import torch


class _LazyPerfetto:
    """Lazy initializer/evaluator for _PERFETTO flag using sglang envs."""

    def __bool__(self) -> bool:
        try:
            from sglang.srt.environ import envs

            return bool(envs.SGLANG_TPU_ENABLE_DISAGGREGATION_PERFETTO_TRACING.get())
        except Exception:
            return False


_PERFETTO = _LazyPerfetto()


def fmt_bytes(n: float) -> str:
    """Format bytes to readable string: e.g. ``12.3MB``."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0


class _TraceSpan:
    """Trace span objects. Use slots for minimum memory consumption."""

    __slots__ = ("name", "args")

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.args = kwargs


# FIXME(junyanxu): Consider integrating with sglang/srt/observability/trace.py (OTEL)
# in the future if we can bypass the microsecond-level OTEL overhead for high-frequency data planes.
@contextlib.contextmanager
def perfetto_trace(pid: str, name: str, **kwargs):
    """Perfetto trace context manager using process/track ID string."""
    if not _PERFETTO:
        yield _TraceSpan(name, **kwargs)
        return
    span = _TraceSpan(name, **kwargs)
    t_wall = time.time()
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        ts_us = int(t_wall * 1e6)
        dur_us = int((time.perf_counter() - t0) * 1e6)
        tid = threading.current_thread().name
        if span.args:
            args_str = " ".join(f"{k}={v}" for k, v in span.args.items())
            print(
                f"trace-event: name={span.name} ts={ts_us} dur={dur_us} "
                f"pid={pid} tid={tid} {args_str}",
                flush=True,
            )
        else:
            print(
                f"trace-event: name={span.name} ts={ts_us} dur={dur_us} "
                f"pid={pid} tid={tid}",
                flush=True,
            )
