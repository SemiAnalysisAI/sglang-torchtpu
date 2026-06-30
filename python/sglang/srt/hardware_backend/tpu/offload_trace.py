"""Lightweight per-thread Perfetto-style CPU tracing for the TPU KV-offload path.

Emits ``trace-start-event`` / ``trace-end-event`` log lines (the format a
Perfetto converter / cdk's parser ingests) around scheduler and offload hot
paths so the CPU-side timeline -- scheduling, forward dispatch, D2H/H2D issue,
and the poll-to-complete wait -- can be lined up against the device trace.

Off by default; enable with ``SGLANG_KV_TRACE=1``. When off, ``trace()`` is a
no-op context manager with no logging or timestamp cost. Diagnostic only -- not
intended for upstream.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

logger = logging.getLogger(__name__)

ENABLED: bool = os.environ.get("SGLANG_KV_TRACE", "0") == "1"

# Perfetto "pid" lane. One process per TP rank, so the pid distinguishes ranks;
# the scheduler overwrites this with a rank-tagged label via set_label().
_LABEL: str = f"sgl-pid{os.getpid()}"


def set_label(label: str) -> None:
    """Tag this process's trace lane (e.g. ``sgl-tp3``)."""
    global _LABEL
    _LABEL = label


@contextlib.contextmanager
def trace(track: str, name: str):
    """Bracket a span on ``track`` (the Perfetto "tid" lane) named ``name``."""
    if not ENABLED:
        yield
        return
    logger.info(
        "trace-start-event: name=%s ts=%.6f pid=%s tid=%s",
        name,
        time.time(),
        _LABEL,
        track,
    )
    try:
        yield
    finally:
        logger.info(
            "trace-end-event: name=%s ts=%.6f pid=%s tid=%s",
            name,
            time.time(),
            _LABEL,
            track,
        )


def instant(track: str, name: str) -> None:
    """Emit a zero-width marker (e.g. a queue depth change) on ``track``."""
    if not ENABLED:
        return
    logger.info(
        "trace-instant-event: name=%s ts=%.6f pid=%s tid=%s",
        name,
        time.time(),
        _LABEL,
        track,
    )


class OffloadStats:
    """Periodic in-flight / cumulative-count logging for one transfer direction."""

    def __init__(self, direction: str) -> None:
        self.direction = direction
        self.issued = 0
        self.completed = 0
        self._last_log = 0.0
        self._interval = float(os.environ.get("SGLANG_KV_TRACE_STAT_S", "2.0"))

    def maybe_log(self, in_flight: int, where: str) -> None:
        if not ENABLED:
            return
        now = time.monotonic()
        if (now - self._last_log) < self._interval:
            return
        self._last_log = now
        logger.info(
            "[kv-offload %s @%s] in_flight=%d issued=%d completed=%d",
            self.direction,
            where,
            in_flight,
            self.issued,
            self.completed,
        )
