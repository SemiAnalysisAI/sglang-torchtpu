"""Track XLA compile-cache behaviour so an unexpected recompile is loud.

A warm run is expected to hit the on-disk compile cache (``TORCH_TPU_TIER3_ROOT``)
for every program it executes. A cache *miss* after warmup means a graph or shape
we never precompiled reached the device, and paying a fresh XLA compile mid-serve
is exactly what shows up as a p99 TTFT / ITL spike.

This is not the same signal as dynamo's recompile counter, and one does not imply
the other:

* ``torch.compiler.set_stance("fail_on_recompile")`` (armed by the scheduler once
  warmup finishes) guards only the *torch.compile frontend*. A guard failure
  raises, so post-warmup dynamo recompiles are zero by construction.
* Eager ops outside the compiled region are lowered and compiled by torch_tpu
  independently. Those never touch a dynamo guard, so they can compile freely at
  steady state and the dynamo counter stays at zero while the device stalls.

The second kind is the one that has actually bitten this backend, so it is the one
worth counting.
"""

from __future__ import annotations

import logging

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class CacheCounters(msgspec.Struct, frozen=True):
    """XLA compile-cache counters at one instant."""

    reqs: int = 0
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.reqs if self.reqs else 0.0

    def delta_from(self, other: "CacheCounters") -> "CacheCounters":
        return CacheCounters(
            reqs=self.reqs - other.reqs,
            hits=self.hits - other.hits,
            misses=self.misses - other.misses,
        )


def _supported() -> bool:
    """torch_tpu exposes the counters; other backends and older builds do not."""
    tpu = getattr(torch, "tpu", None)
    return tpu is not None and hasattr(tpu, "_get_cache_requests")


def snapshot() -> CacheCounters | None:
    """Read the counters, or None when unavailable."""
    if not _supported():
        return None
    try:
        return CacheCounters(
            reqs=torch.tpu._get_cache_requests(),
            hits=torch.tpu._get_cache_hits(),
            misses=torch.tpu._get_cache_misses(),
        )
    except Exception as exc:  # counters are diagnostics; never break serving
        logger.debug("[tpu] compile-cache counters unavailable: %s", exc)
        return None


class CompileTracker:
    """Baseline the counters at warmup, then report anything new.

    Held by the scheduler. ``arm`` once, ``poll`` as often as convenient -- it is
    two integer reads, cheap enough for the idle path.
    """

    def __init__(self) -> None:
        self._baseline: CacheCounters | None = None
        self._last: CacheCounters | None = None
        self._unexpected: int = 0

    def arm(self) -> None:
        """Record the post-warmup baseline and report how the disk cache did.

        The startup hit rate answers "did the on-disk cache actually work" -- a
        warm run that reports 0% hits recompiled everything from scratch and the
        cache is not being reused, whatever the wall time suggests.
        """
        now = snapshot()
        if now is None:
            return
        self._baseline = now
        self._last = now
        logger.info(
            "[tpu] compile cache after warmup: %d/%d hits (%.1f%%), %d misses. "
            "Post-warmup misses are unexpected and will be logged.",
            now.hits,
            now.reqs,
            100.0 * now.hit_rate,
            now.misses,
        )

    def poll(self, *, context: str = "") -> None:
        """Warn if anything compiled since the last poll."""
        if self._last is None:
            return
        now = snapshot()
        if now is None:
            return
        step = now.delta_from(self._last)
        self._last = now
        if step.misses <= 0:
            return
        self._unexpected += step.misses
        total = now.delta_from(self._baseline)
        logger.warning(
            "[tpu] UNEXPECTED RECOMPILE%s: %d new XLA cache miss(es) "
            "(%d since warmup, %d requests). A fresh compile at serving time "
            "stalls the device and inflates p99 TTFT/ITL. Set "
            "SGLANG_TPU_DUMP_ON_CACHE_MISS=1 to dump the offending graph.",
            f" [{context}]" if context else "",
            step.misses,
            total.misses,
            step.reqs,
        )

    def summary(self) -> str:
        if self._baseline is None or self._last is None:
            return "[tpu] compile cache: counters unavailable"
        total = self._last.delta_from(self._baseline)
        return (
            f"[tpu] compile cache since warmup: {total.reqs} requests, "
            f"{total.hits} hits, {total.misses} misses "
            f"({self._unexpected} unexpected recompiles)"
        )


def maybe_enable_cache_miss_dump() -> None:
    """Dump the aten graph + StableHLO for any program that misses the cache.

    Expensive and very verbose -- only for chasing down a specific recompile.
    """
    if not envs.SGLANG_TPU_DUMP_ON_CACHE_MISS.get() or not _supported():
        return
    try:
        torch.tpu._set_dump_on_cache_miss(True)
        logger.info("[tpu] dumping aten graph + StableHLO on every cache miss.")
    except Exception as exc:
        logger.warning("[tpu] could not enable cache-miss dumping: %s", exc)
