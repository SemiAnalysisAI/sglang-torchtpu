"""Full garbage collections in the same loop iteration on every pipeline stage.

With SGLANG_PP_ALIGNED_GC_ITERS=N > 0, each pipeline stage freezes its heap
when its loop starts, turns off CPython's automatic collection of the oldest
generation, and runs a full collection at the top of every N-th loop
iteration. The stages' loop iterations run in lockstep, so they collect in the
same iteration and the pipeline pauses once per N iterations rather than once
per stage. Survivors are not re-frozen, so garbage cycles that later form among
them are still reclaimed.
"""

from __future__ import annotations

import gc as _gc
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# A gen-2 threshold no count of young collections reaches.
_NEVER = 1 << 30
# Arbitrary: faster collections log at debug level.
_SLOW_S = 0.05


class AlignedGc:
    def __init__(self, every_iters: int, *, gc: Any = None):
        self.every_iters = every_iters
        self._gc = gc if gc is not None else _gc
        self.iterations = 0
        self.collections = 0

    def start(self) -> None:
        """Freeze the startup heap and leave the oldest generation to tick()."""
        self._gc.freeze()
        young, middle, _ = self._gc.get_threshold()
        self._gc.set_threshold(young, middle, _NEVER)
        logger.info(
            "Aligned full GC every %d PP loop iterations; froze the heap, "
            "automatic full collections off",
            self.every_iters,
        )

    def tick(self) -> bool:
        """Call at the top of every loop iteration: collect fully every N."""
        self.iterations += 1
        if self.iterations % self.every_iters:
            return False
        started = time.perf_counter()
        unreachable = self._gc.collect()
        elapsed = time.perf_counter() - started
        self.collections += 1
        log = logger.info if elapsed >= _SLOW_S else logger.debug
        log(
            "Aligned full GC at PP loop iteration %d took %.3f s, %d unreachable",
            self.iterations,
            elapsed,
            unreachable,
        )
        return True
