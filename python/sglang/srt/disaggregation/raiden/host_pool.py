"""Sizing for raiden's host staging pool.

A raiden engine reserves ``num_slots * max_blocks`` pages of pinned host memory
at startup, and borrows one slot per in-flight transfer. Deriving the slot count
from a memory budget keeps a large context length from turning into a host
allocation that fails at startup.
"""

from __future__ import annotations

from typing import Any, List


def slot_host_bytes(kv_caches: List[Any], max_blocks: int) -> int:
    """Host staging bytes one transfer slot reserves.

    Raiden allocates ``max_blocks`` pages of host memory for every layer, so a
    slot costs the widest layer's page times the layer count times the per-slot
    block budget.
    """
    page_bytes = 0
    for buf in kv_caches:
        layer_bytes = buf.element_size()
        for dim in buf.shape[1:]:
            layer_bytes *= int(dim)
        page_bytes = max(page_bytes, layer_bytes)
    return page_bytes * len(kv_caches) * max_blocks


def slots_within_budget(budget_bytes: float, slot_bytes: int, ceiling: int) -> int:
    """Slots that fit ``budget_bytes``, at least one and at most ``ceiling``."""
    return max(1, min(ceiling, int(budget_bytes // max(1, slot_bytes))))
