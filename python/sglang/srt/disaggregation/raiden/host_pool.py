"""Sizing and admission control for raiden's host staging pool.

A raiden engine reserves ``num_slots * max_blocks`` pages of pinned host memory
at startup. Fixed staging borrows one complete slot per transfer; dynamic
staging borrows the transfer's actual block count from the same pool. Deriving
the pool size from a memory budget prevents a large context length from causing
an unbounded startup allocation, while admission prevents pool overcommit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class SlotAdmission:
    """Admission control over raiden's host staging pool.

    With dynamic staging, Raiden borrows the actual number of full-attention
    and SWA blocks a transfer needs from separate pools. ``capacity`` remains a
    count ceiling so tiny transfers cannot create unbounded concurrent work.

    When block capacities are omitted this retains the legacy count-only
    behavior used with fixed native staging slots. Keys are whatever identifies
    one read -- a room, or a room's chunk.
    """

    def __init__(
        self,
        capacity: int,
        *,
        full_block_capacity: Optional[int] = None,
        swa_block_capacity: Optional[int] = None,
    ):
        self.capacity = max(1, int(capacity))
        self.full_block_capacity = (
            None if full_block_capacity is None else max(0, int(full_block_capacity))
        )
        self.swa_block_capacity = (
            None if swa_block_capacity is None else max(0, int(swa_block_capacity))
        )
        self._in_flight: Dict[Any, Tuple[int, int]] = {}
        self._full_blocks_in_flight = 0
        self._swa_blocks_in_flight = 0

    def __len__(self) -> int:
        return len(self._in_flight)

    def __contains__(self, key: Any) -> bool:
        return key in self._in_flight

    def admit(
        self,
        key: Any,
        *,
        full_blocks: Optional[int] = None,
        swa_blocks: int = 0,
    ) -> bool:
        """Reserves pool capacity for ``key``. False means retry later."""
        if key in self._in_flight:
            return True
        if len(self._in_flight) >= self.capacity:
            return False

        # No block budget means native Raiden is using fixed slots. Preserve
        # the old one-transfer-per-slot admission behavior in that mode.
        if self.full_block_capacity is None:
            self._in_flight[key] = (0, 0)
            return True

        if full_blocks is None:
            raise ValueError("full_blocks is required for block-weighted admission")
        full_blocks = int(full_blocks)
        swa_blocks = int(swa_blocks)
        if full_blocks < 0 or swa_blocks < 0:
            raise ValueError("staging block counts must be non-negative")
        if self._full_blocks_in_flight + full_blocks > self.full_block_capacity:
            return False
        if swa_blocks and self.swa_block_capacity is None:
            return False
        if (
            self.swa_block_capacity is not None
            and self._swa_blocks_in_flight + swa_blocks > self.swa_block_capacity
        ):
            return False

        self._in_flight[key] = (full_blocks, swa_blocks)
        self._full_blocks_in_flight += full_blocks
        self._swa_blocks_in_flight += swa_blocks
        return True

    def release(self, key: Any) -> None:
        held = self._in_flight.pop(key, None)
        if held is None:
            return
        full_blocks, swa_blocks = held
        self._full_blocks_in_flight -= full_blocks
        self._swa_blocks_in_flight -= swa_blocks

    def rooms(self) -> List[Any]:
        return list(self._in_flight)


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


def ranks_per_node(
    attn_tp_size: int,
    attn_dp_size: int,
    system_dp_size: int,
    nnodes: int,
    attn_cp_size: int = 1,
) -> int:
    """Attention ranks resident on one host.

    A job's ranks -- every attention-TP rank of every context-parallel,
    attention-DP, and system-DP replica -- are spread evenly over its nodes,
    and each of them reserves its own host staging, so a per-node budget is
    shared by this many.
    """
    total = (
        max(1, attn_tp_size)
        * max(1, attn_dp_size)
        * max(1, system_dp_size)
        * max(1, attn_cp_size)
    )
    return max(1, total // max(1, nnodes))


def node_worker_rank(
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_dp_rank: int,
    attn_dp_size: int,
    system_dp_rank: int,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
) -> int:
    """Distinct rank of one worker among every worker on its node.

    Counts the attention-TP ranks of every context-parallel, attention-DP,
    and system-DP replica, so no two workers on a host share the ports,
    raiden engine id, or uuid namespace derived from it.
    """
    replica = system_dp_rank * max(1, attn_dp_size) + attn_dp_rank
    replica = replica * max(1, attn_cp_size) + attn_cp_rank
    return replica * max(1, attn_tp_size) + attn_tp_rank


def slots_within_budget(budget_bytes: float, slot_bytes: int, ceiling: int) -> int:
    """Slots that fit ``budget_bytes``, at most ``ceiling``.

    Raises ``ValueError`` when not even one slot fits: the pool cannot be
    allocated within the budget, so the configuration has to change.
    """
    if slot_bytes > budget_bytes:
        raise ValueError(
            "raiden host staging: one transfer slot needs "
            f"{slot_bytes / 1024**3:.2f} GiB but this engine's share of "
            f"SGLANG_RAIDEN_HOST_POOL_GB is {budget_bytes / 1024**3:.2f} GiB; "
            "raise SGLANG_RAIDEN_HOST_POOL_GB or lower the context length"
        )
    return max(1, min(ceiling, int(budget_bytes // max(1, slot_bytes))))
