# SPDX-License-Identifier: Apache-2.0
"""Fixed-row decode slots for the Kimi-K3 megakernel.

The megakernel decodes a batch tile of ``B`` rows (``B`` in ``ROW_BUCKETS``) and
reads its caches from rows ``[0, B)`` of whole state tensors that hold
``max_rows`` rows. Every request that the megakernel decodes owns one row for
its lifetime. Rows stay compact: when a request leaves, the request in the
highest row moves into the freed row, so a batch of ``n`` live requests always
fits the smallest bucket that holds ``n``.

This module is host-side bookkeeping only. The caller applies the row moves it
returns (copying the four state tensors' rows on device) before the next step.
"""

from __future__ import annotations

from dataclasses import dataclass

ROW_BUCKETS = (1, 2, 4, 8)


def row_bucket(rows: int) -> int:
    """Smallest megakernel batch tile that holds ``rows`` live rows."""
    for bucket in ROW_BUCKETS:
        if rows <= bucket:
            return bucket
    raise ValueError(
        f"{rows} rows exceed the megakernel's largest tile {ROW_BUCKETS[-1]}"
    )


@dataclass(frozen=True)
class RowMove:
    """Copy every state tensor's row ``source`` into row ``destination``."""

    source: int
    destination: int


class SlotTable:
    """Request id -> megakernel row, kept dense in ``[0, len(self))``."""

    def __init__(self, max_rows: int):
        if max_rows not in ROW_BUCKETS:
            raise ValueError(f"max_rows must be one of {ROW_BUCKETS}, got {max_rows}")
        self.max_rows = max_rows
        self._row_of: dict[str, int] = {}
        self._owner: list[str] = []

    def __len__(self) -> int:
        return len(self._owner)

    def __contains__(self, rid: str) -> bool:
        return rid in self._row_of

    def row(self, rid: str) -> int:
        return self._row_of[rid]

    def owner(self, row: int) -> str:
        return self._owner[row]

    def assign(self, rid: str) -> int:
        """Give ``rid`` the next free row and return it."""
        if rid in self._row_of:
            raise ValueError(f"request {rid!r} already owns row {self._row_of[rid]}")
        if len(self._owner) >= self.max_rows:
            raise RuntimeError(f"all {self.max_rows} megakernel rows are in use")
        row = len(self._owner)
        self._owner.append(rid)
        self._row_of[rid] = row
        return row

    def release(self, rid: str) -> RowMove | None:
        """Free ``rid``'s row. Returns the move that keeps rows dense, if any.

        The move must be applied to the device state before the next decode
        step, because the moved request's row index changes immediately.
        """
        row = self._row_of.pop(rid)
        last = len(self._owner) - 1
        if row == last:
            self._owner.pop()
            return None
        moved = self._owner.pop()
        self._owner[row] = moved
        self._row_of[moved] = row
        return RowMove(source=last, destination=row)

    def rows_for(self, rids: list[str]) -> list[int]:
        """Rows of ``rids`` in batch order."""
        return [self._row_of[rid] for rid in rids]
