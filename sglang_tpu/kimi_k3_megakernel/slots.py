# SPDX-License-Identifier: Apache-2.0
"""Fixed decode rows for the Kimi-K3 megakernel.

The megakernel decodes a tile of ``B`` rows (``B`` in ``ROW_BUCKETS``) over rows
``[0, B)`` of state arrays that hold ``max_rows`` rows. Every request it decodes
owns one row for its lifetime, and rows stay dense: when a request leaves, the
request in the highest row moves into the freed row, so ``n`` live requests
always occupy ``[0, n)`` and fit the smallest tile that holds ``n``. Host-side
bookkeeping only; the caller copies the moved row on device.
"""

from __future__ import annotations

from typing import NamedTuple

from .config import ROW_BUCKETS

# (req_pool_idx, req_generation): unique per request across slot reuse.
RequestKey = tuple[int, int]


class RowMove(NamedTuple):
    """Copy every state array's row ``source`` into row ``destination``."""

    source: int
    destination: int


class SlotTable:
    """Request -> megakernel row, dense in ``[0, len(self))``."""

    def __init__(self, max_rows: int):
        if max_rows not in ROW_BUCKETS:
            raise ValueError(f"max_rows must be one of {ROW_BUCKETS}, got {max_rows}")
        self.max_rows = max_rows
        self._row_of: dict[RequestKey, int] = {}
        self._owner: list[RequestKey] = []

    def __len__(self) -> int:
        return len(self._owner)

    def __contains__(self, key: RequestKey) -> bool:
        return key in self._row_of

    def owners(self) -> tuple[RequestKey, ...]:
        """Owners in row order."""
        return tuple(self._owner)

    def assign(self, key: RequestKey) -> int:
        """Give ``key`` the next free row and return it."""
        if key in self._row_of:
            raise ValueError(f"{key!r} already owns row {self._row_of[key]}")
        if len(self._owner) >= self.max_rows:
            raise RuntimeError(f"all {self.max_rows} megakernel rows are in use")
        row = len(self._owner)
        self._owner.append(key)
        self._row_of[key] = row
        return row

    def release(self, key: RequestKey) -> RowMove | None:
        """Free ``key``'s row; return the move that keeps the rows dense, if any.

        The moved request's row changes immediately, so the caller must apply
        the move on device before the next decode step.
        """
        row = self._row_of.pop(key)
        last = len(self._owner) - 1
        moved = self._owner.pop()
        if row == last:
            return None
        self._owner[row] = moved
        self._row_of[moved] = row
        return RowMove(source=last, destination=row)

    def rows_for(self, keys: list[RequestKey]) -> list[int]:
        """Rows of ``keys``, in the given order."""
        return [self._row_of[key] for key in keys]
