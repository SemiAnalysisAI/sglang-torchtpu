"""Deferred call primitive shared by TPU compilation policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PendingCall:
    """A deferred compiled-function call, including any required compilation."""

    _wait: Callable[[], Any]

    def wait(self) -> Any:
        return self._wait()

    def then(self, callback: Callable[[Any], Any]) -> PendingCall:
        """Apply ``callback`` after this result is ready."""
        return PendingCall(lambda: callback(self.wait()))
