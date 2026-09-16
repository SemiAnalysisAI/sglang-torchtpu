"""Deferred call primitive shared by TPU compilation policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingCall:
    """A deferred call whose compilation can be joined independently."""

    _wait_compiled: Callable[[], Any]
    _execute: Callable[[], Any]
    _start_compile: Callable[[], Any] = lambda: None
    _is_started: bool = field(default=False, init=False)
    _is_compiled: bool = field(default=False, init=False)

    def start_compiling(self) -> None:
        """Start deferred compilation without waiting for it."""
        if not self._is_started:
            self._start_compile()
            self._is_started = True

    def wait_compiled(self) -> None:
        """Wait until compilation finishes without executing the callable."""
        if not self._is_compiled:
            self.start_compiling()
            self._wait_compiled()
            self._is_compiled = True

    def wait(self) -> Any:
        self.wait_compiled()
        return self._execute()

    def then(self, callback: Callable[[Any], Any]) -> PendingCall:
        """Apply ``callback`` after this result is ready."""
        return PendingCall(
            self.wait_compiled,
            lambda: callback(self._execute()),
            self.start_compiling,
        )
