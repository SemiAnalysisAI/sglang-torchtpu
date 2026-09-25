"""Runtime control of tpu-sync fault injection inside a Raiden disaggregation worker.

Testing only; wired when SGLANG_ENABLE_RAIDEN_FAULT_INJECTION is set. The tpu-sync
injector is process-wide, so every scheduler process that loaded the Raiden
extension owns one controller and arms its own copy of the rules.
"""

import logging
import threading
from types import ModuleType
from typing import Any, Dict, List, Optional

from sglang.srt.managers.io_struct import RaidenFaultReqInput, RaidenFaultReqOutput

logger = logging.getLogger(__name__)


class RaidenFaultController:
    """Arms, disarms, and reports tpu-sync fault rules for this process.

    Every inject is bounded by a timer thread that resets the rules without the
    scheduler loop, so a scheduler wedged by an injected fault is still disarmed.
    """

    def __init__(self, *, fault_injection: ModuleType):
        self._fault_injection = fault_injection
        self._lock = threading.Lock()
        self._expiry_timer: Optional[threading.Timer] = None
        # Bumped on every inject and reset; a timer only disarms the window it armed.
        self._window = 0
        # tpu-sync clears hit counts on reset, so the last window's are kept here.
        self._last_window_hits: Dict[str, int] = {}

    def handle(self, req: RaidenFaultReqInput) -> RaidenFaultReqOutput:
        try:
            if req.op == "inject":
                self._inject(rules=req.rules, duration_s=req.duration_s)
            elif req.op == "reset":
                self._reset(reason="request")
            elif req.op != "status":
                raise ValueError(f"unknown raiden fault op {req.op!r}")
        except Exception as e:
            logger.error("RAIDEN_FAULT op=%s failed: %s", req.op, e)
            return self._output(success=False, message=str(e))
        return self._output(success=True)

    def _inject(self, *, rules: List[Dict[str, Any]], duration_s: float) -> None:
        if not rules:
            raise ValueError("inject needs at least one rule; use reset to disarm")
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        with self._lock:
            self._cancel_timer_locked()
            # Zero the hit counts first: inject_faults replaces rules but keeps counts.
            self._fault_injection.reset_faults()
            self._fault_injection.inject_faults(rules)
            self._window += 1
            self._last_window_hits = {}
            self._expiry_timer = threading.Timer(
                duration_s, self._expire, kwargs={"window": self._window}
            )
            self._expiry_timer.daemon = True
            self._expiry_timer.start()
        logger.warning(
            "RAIDEN_FAULT armed window=%d duration_s=%s rules=%s",
            self._window,
            duration_s,
            rules,
        )

    def _reset(self, *, reason: str) -> None:
        with self._lock:
            self._reset_locked(reason=reason)

    def _expire(self, *, window: int) -> None:
        with self._lock:
            if window != self._window:
                return
            self._reset_locked(reason="duration elapsed")

    def _reset_locked(self, *, reason: str) -> None:
        self._cancel_timer_locked()
        if self._fault_injection.has_active_injections():
            self._last_window_hits = dict(self._fault_injection.get_fault_status())
        self._fault_injection.reset_faults()
        self._window += 1
        logger.warning(
            "RAIDEN_FAULT disarmed (%s) hits=%s", reason, self._last_window_hits
        )

    def _cancel_timer_locked(self) -> None:
        if self._expiry_timer is not None:
            self._expiry_timer.cancel()
            self._expiry_timer = None

    def _output(self, *, success: bool, message: str = "") -> RaidenFaultReqOutput:
        with self._lock:
            armed = self._fault_injection.has_active_injections()
            hits = (
                dict(self._fault_injection.get_fault_status())
                if armed
                else dict(self._last_window_hits)
            )
        return RaidenFaultReqOutput(
            success=success, message=message, armed=armed, hits=hits
        )


def create_raiden_fault_controller() -> RaidenFaultController:
    try:
        from tpu_sync.api.torch import fault_injection
    except ImportError as e:
        raise RuntimeError(
            "SGLANG_ENABLE_RAIDEN_FAULT_INJECTION needs a tpu-sync torch wheel "
            "that ships tpu_sync.api.torch.fault_injection (google/tpu-sync#1033)"
        ) from e
    return RaidenFaultController(fault_injection=fault_injection)
