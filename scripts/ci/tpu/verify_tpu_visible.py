"""Fail fast if the runner lost its TPU.

Imported as the last step of tpu_ci_install_dependency.sh. A miss here
means the dependency install completed but the TPU is unreachable, which
otherwise surfaces minutes later as a confusing test failure.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"FAIL: torch is not importable: {exc}", file=sys.stderr)
        return 1

    try:
        import torch_tpu  # noqa: F401
    except ImportError as exc:
        print(f"FAIL: torch_tpu is not importable: {exc}", file=sys.stderr)
        return 1

    is_available_fn = getattr(getattr(torch, "tpu", None), "is_available", None)
    if is_available_fn is None:
        print(
            "FAIL: torch.tpu.is_available is not exposed; torch_tpu may not be "
            "registered with this torch build.",
            file=sys.stderr,
        )
        return 1

    if not is_available_fn():
        print("FAIL: torch.tpu.is_available() returned False.", file=sys.stderr)
        return 1

    device_count = getattr(torch.tpu, "device_count", lambda: -1)()
    print(f"OK: torch.tpu.is_available()=True, device_count={device_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
