#!/usr/bin/env python3
"""Non-destructive TPU device-state probe for the reboot-gated DFlash bs=16 artifact.

The live concurrent bs=16 parity artifact is collected sequentially on a 4-chip
slice and has repeatedly been blocked by accumulated TPU multi-worker session-state
degradation (single-chip ops work while a TP=4 slice init errors at shard-load with
``RuntimeError: TPU device returned error``), which only a VM reboot clears.

This tool gathers a CURRENT, non-destructive reading of that blocker so the
"reboot-pending" status rests on present evidence, not a stale assumption:
  * stale SGLang holders (must be none),
  * occupied serving ports (must be free),
  * the documented SAFE single-chip TPU op probe (``torch.ones(...,device='tpu')+1``).

It deliberately does NOT launch a multi-worker TP=4 server: that is resource-heavy
on shared hardware and stalls/errors before a reboot. A passing single-chip probe is
NECESSARY-BUT-NOT-SUFFICIENT for recovery (it is consistent with both a recovered and
a still-multi-worker-degraded device), so the interpreter reports the multi-worker
status as UNVERIFIED rather than claiming recovery.

Usage:
  dflash_tpu_device_probe.py            # run the live checks + print the verdict
"""
import argparse
import json
import subprocess
import sys
from typing import Dict, List, Optional


def interpret_probe(
    sglang_holders: List[str],
    ports_in_use: List[int],
    single_chip_ok: Optional[bool],
) -> Dict:
    """Pure interpretation of the probe results (no I/O), per the R28 degradation
    signature. Returns a structured verdict:

      * ``state``: "holders-present" | "single-chip-failed" |
                   "single-chip-healthy-no-holders" | "single-chip-unprobed"
      * ``multiworker_status``: always "UNVERIFIED" here -- a single-chip pass cannot
        settle the TP=4 slice-init question without an actual multi-worker launch.
      * ``blocker``: one-line current blocker reading.
      * ``recommendation``: the concrete next action.

    Order matters: holders/ports must be clear FIRST (a live holder invalidates the
    op probe), then the single-chip op, then the residual multi-worker question."""
    if sglang_holders or ports_in_use:
        return {
            "state": "holders-present",
            "multiworker_status": "UNVERIFIED",
            "blocker": "stale SGLang holders / occupied serving ports remain",
            "recommendation": (
                "clear holders by PORT (fuser -k <port>/tcp), never pkill -f "
                "launch_server, then re-probe"
            ),
        }
    if single_chip_ok is None:
        return {
            "state": "single-chip-unprobed",
            "multiworker_status": "UNVERIFIED",
            "blocker": "single-chip TPU op was not probed",
            "recommendation": "run the single-chip op probe to gauge basic device health",
        }
    if not single_chip_ok:
        return {
            "state": "single-chip-failed",
            "multiworker_status": "UNVERIFIED",
            "blocker": "even a single-chip TPU op fails -- deeper device/env fault",
            "recommendation": (
                "a hard device fault (beyond the R28 multi-worker session degradation); "
                "owner reboot required and re-verify a single-chip op before any launch"
            ),
        }
    return {
        "state": "single-chip-healthy-no-holders",
        "multiworker_status": "UNVERIFIED",
        "blocker": (
            "environment clean + single-chip healthy; the ONLY residual risk is "
            "accumulated multi-worker (TP=4) session-state degradation (R28), which a "
            "single-chip pass cannot rule out"
        ),
        "recommendation": (
            "owner VM reboot, then run the sequential DFlash-TP4 / no-spec-TP4 collect "
            "+ consolidated report flow; OR sanction one bounded TP=4 launch (kill by "
            "PORT) to test multi-worker self-recovery -- a shard-load 'TPU device "
            "returned error' confirms the reboot is still required"
        ),
    }


def _scan_sglang_holders() -> List[str]:
    """Real SGLang server processes (excluding this probe's own command line)."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "sglang"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return []
    holders = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # Exclude our own probe command line (it contains 'sglang' via the path).
        if "dflash_tpu_device_probe" in line or "pgrep" in line:
            continue
        if "launch_server" in line or "sglang.launch_server" in line or "scheduler" in line:
            holders.append(line.strip())
    return holders


def _scan_ports(lo: int = 30000, hi: int = 30060) -> List[int]:
    try:
        out = subprocess.run(
            ["ss", "-ltn"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return []
    ports = []
    for line in out.splitlines():
        for p in range(lo, hi + 1):
            if f":{p} " in line or line.rstrip().endswith(f":{p}"):
                ports.append(p)
    return sorted(set(ports))


def _single_chip_op() -> Optional[bool]:
    try:
        import torch
        import torch_tpu  # noqa: F401

        x = torch.ones(4, device="tpu") + 1
        return x.cpu().tolist() == [2.0, 2.0, 2.0, 2.0]
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the probe verdict JSON here")
    ap.add_argument(
        "--no-op", action="store_true", help="skip the single-chip op (holders/ports only)"
    )
    args = ap.parse_args()
    holders = _scan_sglang_holders()
    ports = _scan_ports()
    single = None if args.no_op else _single_chip_op()
    verdict = interpret_probe(holders, ports, single)
    result = {
        "sglang_holders": holders,
        "ports_in_use": ports,
        "single_chip_ok": single,
        **verdict,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
