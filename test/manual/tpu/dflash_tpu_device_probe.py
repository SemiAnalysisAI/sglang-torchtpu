#!/usr/bin/env python3
"""Non-destructive TPU device-state probe for the reboot-gated DFlash bs=16 artifact.

The live concurrent bs=16 parity artifact is collected sequentially on a 4-chip
slice and has repeatedly been blocked by accumulated TPU multi-worker session-state
degradation (single-chip ops work while a TP=4 slice init errors at shard-load with
``RuntimeError: TPU device returned error``), which only a VM reboot clears.

This tool gathers a CURRENT, non-destructive reading of that blocker so the
"reboot-pending" status rests on present evidence, not a stale assumption:
  * stale SGLang server holders (must be none),
  * occupied serving ports (must be free),
  * ``/dev/vfio`` (PJRT) holders (must be none),
  * TPU device-node presence/count under ``/dev/vfio`` (must equal the slice size),
  * the documented SAFE single-chip TPU op (``torch.ones(...,device='tpu')+1``).

It deliberately does NOT launch a multi-worker TP=4 server: that is resource-heavy
on shared hardware and stalls/errors before a reboot. A passing single-chip probe is
NECESSARY-BUT-NOT-SUFFICIENT for recovery (it is consistent with both a recovered and
a still-multi-worker-degraded device), so the interpreter reports the multi-worker
status as UNVERIFIED rather than claiming recovery.

A scanner that cannot run (e.g. no non-interactive privilege for ``/dev/vfio``)
yields an explicit UNKNOWN, and the interpreter then REFUSES the healthy verdict
rather than silently treating unprobed state as clear.

Usage:
  dflash_tpu_device_probe.py            # run the live checks + print the verdict
  dflash_tpu_device_probe.py --no-op    # holders/ports/vfio/nodes only (skip op)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional


def interpret_probe(
    sglang_holders: List[str],
    ports_in_use: List[int],
    vfio_holders: Optional[List[str]],
    tpu_nodes: Optional[List[int]],
    single_chip_ok: Optional[bool],
    expected_nodes: int = 4,
) -> Dict:
    """Pure interpretation of the probe results (no I/O), per the prior multi-worker
    degradation signature. Returns a structured verdict:

      * ``state``: "holders-present" | "vfio-holders-present" | "tpu-nodes-mismatch"
                   | "holder-state-unknown" | "single-chip-unprobed"
                   | "single-chip-failed" | "single-chip-healthy-no-holders"
      * ``multiworker_status``: always "UNVERIFIED" here -- a single-chip pass cannot
        settle the TP=4 slice-init question without an actual multi-worker launch.
      * ``blocker`` / ``recommendation``: the current reading and next action.

    Order-sensitive: SGLang/port holders first (a live holder invalidates the op
    probe), then vfio/PJRT holders, then missing/wrong-count/unprobed TPU nodes,
    then the single-chip op, then the residual multi-worker question. A vfio or node
    scanner that could not run passes ``None`` (UNKNOWN); the healthy verdict is
    emitted ONLY when every holder/node check is affirmatively clear."""
    if sglang_holders or ports_in_use:
        return _verdict(
            "holders-present",
            "stale SGLang holders / occupied serving ports remain",
            "clear holders by PORT (fuser -k <port>/tcp), never pkill -f "
            "launch_server, then re-probe",
        )
    if vfio_holders:  # non-empty list -> a process holds a /dev/vfio group
        return _verdict(
            "vfio-holders-present",
            "a process holds a /dev/vfio (PJRT) group",
            "identify and stop the holder (by PORT if it is a server), then re-probe",
        )
    if tpu_nodes is not None and len(tpu_nodes) != expected_nodes:
        return _verdict(
            "tpu-nodes-mismatch",
            f"found {len(tpu_nodes)} TPU device node(s), expected {expected_nodes}",
            "device nodes missing/mismatched -- owner reboot and re-verify the slice",
        )
    if vfio_holders is None or tpu_nodes is None:
        return _verdict(
            "holder-state-unknown",
            "could not probe /dev/vfio holders and/or TPU device nodes (no privilege)",
            "re-run with sufficient privilege (sudo -n lsof); do NOT treat unprobed "
            "holder state as clear",
        )
    if single_chip_ok is None:
        return _verdict(
            "single-chip-unprobed",
            "single-chip TPU op was not probed",
            "run the single-chip op probe to gauge basic device health",
        )
    if not single_chip_ok:
        return _verdict(
            "single-chip-failed",
            "even a single-chip TPU op fails -- deeper device/env fault",
            "a hard device fault (beyond multi-worker session degradation); owner "
            "reboot required and re-verify a single-chip op before any launch",
        )
    return _verdict(
        "single-chip-healthy-no-holders",
        "environment clean (no SGLang/port/vfio holders, TPU nodes present) + "
        "single-chip healthy; the ONLY residual risk is accumulated multi-worker "
        "(TP=4) session-state degradation, which a single-chip pass cannot rule out",
        "owner VM reboot, then run the sequential DFlash-TP4 / no-spec-TP4 collect "
        "+ consolidated report flow; OR sanction one bounded TP=4 launch (kill by "
        "PORT) to test multi-worker self-recovery -- a shard-load 'TPU device "
        "returned error' confirms the reboot is still required",
    )


def _verdict(state: str, blocker: str, recommendation: str) -> Dict:
    return {
        "state": state,
        "multiworker_status": "UNVERIFIED",
        "blocker": blocker,
        "recommendation": recommendation,
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


def _scan_tpu_nodes(vfio_dir: str = "/dev/vfio") -> List[int]:
    """Numeric TPU vfio group nodes present under /dev/vfio (read-only, no sudo)."""
    try:
        return sorted(
            int(n) for n in os.listdir(vfio_dir) if n.isdigit()
        )
    except Exception:
        return []


def _scan_vfio_holders() -> Optional[List[str]]:
    """Processes holding a /dev/vfio group, via non-interactive ``sudo -n lsof``.
    Returns None (UNKNOWN) if it cannot probe with privilege, [] if it ran and found
    no holders, or the holder lines. Never claims "clear" without actually running."""
    nodes = sorted(glob.glob("/dev/vfio/[0-9]*"))
    if not nodes:
        return []
    try:
        r = subprocess.run(
            ["sudo", "-n", "lsof", *nodes],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return None
    err = (r.stderr or "")
    # sudo -n was blocked (needs a password) -> could not probe -> UNKNOWN.
    if "password is required" in err or "a terminal is required" in err:
        return None
    out = (r.stdout or "").splitlines()
    lines = [l for l in out if l.strip()]
    if not lines:
        return []  # lsof ran, no open handles on any group
    return [l for l in lines if not l.startswith("COMMAND")]


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
        "--no-op", action="store_true", help="skip the single-chip op (holders/nodes only)"
    )
    ap.add_argument("--expected-nodes", type=int, default=4, help="TPU slice size")
    args = ap.parse_args()
    holders = _scan_sglang_holders()
    ports = _scan_ports()
    vfio_holders = _scan_vfio_holders()
    tpu_nodes = _scan_tpu_nodes()
    single = None if args.no_op else _single_chip_op()
    verdict = interpret_probe(
        holders, ports, vfio_holders, tpu_nodes, single, args.expected_nodes
    )
    result = {
        "sglang_holders": holders,
        "ports_in_use": ports,
        "vfio_holders": vfio_holders,
        "tpu_nodes": tpu_nodes,
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
