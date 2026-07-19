#!/usr/bin/env python3
"""Parse a DFlash-on-TPU compile-checker server log into a structured summary.

Turns the raw lines emitted by a server launched with
``SGLANG_TPU_CHECK_RECOMPILATION=1`` (and optionally
``SGLANG_DFLASH_VERIFY_SHAPE_TRACE=1``) into a machine-readable dict, replacing
the ad-hoc grep pipelines used to build compile-cleanliness evidence.

Recognized lines (exact formats):
  * ``DFLASH xla-attrib: phase=<name> compiled <N> new XLA program(s) (<a> -> <b>)``
    -> per-phase first-touch/compile totals.
  * ``... recompilation checker] <N> new XLA program ...``
    -> total global recompilation-checker deltas.
  * ``DFLASH verify-shape-trace: ... free_locs=<N> ... xla_delta_since_prev_verify=<N>``
    -> the eager verify-glue residual summary (max free_locs, total/steps delta).
  * ``#running-req: <N>`` -> max real running batch observed.

Usage:
  dflash_parse_checker_log.py <server_log> [--from-line N]   # JSON + table
"""
import argparse
import json
import re
import sys
from typing import Dict, List

_ATTRIB = re.compile(
    r"DFLASH xla-attrib: phase=([a-z_]+) compiled (\d+) new XLA program"
)
_CHECKER = re.compile(r"recompilation checker\] (\d+) new XLA program")
_VERIFY = re.compile(
    r"DFLASH verify-shape-trace:.*?free_locs=(\d+).*?"
    r"xla_delta_since_prev_verify=(\d+)"
)
_RUNNING = re.compile(r"#running-req: (\d+)")


def parse_checker_log(lines) -> Dict:
    """Parse an iterable of log lines into a structured compile-checker summary.

    Pure (no I/O) so it is hermetically CPU-testable. Returns:
      {
        "phase_compiles":   {phase: total_new_xla_programs},
        "checker_total":    int,   # sum of global recompilation-checker deltas
        "max_running":      int,   # max #running-req seen
        "verify_glue": {           # eager verify-glue residual (free_locs-keyed)
            "steps": int, "total_delta": int, "max_free_locs": int,
            "recompiling_steps": int,   # verify steps with a nonzero delta
        },
      }
    """
    phase_compiles: Dict[str, int] = {}
    checker_total = 0
    max_running = 0
    v_steps = v_total = v_max_free = v_recompiling = 0
    for line in lines:
        m = _ATTRIB.search(line)
        if m:
            phase_compiles[m.group(1)] = phase_compiles.get(m.group(1), 0) + int(
                m.group(2)
            )
        m = _CHECKER.search(line)
        if m:
            checker_total += int(m.group(1))
        m = _RUNNING.search(line)
        if m:
            max_running = max(max_running, int(m.group(1)))
        m = _VERIFY.search(line)
        if m:
            free_locs, delta = int(m.group(1)), int(m.group(2))
            v_steps += 1
            v_total += delta
            v_max_free = max(v_max_free, free_locs)
            if delta:
                v_recompiling += 1
    return {
        "phase_compiles": phase_compiles,
        "checker_total": checker_total,
        "max_running": max_running,
        "verify_glue": {
            "steps": v_steps,
            "total_delta": v_total,
            "max_free_locs": v_max_free,
            "recompiling_steps": v_recompiling,
        },
    }


def _format_table(summary: Dict) -> str:
    rows: List[str] = []
    rows.append(f"max #running-req: {summary['max_running']}")
    rows.append(f"recompilation-checker total new XLA: {summary['checker_total']}")
    rows.append("per-phase compiles (xla-attrib):")
    for phase, n in sorted(
        summary["phase_compiles"].items(), key=lambda kv: -kv[1]
    ):
        rows.append(f"  {phase}: {n}")
    vg = summary["verify_glue"]
    rows.append(
        "verify_glue residual: "
        f"{vg['recompiling_steps']}/{vg['steps']} steps recompiled, "
        f"total_delta={vg['total_delta']}, max_free_locs={vg['max_free_locs']}"
    )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("server_log")
    ap.add_argument(
        "--from-line", type=int, default=0, help="skip the first N lines (window)"
    )
    args = ap.parse_args()
    with open(args.server_log) as f:
        lines = f.readlines()[args.from_line :]
    summary = parse_checker_log(lines)
    print(json.dumps(summary, indent=2))
    print("-" * 60)
    print(_format_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
