# SPDX-License-Identifier: Apache-2.0
"""Pre-launch TPU topology discovery.

The probe runs in a short-lived subprocess so PJRT is released before SGLang
spawns its long-lived scheduler workers.  ``TPU_WORKER_ID`` is deliberately not
used: it is an orchestration identifier and is not guaranteed to follow the
physical TPU topology.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_RESULT_PREFIX = "SGLANG_TPU_TOPOLOGY="
_RANK_PREFIX = "SGLANG_TPU_NODE_RANK="


def _query_topological_node_rank() -> dict:
    """Query this host's coordinates and return its topological node rank."""
    from torch_tpu._internal.distributed import tpu_distributed
    from torch_tpu._internal.distributed.launchers.multihost_wrapper import (
        calculate_topological_node_rank,
    )

    topology = os.environ.get("TORCH_TPU_TOPOLOGY") or os.environ.get("TPU_TOPOLOGY")
    if not topology:
        raise RuntimeError(
            "TPU topology discovery requires TORCH_TPU_TOPOLOGY or TPU_TOPOLOGY."
        )

    coords = list(tpu_distributed.local_device_coords())
    if not coords:
        raise RuntimeError("TorchTPU returned no local TPU device coordinates.")

    node_rank = calculate_topological_node_rank(coords, topology)
    return {"node_rank": int(node_rank), "coords": coords, "topology": topology}


def probe_topological_node_rank(timeout: float = 120.0) -> dict:
    """Run coordinate discovery in a disposable Python process."""
    result = subprocess.run(
        [sys.executable, "-m", __name__, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode == 0:
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(_RESULT_PREFIX):
                return json.loads(line.removeprefix(_RESULT_PREFIX))

    detail = result.stderr.strip() or result.stdout.strip() or "no output"
    raise RuntimeError(
        "Failed to discover the topological TPU node rank in a short-lived "
        f"client (exit code {result.returncode}): {detail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = _query_topological_node_rank()
    if args.json:
        print(f"{_RESULT_PREFIX}{json.dumps(result, separators=(',', ':'))}")
    else:
        print(f"{_RANK_PREFIX}{result['node_rank']}")


if __name__ == "__main__":
    main()
