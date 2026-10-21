# SPDX-License-Identifier: Apache-2.0
"""Bounded parallel extraction using the unchanged per-case XProf attribution."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    opts = parser.parse_args()
    if not 1 <= opts.workers <= 4:
        parser.error("workers must be between 1 and 4")
    metrics = json.loads((opts.capture_root / "benchmark/metrics.json").read_text())
    if metrics["status"] != "completed_pending_accuracy":
        raise ValueError("Capture must be complete before matrix analysis")
    opts.output.mkdir(parents=True, exist_ok=True)

    def extract(case):
        destination = opts.output / case["name"]
        destination.mkdir(exist_ok=True)
        with (destination / "analysis.log").open("w") as output:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("profile_analyze.py")),
                    "--capture-root",
                    str(opts.capture_root),
                    "--output",
                    str(destination),
                    "--case",
                    case["name"],
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode:
            raise RuntimeError(f"{case['name']}: analysis exited {proc.returncode}")
        result = json.loads((destination / "summary.json").read_text())
        if len(result["cases"]) != 1:
            raise ValueError("Expected one analyzed case per worker")
        return result["cases"][0]

    rows = []
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures = {pool.submit(extract, case): case for case in metrics["cases"]}
        for future in as_completed(futures):
            rows.append(future.result())
            (opts.output / "matrix-progress.json").write_text(
                json.dumps(
                    {"completed": len(rows), "expected": len(futures), "cases": rows},
                    indent=2,
                )
            )
            print("MATRIX_ANALYSIS_PROGRESS", len(rows), len(futures), flush=True)
    order = {case["name"]: i for i, case in enumerate(metrics["cases"])}
    rows.sort(key=lambda row: order[row["case"]])
    compact = []
    for row in rows:
        case = metrics["cases"][order[row["case"]]]
        compact.append(
            {
                "case": row["case"],
                "mode": "decode" if case["name"].startswith("decode") else "prefill",
                "batch": case["batch"],
                "query_length": case["query_length"],
                "token_bucket": case["token_bucket"],
                "closed_pools": case["closed_pools"],
                "variants": {
                    label: {
                        device: {
                            key: value[key]
                            for key in (
                                "module_us",
                                "pallas_us",
                                "complete",
                                "observed_modules",
                                "pallas_names",
                            )
                        }
                        for device, value in devices.items()
                    }
                    for label, devices in row["variants"].items()
                },
            }
        )
    (opts.output / "compact.json").write_text(
        json.dumps(
            {
                "source_revision": metrics["source_revision"],
                "candidate_sha256": metrics["candidate_sha256"],
                "reference_sha256": metrics["reference_sha256"],
                "runtime": metrics["runtime"],
                "warmup": metrics["warmup"],
                "repeats": metrics["repeats"],
                "expected_cases": len(order),
                "complete": len(rows) == len(order),
                "cases": compact,
                "timing": "XProf device events only",
            },
            indent=2,
        )
        + "\n"
    )
    (opts.output / "summary.json").write_text(
        json.dumps(
            {
                "capture_metrics": metrics,
                "cases": rows,
                "complete": len(rows) == len(order),
                "timing": "XProf device events only",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
