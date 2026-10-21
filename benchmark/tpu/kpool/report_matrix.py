# SPDX-License-Identifier: Apache-2.0
"""Export public Q/B tables from compact, validated device timing results."""

import argparse
import csv
import json
from pathlib import Path

import shape_matrix


def read_rows(paths):
    expected = {
        (c["mode"], c["query_length"], c["batch"]): c for c in shape_matrix.cases()
    }
    rows = {}
    identities = set()
    for path in paths:
        capture = json.loads(path.read_text())
        if not capture["complete"]:
            raise ValueError("Incomplete capture analysis")
        identities.add(
            (
                capture["candidate_sha256"],
                capture["reference_sha256"],
                json.dumps(capture["runtime"], sort_keys=True),
                capture["repeats"],
                capture["warmup"],
            )
        )
        for case in capture["cases"]:
            key = (case["mode"], case["query_length"], case["batch"])
            if key not in expected or key in rows:
                raise ValueError(f"Unexpected or duplicate case: {key}")
            if case["token_bucket"] != expected[key]["tokens"]:
                raise ValueError(f"Wrong token bucket: {key}")
            variants = case["variants"]
            if set(variants) != {"reference", "fused"}:
                raise ValueError(f"Expected reference/fused pair: {key}")
            if any(len(devices) != 1 for devices in variants.values()):
                raise ValueError("Expected one measured device per variant")
            if set(variants["reference"]) != set(variants["fused"]):
                raise ValueError("Reference and fused devices differ")
            row = {
                "mode": key[0],
                "Q": key[1],
                "B": key[2],
                "T": case["token_bucket"],
                "closed_pools": case["closed_pools"],
                "source_revision": capture["source_revision"],
                "accuracy": "not_run",
            }
            for label in ("reference", "fused"):
                value = next(iter(variants[label].values()))
                timing = value["module_us"]
                if not value["complete"] or timing["count"] != capture["repeats"]:
                    raise ValueError(f"Incomplete module samples: {key}/{label}")
                for percentile in ("p10", "p50", "p90"):
                    row[f"{label}_{percentile}_us"] = timing[percentile]
                row[f"{label}_samples"] = timing["count"]
                if label == "fused":
                    pallas = value["pallas_us"]
                    if not pallas or pallas["count"] != capture["repeats"]:
                        raise ValueError(f"Incomplete Pallas samples: {key}")
                    row["pallas_p50_us"] = pallas["p50"]
            row["speedup"] = row["reference_p50_us"] / row["fused_p50_us"]
            row["latency_change_pct"] = (1 / row["speedup"] - 1) * 100
            rows[key] = row
    if set(rows) != set(expected):
        raise ValueError(f"Missing {len(set(expected) - set(rows))} matrix cases")
    if len(identities) != 1:
        raise ValueError("Capture source, runtime, or sampling recipes differ")
    return [rows[key] for key in expected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--accuracy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    rows = read_rows(opts.input)
    captures = [json.loads(path.read_text()) for path in opts.input]
    result = None
    if opts.accuracy:
        result = json.loads(opts.accuracy.read_text())
        if any(
            c[key] != result[key]
            for c in captures
            for key in ("candidate_sha256", "reference_sha256", "runtime")
        ):
            raise ValueError("Accuracy and performance sources or runtimes differ")
        checks = {
            (c["mode"], c["query_length"], c["batch"]): c
            for c in result["cases"]
            if c["edge"] == "uniform"
        }
        for row in rows:
            case = checks.get((row["mode"], row["Q"], row["B"]))
            if case is not None:
                row["accuracy"] = "pass" if case["passed"] else "FAIL"
    opts.output.mkdir(parents=True, exist_ok=True)
    if result is not None:
        with (opts.output / "accuracy.csv").open("w", newline="") as handle:
            fields = (
                "mode",
                "batch",
                "query_length",
                "tokens",
                "edge",
                "passed",
                "mismatched_bytes",
            )
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {key: case.get(key) for key in fields} for case in result["cases"]
            )
    with (opts.output / "matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Q/B device timing matrix",
        "",
        "All times are XProf device module medians in microseconds. Each cell",
        "is **JAX / fused (speedup)**. A speedup below 1 means a regression.",
        "Pallas event time is a subset of module time. The CSV includes p10/p90,",
        "Pallas medians, sample counts, source revisions, and accuracy status.",
        "A dagger (†) marks a shape that failed exact-byte validation.",
        "",
        "Fixed K=4, D=128, raw page=128, raw dtype=BF16, raw pages=14,849.",
        "Decode uses T=B*Q. Prefill uses T=max(64,B*Q), with B*Q<=8192.",
        "Prefill starts at sequence position zero: Q=1/2 closes no K4 pool and",
        "measures metadata/call overhead. The CSV records the closed-pool count.",
        "These are single-device module tests; serving and end-to-end are separate.",
        "",
        "Runtime: "
        + ", ".join(
            f"{name} {captures[0]['runtime'][name]}"
            for name in ("jax", "jaxlib", "libtpu", "numpy")
        )
        + ".",
        f"Warmups: {captures[0]['warmup']}; recorded calls per variant: {captures[0]['repeats']}.",
        "",
    ]
    if result is None:
        lines[2:2] = ["**Accuracy has not been validated for this matrix.**", ""]
    elif not result["passed"]:
        lines[2:2] = [
            "**The candidate does not pass the full exact-byte accuracy gate.**",
            "Timing ratios below are measurements of this candidate, not validated replacements.",
            "",
        ]
    for mode in ("decode", "prefill"):
        selected = [row for row in rows if row["mode"] == mode]
        lookup = {(row["Q"], row["B"]): row for row in selected}
        lines += [
            f"## {mode.capitalize()}",
            "",
            "| Q | " + " | ".join(f"B={b}" for b in shape_matrix.BATCHES) + " |",
            "|---:|" + "---:|" * len(shape_matrix.BATCHES),
        ]
        for q in sorted({row["Q"] for row in selected}):
            cells = []
            for b in shape_matrix.BATCHES:
                row = lookup.get((q, b))
                cells.append(
                    f"{row['reference_p50_us']:.2f} / {row['fused_p50_us']:.2f} "
                    f"({row['speedup']:.2f}x){'†' if row['accuracy'] == 'FAIL' else ''}"
                    if row
                    else "—"
                )
            lines.append(f"| {q} | " + " | ".join(cells) + " |")
        lines.append("")
    counts = {
        state: sum(r["accuracy"] == state for r in rows)
        for state in ("pass", "FAIL", "not_run")
    }
    lines += [
        (
            f"Uniform-shape exact-byte checks: {counts['pass']} passed, "
            f"{counts['FAIL']} failed, {counts['not_run']} not run."
        ),
        "",
    ]
    if result is not None:
        passed = sum(case["passed"] for case in result["cases"])
        lines += [
            (
                f"Full accuracy gate (including edge cases): {passed}/{len(result['cases'])} "
                f"passed, {len(result['cases'])}/{result['expected_cases']} executed."
            ),
            "See accuracy.csv for each case. A failed accuracy row is not a validated replacement.",
            "",
        ]
    (opts.output / "matrix.md").write_text("\n".join(lines))
    print(json.dumps({"cases": len(rows), "accuracy": counts}))


if __name__ == "__main__":
    main()
