# SPDX-License-Identifier: Apache-2.0
"""Read kpool captures on Falcon CPU with the installed XProf CLI.

Full traces, event tables and HLO remain remote. summary.json and the selected
sample tables are small evidence outputs. Host annotations never supply times.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": len(values),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def outermost(rows):
    """Remove exact duplicates and children, separately for each device."""
    out = []
    for row in sorted(rows, key=lambda r: (r["device"], r["start_ns"], -r["end_ns"])):
        if (
            out
            and row["device"] == out[-1]["device"]
            and row["end_ns"] <= out[-1]["end_ns"]
        ):
            continue
        out.append(row)
    return out


def union_ns(rows):
    total = 0
    end = None
    for row in sorted(rows, key=lambda r: (r["start_ns"], r["end_ns"])):
        start, stop = row["start_ns"], row["end_ns"]
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(end if end is not None else stop, stop)
    return total


def module_ops(events, module):
    # Events DB rounds picosecond starts/durations to integer nanoseconds.
    # An op sharing the module end can therefore end 1 ns after its parent.
    # This tolerance joins records; it does not alter either measured duration.
    return [
        event
        for event in events
        if event["device"] == module["device"]
        and event["category"] == "XLA Ops"
        and event["start_ns"] >= module["start_ns"] - 1
        and event["end_ns"] <= module["end_ns"] + 1
    ]


def fusion_context(graph, instruction):
    if not isinstance(graph, str):
        return None, [], "Fusion was not determined: graph text unavailable."
    instructions = {}
    computations = defaultdict(list)
    computation = None
    for line in graph.splitlines():
        header = re.match(r"^(?:ENTRY )?%?([\w.-]+)(?:\s*\(.*)?\s+\{$", line)
        if header:
            computation = header.group(1)
        item = re.match(r"\s+(?:ROOT )?%?([\w.-]+)\s*=\s*(.*)", line)
        if item:
            name, body = item.groups()
            instructions[name] = body
            opcode = re.search(r"\s([a-z][a-z0-9-]*)\(", body)
            if opcode and opcode.group(1) != "parameter":
                computations[computation].append(f"{name} ({opcode.group(1)})")
    body = instructions.get(instruction.lstrip("%"))
    if body is None:
        return (
            None,
            [],
            "Fusion was not determined: instruction absent from selected graph text.",
        )
    if not re.search(r"\bfusion\(", body):
        return None, [], "Selected HLO instruction is not fused."
    called = re.search(r"calls=%?([\w.-]+)", body)
    if not called or called.group(1) not in computations:
        return (
            instruction,
            [],
            "Fusion owner is known; fused computation members were unavailable.",
        )
    return (
        instruction,
        computations[called.group(1)],
        "Members are from the compiled fused computation.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", nargs="+", dest="selected_cases")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xprof = str(Path(sys.executable).with_name("xprof"))
    metrics = json.loads((args.capture_root / "benchmark/metrics.json").read_text())
    selected = metrics["cases"]
    if args.selected_cases:
        names = {case["name"] for case in selected}
        if set(args.selected_cases) - names:
            raise ValueError("Requested case absent from capture")
        selected = [case for case in selected if case["name"] in args.selected_cases]
    summary = {
        "xprof_version": importlib.metadata.version("xprof-nightly"),
        "capture_root": str(args.capture_root),
        "capture_metrics": metrics,
        "cases": [],
        "device_boundary_rounding_tolerance_ns": 1,
        "attribution": "Pair host sample marker order with serialized device module order, after validating counts and the alternating variant schedule; host timestamps and durations are not used. Identical executable fingerprints can alias module names.",
        "formula": "module_us=(end_ns-start_ns)/1000 on XLA Modules; Pallas_us=union of outermost kpool XLA Ops within that module /1000, separately for each device; percentiles use numpy linear interpolation",
    }
    for case in selected:
        start_time = time.monotonic()
        name = case["name"]
        dest = args.output / name
        dest.mkdir()
        profile = args.capture_root / "benchmark" / name / "xprof"
        sources = list(profile.rglob("*.xplane.pb"))
        if len(sources) != 1:
            raise RuntimeError(f"{name}: expected one host trace, got {sources}")
        source = sources[0]
        local = dest / "trace" / source.name
        local.parent.mkdir()
        shutil.copyfile(source, local)
        tool_outputs = []

        def run(
            tool,
            label=None,
            *,
            local=local,
            dest=dest,
            tool_outputs=tool_outputs,
            **kwargs,
        ):
            label = label or tool
            cmd = [xprof, tool, str(local)] + [f"--{k}={v}" for k, v in kwargs.items()]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=900, check=False
            )
            (dest / f"{label}.stdout.json").write_text(proc.stdout)
            (dest / f"{label}.stderr.txt").write_text(proc.stderr)
            tool_outputs.append(
                {
                    "tool": tool,
                    "path": str(dest / f"{label}.stdout.json"),
                    "returncode": proc.returncode,
                }
            )
            if proc.returncode:
                return {"error": proc.stderr, "returncode": proc.returncode}
            try:
                value = json.loads(proc.stdout)
            except json.JSONDecodeError:
                # Graph tools emit HLO text, despite the CLI's general JSON rule.
                if tool in ("get_graph_viewer", "get_hlo_neighborhood"):
                    return proc.stdout
                raise
            if isinstance(value, dict) and value.get("status") == "SAVED_TO_FILE":
                value = json.loads(Path(value["file_path"]).read_text())
                (dest / f"{label}.json").write_text(json.dumps(value))
            if (
                tool == "query_events_db"
                and isinstance(value, dict)
                and "path" in value
            ):
                value = json.loads(Path(value["path"]).read_text())
                (dest / f"{label}.json").write_text(json.dumps(value))
            return value

        run("get_kernel_stats", include_summary=True, limit=0)
        overview = run("get_overview")
        tops = run("get_top_hlo_ops", limit=0)
        run("create_events_db")
        coverage = run(
            "query_events_db",
            "coverage",
            query="SELECT device, category, COUNT(*) AS count FROM Events GROUP BY device, category ORDER BY device,category",
        )
        # Keep the complete device op/module input table remote for auditing.
        events = run(
            "query_events_db",
            "device_events",
            query="SELECT device,category,start_ns,end_ns,self_time_ns,hlo_op,hlo_module,kernel_name,source_line FROM Events WHERE device LIKE 'TPU:%' AND category IN ('XLA Modules','XLA Ops') ORDER BY device,start_ns,end_ns DESC",
        )
        if not isinstance(events, list) or not events:
            raise RuntimeError(f"{name}: no usable device events: {events}")
        markers = run(
            "query_events_db",
            "sample_markers",
            query="SELECT start_ns,kernel_name,trace_args FROM Events WHERE kernel_name LIKE 'sample/%' ORDER BY start_ns",
        )
        expected_labels = []
        for repetition in range(case["expected_samples_per_variant"]):
            order = list(case["variants"])
            expected_labels.extend(order if repetition % 2 == 0 else reversed(order))
        actual_labels = [
            r["kernel_name"].removeprefix(f"sample/{name}/") for r in markers
        ]
        if actual_labels != expected_labels:
            raise RuntimeError(f"{name}: sample marker order/count is incomplete")
        all_modules = outermost([r for r in events if r["category"] == "XLA Modules"])
        module_by_label = {label: [] for label in case["variants"]}
        for device in sorted({r["device"] for r in all_modules}):
            ordered = [r for r in all_modules if r["device"] == device]
            if len(ordered) != len(expected_labels):
                raise RuntimeError(
                    f"{name}/{device}: incomplete module events: {len(ordered)} != {len(expected_labels)}"
                )
            for row, label in zip(ordered, expected_labels, strict=True):
                module_by_label[label].append(row)
        records = {}
        graphs = {}
        sample_evidence = []
        for label, variant in case["variants"].items():
            module = variant["module_name"]
            modules = module_by_label[label]
            records[label] = {}
            for device in sorted({r["device"] for r in modules}):
                module_rows = [r for r in modules if r["device"] == device]
                samples = []
                op_totals = defaultdict(
                    lambda: {"count": 0, "self_us": 0.0, "inclusive_us": 0.0}
                )
                for index, row in enumerate(module_rows):
                    child = module_ops(events, row)
                    for op in child:
                        key = op["hlo_op"] or op["kernel_name"]
                        totals = op_totals[key]
                        totals["count"] += 1
                        totals["self_us"] += (op["self_time_ns"] or 0) / 1000
                        totals["inclusive_us"] += (op["end_ns"] - op["start_ns"]) / 1000
                    pallas = outermost(
                        [
                            r
                            for r in child
                            if re.search(
                                r"^%?kpool_(?:decode_store|compress_store)",
                                r.get("hlo_op") or r["kernel_name"].split(" = ", 1)[0],
                            )
                        ]
                    )
                    sample = {
                        "variant": label,
                        "device": device,
                        "index": index,
                        "observed_module": row["kernel_name"],
                        "start_ns": row["start_ns"],
                        "end_ns": row["end_ns"],
                        "module_us": (row["end_ns"] - row["start_ns"]) / 1000,
                        "pallas_us": union_ns(pallas) / 1000 if pallas else None,
                        "pallas_event_count": len(pallas),
                        "pallas_names": sorted(
                            {r["hlo_op"] or r["kernel_name"] for r in pallas}
                        ),
                        "ops_union_us": union_ns(child) / 1000,
                    }
                    samples.append(sample)
                    sample_evidence.append(sample)
                if label != "reference" and any(
                    s["pallas_event_count"] != 1 for s in samples
                ):
                    raise RuntimeError(
                        f"{name}/{label}/{device}: expected one outer Pallas event in every module"
                    )
                records[label][device] = {
                    "module_us": distribution([s["module_us"] for s in samples]),
                    "pallas_us": distribution(
                        [s["pallas_us"] for s in samples if s["pallas_us"] is not None]
                    ),
                    "ops_union_us": distribution([s["ops_union_us"] for s in samples]),
                    "device_op_totals": dict(
                        sorted(op_totals.items(), key=lambda item: -item[1]["self_us"])
                    ),
                    "observed_modules": sorted({s["observed_module"] for s in samples}),
                    "pallas_event_counts": sorted(
                        {s["pallas_event_count"] for s in samples}
                    ),
                    "pallas_names": sorted(
                        {n for s in samples for n in s["pallas_names"]}
                    ),
                    "expected_samples": case["expected_samples_per_variant"],
                    "complete": len(samples) == case["expected_samples_per_variant"],
                }
            # XLA may deduplicate identical variants. Ask for the name that
            # the trace records, not a name erased by executable reuse.
            observed = sorted(
                {r["kernel_name"].removeprefix("HLO Module:") for r in modules}
            )
            if len(observed) != 1:
                raise RuntimeError(
                    f"{name}/{label}: changing executable identity: {observed}"
                )
            if label == "reference" and "_reference(" not in observed[0]:
                raise RuntimeError(
                    f"{name}: serialized attribution does not match reference module"
                )
            graphs[observed[0]] = run(
                "get_graph_viewer",
                f"graph_{label}",
                module_name=observed[0],
                output_type="short_txt",
            )
        hot_ops = []
        for op in tops.get("top_by_time", []):
            if op.get("category") == "IDLE":
                continue
            match = re.match(r"by_program/([^/]+)/[^/]+/([^ ]+)", op["name"])
            if not match:
                continue
            module, instruction = match.groups()
            graph = graphs.get(module)
            fusion, fused_with, fusion_note = fusion_context(graph, instruction)
            hot_ops.append(
                {
                    "trace_name": op["name"],
                    "hlo_op": instruction,
                    "source": f"{op['source_file']}:{op['source_line']}"
                    if op.get("source_file")
                    else "empty in trace",
                    "per_call_us": op["total_self_time_ms"] * 1000 / op["occurrences"]
                    if op.get("occurrences")
                    else "N/A",
                    "fusion": fusion,
                    "fused_with": fused_with,
                    "source_tool": "get_top_hlo_ops, get_graph_viewer",
                    "note": fusion_note
                    + " Per-call is measured self-time divided by occurrences. XProf may merge identical executable variants in this op table; sample attribution uses serialized module order.",
                }
            )
            if len(hot_ops) == 15:
                break
        (dest / "hot_ops.json").write_text(json.dumps(hot_ops, indent=2))
        (dest / "samples.json").write_text(json.dumps(sample_evidence, indent=2))
        case_summary = {
            "case": name,
            "trace_path": str(source),
            "trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "session_id": source.parent.name,
            "coverage": coverage,
            "variants": records,
            "hot_ops": hot_ops,
            "run_environment": overview.get("run_environment"),
            "read_duration_seconds": time.monotonic() - start_time,
            "tool_outputs": tool_outputs,
        }
        summary["cases"].append(case_summary)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("CASE_ANALYZED", name, flush=True)
    print("ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
