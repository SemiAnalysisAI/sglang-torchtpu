#!/usr/bin/env python3
"""Build the comparison table from each arm's client output and server log.

    python3 compare.py A=client_a.jsonl:server_a.log[:metrics_a.txt] B=... [--start-A "YYYY-MM-DD HH:MM:SS"]

Per arm: the client summary (last line of agentic_multiturn_client.py output),
the server log (scheduler `Prefill batch` / `Decode batch` lines of TP rank 0),
and optionally a final /metrics dump for eviction time. Only server lines inside
[client start, client start + duration) count. The client start is the summary's
`start_local`, which is in the server log's clock when both run on one host;
--start-<ARM> overrides it. Stdlib only.
"""

import json
import re
import statistics
import sys
from datetime import datetime

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0[ \]]")
FIELDS = (("new", "new-token"), ("device", "hbm-hit-token"), ("host", "host-hit-token"))


def parse_args(argv):
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        sys.exit(0)
    arms, starts = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--start-"):
            if i + 1 >= len(argv):
                sys.exit(f"{a} needs a value")
            starts[a[len("--start-") :]] = argv[i + 1]
            i += 2
            continue
        if "=" not in a:
            sys.exit(f"expected NAME=client.jsonl:server.log[:metrics.txt], got {a!r}")
        name, paths = a.split("=", 1)
        parts = paths.split(":")
        if len(parts) not in (2, 3):
            sys.exit(f"expected client.jsonl:server.log[:metrics.txt] for {name}")
        arms.append((name, parts))
        i += 1
    if not arms:
        sys.exit(__doc__)
    return arms, starts


def client_summary(path):
    summary = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('{"summary"'):
                summary = json.loads(line)["summary"]
    if summary is None:
        sys.exit(f"{path}: no summary line (did the client finish?)")
    return summary


def server_stats(path, start, duration):
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    total = dict.fromkeys(("new", "device", "host"), 0)
    buckets, decode = {}, []
    with open(path, errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            dt = (
                datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") - t0
            ).total_seconds()
            if not 0 <= dt < duration:
                continue
            if "Prefill batch" in line:
                values = {}
                for key, field in FIELDS:
                    mm = re.search(rf"#{field}: (\d+)", line)
                    if not mm:
                        sys.exit(
                            f"{path}: Prefill batch line without #{field}: {line[:200]}"
                        )
                    values[key] = int(mm.group(1))
                b = buckets.setdefault(int(dt // 600), dict.fromkeys(total, 0))
                for key, v in values.items():
                    b[key] += v
                    total[key] += v
            elif "Decode batch" in line:
                mm = re.search(r"gen throughput \(token/s\): ([\d.]+)", line)
                if mm:
                    decode.append(float(mm.group(1)))
    if not buckets:
        sys.exit(
            f"{path}: no TP0 Prefill batch lines in [{start}, +{duration} s); check the start time"
        )
    return total, buckets, decode


def hit(b):
    n = sum(b.values())
    return 100 * (b["device"] + b["host"]) / n if n else float("nan")


def eviction_seconds(path):
    with open(path, errors="replace") as f:
        for line in f:
            if (
                line.startswith("sglang:eviction_duration_seconds_sum")
                and 'tp_rank="0"' in line
            ):
                # Value is the first field after the labels; an optional timestamp may follow.
                return float(line[line.index("}") + 1 :].split()[0])
    return None


def main():
    arms, starts = parse_args(sys.argv[1:])
    rows = []
    for name, parts in arms:
        s = client_summary(parts[0])
        start = starts.get(name) or s.get("start_local")
        if not start:
            sys.exit(
                f"{name}: no start_local in the client summary; pass --start-{name}"
            )
        if s["errors"] or s.get("worker_crashes"):
            print(
                f"WARNING: {name}: {s['errors']} errors, worker crashes {s.get('worker_crashes')}; "
                "this run does not count",
                file=sys.stderr,
            )
        total, buckets, decode = server_stats(parts[1], start, s["duration_s"])
        n_buckets = -(-s["duration_s"] // 600)
        rows.append(
            {
                "arm": name,
                "requests": s["requests"],
                "errors": s["errors"],
                "req_s": s["requests_per_s"],
                "hit": hit(total),
                "split": [
                    100 * total[k] / max(1, sum(total.values()))
                    for k in ("device", "host", "new")
                ],
                "per10": [
                    hit(buckets[i]) if i in buckets else float("nan")
                    for i in range(n_buckets)
                ],
                "decode": statistics.mean(decode) if decode else float("nan"),
                "main": s["latency_s"]["main"],
                "evict": eviction_seconds(parts[2]) if len(parts) == 3 else None,
                "prefill_only": s.get("prefill_only", False),
                "latency": s["latency_s"],
                "input_tps": s.get("input_tokens_per_s"),
                "uncached_tps": s.get("uncached_input_tokens_per_s"),
            }
        )
    # Decode tok/s: unweighted mean of the scheduler's sampled `gen throughput` values.
    print(
        "| Arm | Requests | Errors | Req/s | Hit % | Device / host / new % | Mean sampled decode tok/s | Main p50 / p90 / p99 s | Eviction s (TP0) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["main"]
        ev = f"{r['evict']:.0f}" if r["evict"] is not None else "-"
        print(
            f"| {r['arm']} | {r['requests']} | {r['errors']} | {r['req_s']:.3f} | {r['hit']:.1f} | "
            + " / ".join(f"{x:.1f}" for x in r["split"])
            + " | "
            + ("-" if r["prefill_only"] else f"{r['decode']:.0f}")
            + f" | {m['p50']} / {m['p90']} / {m['p99']} | {ev} |"
        )
    print()
    prefill = [r for r in rows if r["prefill_only"]]
    if prefill:
        # With --prefill-only every request returns one token, so its latency is the TTFT.
        print(
            "| Arm | Input tok/s | Uncached input tok/s | TTFT main p50 / p90 / p99 s | TTFT reminder p50 / p90 / p99 s |"
        )
        print("|---|---|---|---|---|")
        for r in prefill:
            m, rem = r["latency"]["main"], r["latency"]["reminder"]
            print(
                f"| {r['arm']} | {r['input_tps']:.0f} | {r['uncached_tps']:.0f} | "
                f"{m['p50']} / {m['p90']} / {m['p99']} | {rem['p50']} / {rem['p90']} / {rem['p99']} |"
            )
        print()
    n = max(len(r["per10"]) for r in rows)
    print("| Minutes | " + " | ".join(r["arm"] for r in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for i in range(n):
        cells = [f"{r['per10'][i]:.1f}" if i < len(r["per10"]) else "-" for r in rows]
        print(f"| {i * 10}-{i * 10 + 10} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
