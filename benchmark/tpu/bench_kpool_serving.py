"""Record and compare fixed greedy requests for a kpool serving change.

Run each revision on the same allocation with the same recipe. This client
records request latency, not kernel time, and does not launch a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    context = tokenizer.encode(
        "A paged cache stores records in physical pages. A request uses a page "
        "table to find its records. Different requests can use different pages. "
        "A partial update must preserve records that it does not replace.\n",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        "\nExplain why a partial cache update must preserve other records.\n",
        add_special_tokens=False,
    )
    cases = []
    for length in (127, 128, 129, 2047, 2048, 2049, 4095, 4096, 8192):
        prefix_len = length - len(question)
        if prefix_len <= 0 or not context:
            raise ValueError("Tokenizer does not fit the requested input lengths")
        ids = (context * ((prefix_len + len(context) - 1) // len(context)))[
            :prefix_len
        ] + question
        cases.append({"name": f"tokens-{length}", "input_ids": ids})
    args.output.write_text(json.dumps({"model": args.model, "cases": cases}))


def generate(base_url, case, output_tokens):
    payload = {
        "input_ids": case["input_ids"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    last_token_time = None
    previous_count = 0
    final = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            ids = event.get("output_ids", [])
            if len(ids) < previous_count:
                raise ValueError("Expected cumulative output_ids in the stream")
            if len(ids) > previous_count:
                now = time.perf_counter()
                first = now if first is None else first
                last_token_time = now
                previous_count = len(ids)
            final = event
    finished = time.perf_counter()
    if final is None or first is None or last_token_time is None:
        raise ValueError("Generation returned no tokens")
    ids = final.get("output_ids", [])
    meta = final.get("meta_info", {})
    if len(ids) != output_tokens or meta.get("completion_tokens") != output_tokens:
        raise ValueError(f"Incomplete generation: {len(ids)} tokens, {meta}")
    if meta.get("prompt_tokens") != len(case["input_ids"]):
        raise ValueError("Server prompt length differs from the frozen input")
    logprobs = meta.get("output_token_logprobs", [])
    if len(logprobs) != output_tokens or not all(
        item[0] is not None and math.isfinite(item[0]) for item in logprobs
    ):
        raise ValueError("Missing or nonfinite output log probabilities")
    return {
        "name": case["name"],
        "input_sha256": digest(case["input_ids"]),
        "prompt_tokens": len(case["input_ids"]),
        "output_ids": ids,
        "output_logprobs": [item[0] for item in logprobs],
        "ttft_ms": (first - started) * 1000,
        "latency_ms": (finished - started) * 1000,
        "tpot_ms": (last_token_time - first) * 1000 / max(1, len(ids) - 1),
        "server_meta": meta,
    }


def run(args):
    frozen = json.loads(args.cases.read_text())
    result = {
        "label": args.label,
        "source_revision": args.source_revision,
        "cases_sha256": digest(frozen),
        "output_tokens": args.output_tokens,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "records": [],
        "completed": False,
    }
    args.output.write_text(json.dumps(result, indent=2))
    for case in frozen["cases"]:
        for _ in range(args.warmups):
            generate(args.base_url, case, args.output_tokens)
        for repeat in range(args.repeats):
            record = generate(args.base_url, case, args.output_tokens)
            record["repeat"] = repeat
            result["records"].append(record)
            args.output.write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in ("name", "repeat", "ttft_ms", "latency_ms", "tpot_ms")
                    }
                ),
                flush=True,
            )
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2))


def compare(args):
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    for field in ("cases_sha256", "output_tokens", "warmups", "repeats"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Comparison identity mismatch: {field}")
    if not baseline["completed"] or not candidate["completed"]:
        raise ValueError("Cannot compare incomplete runs")
    left = {(r["name"], r["repeat"]): r for r in baseline["records"]}
    right = {(r["name"], r["repeat"]): r for r in candidate["records"]}
    if not left or left.keys() != right.keys():
        raise ValueError("Measured requests differ")
    mismatches = []
    for key, before in left.items():
        after = right[key]
        if before["input_sha256"] != after["input_sha256"]:
            raise ValueError(f"Input identity mismatch: {key}")
        if before["output_ids"] != after["output_ids"]:
            mismatches.append(list(key))
    rows = []
    for name in sorted({key[0] for key in left}):
        row = {"name": name}
        for metric in ("ttft_ms", "latency_ms", "tpot_ms"):
            row[metric] = {
                label: statistics.median(
                    r[metric] for (n, _), r in records.items() if n == name
                )
                for label, records in (("baseline", left), ("candidate", right))
            }
        rows.append(row)
    result = {
        "exact_generation_match": not mismatches,
        "mismatches": mismatches,
        "medians": rows,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.set_defaults(function=prepare)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cases", type=Path, required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--source-revision", required=True)
    run_parser.add_argument("--output-tokens", type=int, default=64)
    run_parser.add_argument("--warmups", type=int, default=1)
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.set_defaults(function=run)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.set_defaults(function=compare)
    for child in (prepare_parser, run_parser, compare_parser):
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run" and (
        args.output_tokens < 1 or args.warmups < 1 or args.repeats < 1
    ):
        parser.error("output-tokens, warmups and repeats must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
