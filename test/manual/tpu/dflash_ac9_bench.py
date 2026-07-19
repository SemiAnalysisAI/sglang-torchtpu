#!/usr/bin/env python3
"""AC-9 throughput benchmark for DFlash-on-TPU vs a no-spec target baseline.

Measures warmed steady-state OUTPUT (decode) throughput at a real batch size N by
sending N concurrent equal-length `input_ids` requests (`ignore_eos`, temperature 0
so the run is deterministic and greedy). Each batch size is warmed once (discarded)
before the measured pass, because the first touch of a bs/context bucket compiles
(the warmup tail otherwise drags a short-run average -- see BitLesson
BL-20260718-tpu-dflash-draft-eager-perf).

Reports, per batch size:
  * aggregate output tokens/s = sum(completion_tokens) / wall_seconds
  * per-request output tokens/s = aggregate / N
  * mean DFlash accept length (spec_verify path) when meta_info exposes it

Usage: dflash_ac9_bench.py <port> <label> [bs1,bs2,...]
       port  -- 30030 (DFlash) or 30031 (no-spec)
       label -- free text tag printed in the header (e.g. "dflash", "nospec")
"""
import concurrent.futures
import json
import sys
import time
import urllib.request

PROMPT_LEN = 64
MAX_NEW = 128
FILLER = 1000


def send_one(port, i, max_new):
    ids = [2000 + i] + [FILLER] * (PROMPT_LEN - 1)
    body = json.dumps(
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new,
                "ignore_eos": True,
            },
        }
    ).encode()
    t0 = time.time()
    r = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            ),
            timeout=3600,
        ).read()
    )
    dt = time.time() - t0
    mi = r.get("meta_info", {})
    return {
        "completion_tokens": mi.get("completion_tokens", 0),
        "elapsed": dt,
        "accept_length": mi.get("spec_accept_length") or mi.get("accept_length"),
    }


def run_batch(port, n, max_new):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda i: send_one(port, i, max_new), range(n)))
    wall = time.time() - t0
    return results, wall


def main():
    port = int(sys.argv[1])
    label = sys.argv[2]
    batch_sizes = (
        [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 8]
    )
    print(f"### AC-9 bench: label={label} port={port} "
          f"prompt_len={PROMPT_LEN} max_new={MAX_NEW}")
    print("| bs | warmup? | agg_tok/s | per_req_tok/s | mean_accept_len | "
          "sum_completion | wall_s |")
    print("|----|---------|-----------|---------------|-----------------|"
          "----------------|--------|")
    for n in batch_sizes:
        # Warm the bs/context bucket (discarded).
        run_batch(port, n, max_new=32)
        # Measured pass.
        results, wall = run_batch(port, n, MAX_NEW)
        total = sum(r["completion_tokens"] for r in results)
        accepts = [r["accept_length"] for r in results if r["accept_length"]]
        mean_accept = sum(accepts) / len(accepts) if accepts else None
        agg = total / wall if wall > 0 else 0.0
        print(f"| {n} | measured | {agg:.1f} | {agg / n:.1f} | "
              f"{('%.2f' % mean_accept) if mean_accept else 'n/a'} | "
              f"{total} | {wall:.2f} |")


if __name__ == "__main__":
    main()
