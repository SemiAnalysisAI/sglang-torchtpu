#!/usr/bin/env python3
"""bs>1 recompilation-checker probe for DFlash-on-TPU (compile stability at bs>1).

Forces a steady real batch of N requests (concurrent, equal-length `input_ids`,
`ignore_eos`, same `max_new_tokens` -> they prefill and decode together at bs=N)
and measures, per run, the decode-phase XLA-cache deltas (via the worker's
per-phase probes) and the global recompilation-checker delta.

For each N it does a COLD run (first time this real bs is seen) then a WARM REPEAT
(identical batch). The compile-bucketed phases -- prep_total (draft forward),
target_verify_fwd (target-verify forward), and kv_append_post (the decode
materialize grid) -- are 0 on the warm repeat because they pad to the compile
bucket; a nonzero warm-repeat delta in one of THOSE would be a per-REAL-bs
recompile (a `batch.batch_size()`-keyed eager op). verify_glue is expected to be
NONZERO here and is NOT a per-real-bs recompile: it is the eager free-cache-loc
compaction keyed on the data-dependent `free_locs` count (bs-INDEPENDENT -- it
recompiles the same way at real bs=1), a bounded, plan-accepted eager budget.

Server: SGLANG_TPU_CHECK_RECOMPILATION=1 SGLANG_DFLASH_VERIFY_SHAPE_TRACE=1
        ... --tpu-compile-bs 1 8 --disable-radix-cache --max-running-requests >=N

Usage: dflash_bs_gt1_checker_probe.py <server_log_path>
"""
import concurrent.futures
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:30030/generate"
DECODE_PHASES = ["prep_total", "target_verify_fwd", "verify_glue", "kv_append_post"]
BATCH_SIZES = [4, 6]  # real bs values in (1,8) -> pad to the bs=8 bucket
FILLER = 1000
PROMPT_LEN = 48  # -> pow2 bucket 64 for the prefill; same for all requests
MAX_NEW = 48

_attrib = re.compile(r"xla-attrib: phase=([a-z_]+) compiled (\d+)")
_checker = re.compile(r"recompilation checker\] (\d+) new XLA program")
_running = re.compile(r"#running-req: (\d+)")


def parse(lines):
    phase = {p: 0 for p in DECODE_PHASES}
    checker = 0
    max_run = 0
    for ln in lines:
        m = _running.search(ln)
        if m:
            max_run = max(max_run, int(m.group(1)))
        if "TP0" not in ln:
            continue
        m = _attrib.search(ln)
        if m and m.group(1) in phase:
            phase[m.group(1)] += int(m.group(2))
        m = _checker.search(ln)
        if m:
            checker += int(m.group(1))
    return phase, checker, max_run


def send_one(i):
    # Distinct leading token so nothing dedups, same length for all -> same bucket.
    ids = [2000 + i] + [FILLER] * (PROMPT_LEN - 1)
    d = json.dumps(
        {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_NEW, "ignore_eos": True}}
    ).encode()
    r = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(URL, data=d, headers={"Content-Type": "application/json"}),
            timeout=1800,
        ).read()
    )
    return r["meta_info"].get("completion_tokens")


def run_batch(n):
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(send_one, range(n)))


def main():
    log = sys.argv[1]
    with open(log) as f:
        pos = len(f.readlines())
    hdr = ["bs", "run", "max_running", "prep_total", "target_verify", "verify_glue",
           "kv_app", "checker_delta"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for n in BATCH_SIZES:
        for run in ("cold", "warm_repeat"):
            run_batch(n)
            time.sleep(3)
            with open(log) as f:
                lines = f.readlines()
            window, pos = lines[pos:], len(lines)
            phase, checker, max_run = parse(window)
            row = [str(n), run, str(max_run),
                   str(phase["prep_total"]), str(phase["target_verify_fwd"]),
                   str(phase["verify_glue"]), str(phase["kv_append_post"]),
                   str(checker)]
            print("| " + " | ".join(row) + " |")
    print()
    print("Reading: a WARM_REPEAT row with kv_app=target_verify=prep_total=0 means the")
    print("compile-bucketed path is checker-clean for that real bs after warmup. A nonzero")
    print("kv_app/target_verify/prep_total warm-repeat delta = a per-real-bs recompile")
    print("(batch.batch_size()-keyed). verify_glue is EXPECTED nonzero and bs-independent")
    print("(eager free_locs-keyed compaction, a bounded plan-accepted budget), NOT a regression.")


if __name__ == "__main__":
    main()
