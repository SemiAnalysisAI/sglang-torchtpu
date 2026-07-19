#!/usr/bin/env python3
"""Self-validating per-pow2-bucket recompilation probe for the DFlash extend
materialization (AC-7).

Sends DISTINCT-content prompts of varying length that span the configured pow2
token buckets. For each request it parses the server log window and emits a row
with the OBSERVED prefill ``#new-token`` and ``#cached-token`` (not a guess), the
pow2 bucket, and the per-phase XLA-cache deltas. Rows are grouped by observed
bucket so a reviewer can read, per bucket: a cold first-touch, then further
distinct prompts in the SAME bucket whose ``pf_extend_materialize`` should be a
small bounded residual (ideally 0), never a fresh per-length compile.

Self-validation: launch the server with ``--disable-radix-cache`` so every
prefill is uncached; the probe FAILS any row whose observed ``#cached-token != 0``
(radix reuse would change the extend length and invalidate the bucket probe) —
that was the R11 evidence defect. Distinct prompt content also avoids dedup.

Usage: dflash_recompile_bucket_probe.py <server_log_path>
  (server: SGLANG_TPU_CHECK_RECOMPILATION=1 SGLANG_DFLASH_VERIFY_SHAPE_TRACE=1
           ... --disable-radix-cache)
"""
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:30030/generate"
PHASES = [
    "prefill_target_extend_fwd",
    "prefill_extend_materialize",
    "verify_glue",
    "kv_append_post",
]


def _pow2(n, floor=64):
    f = 1 << (floor - 1).bit_length()
    return f if n <= f else 1 << (n - 1).bit_length()


# Distinct-content prompts (unique leading token per prompt -> no dedup/radix
# reuse) whose lengths intentionally span several pow2 buckets, with >=2 distinct
# lengths per bucket so a bucket's 2nd+ prompt exercises the already-warmed shape.
_UNIQUE = ["zeta", "eta", "theta", "iota", "kappa", "lam", "mu", "nu", "xi", "pi", "rho", "tau"]
_LENGTHS = [30, 50, 80, 110, 200, 230, 400, 450]  # -> buckets 64,64,128,128,256,256,512,512


def _build_requests():
    reqs = []
    for i, ln in enumerate(_LENGTHS):
        # Unique first word guarantees distinct content (no radix reuse).
        text = _UNIQUE[i % len(_UNIQUE)] + " " + ("word%d " % i) * ln
        reqs.append((f"len{ln}_uniq{i}", text))
    return reqs


_attrib = re.compile(r"xla-attrib: phase=([a-z_]+) compiled (\d+)")
_checker = re.compile(r"recompilation checker\] (\d+) new XLA program")
_prefill = re.compile(r"#new-token: (\d+), #cached-token: (\d+)")


def parse(lines):
    phase = {p: 0 for p in PHASES}
    checker = newtok = cached = 0
    for ln in lines:
        m = _prefill.search(ln)
        if m:
            newtok, cached = int(m.group(1)), int(m.group(2))
        if "TP0" not in ln:
            continue
        m = _attrib.search(ln)
        if m and m.group(1) in phase:
            phase[m.group(1)] += int(m.group(2))
        m = _checker.search(ln)
        if m:
            checker += int(m.group(1))
    return phase, checker, newtok, cached


def send(text):
    d = json.dumps(
        {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": 40, "ignore_eos": True}}
    ).encode()
    r = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(URL, data=d, headers={"Content-Type": "application/json"}),
            timeout=1800,
        ).read()
    )
    return r["meta_info"].get("completion_tokens")


def main():
    log = sys.argv[1]
    with open(log) as f:
        pos = len(f.readlines())
    hdr = ["request", "newtok", "cached", "bucket", "pf_tgt_ext", "pf_mat", "vglue", "kv_app", "ckr", "ct", "VALID?"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    seen_bucket = {}
    any_fail = False
    for label, text in _build_requests():
        ct = send(text)
        time.sleep(2)
        with open(log) as f:
            lines = f.readlines()
        window, pos = lines[pos:], len(lines)
        phase, checker, newtok, cached = parse(window)
        bucket = _pow2(newtok)
        # Self-validation: uncached probe required; flag radix reuse.
        ok = cached == 0
        if not ok:
            any_fail = True
        first_in_bucket = bucket not in seen_bucket
        seen_bucket.setdefault(bucket, label)
        note = "OK" if ok else "FAIL(cached!=0)"
        note += " cold" if first_in_bucket else " same-bkt"
        row = [
            label, str(newtok), str(cached), str(bucket),
            str(phase["prefill_target_extend_fwd"]), str(phase["prefill_extend_materialize"]),
            str(phase["verify_glue"]), str(phase["kv_append_post"]),
            str(checker), str(ct), note,
        ]
        print("| " + " | ".join(row) + " |")
    print()
    print(f"HARNESS_RESULT: {'FAIL (a probe was radix-cached)' if any_fail else 'PASS (all probes uncached)'}")


if __name__ == "__main__":
    main()
