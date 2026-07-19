#!/usr/bin/env python3
"""Rigorous, self-validating per-pow2-bucket recompilation probe for the DFlash
extend materialization (AC-7).

For each REQUIRED pow2 token bucket in the configured ctx-2048 envelope
({64,128,256,512,1024,2048}) it sends three validated requests:
  - cold           : first prompt of a distinct length in the bucket
  - same_bucket    : a SECOND distinct length in the SAME bucket
  - repeat         : identical input_ids to `cold` (a true uncached repeat)

Exact lengths are controlled by sending `input_ids` (not approximate text), so
the observed prefill ``#new-token`` lands deterministically in the intended
bucket. The server MUST run ``--disable-radix-cache`` so every prefill is
uncached; each row asserts observed ``#cached-token == 0``.

The probe FAILS (exit 1) if any row: has no parsed prefill line, is radix-cached
(``#cached-token != 0``), lands in a bucket other than expected, or if any
required (bucket x row_type) is missing. It prints the per-row deltas so a
reviewer can confirm the DFlash extend materialize (``pf_extend_materialize``)
first-touches at most once per bucket and same_bucket/repeat rows are 0 (or an
attributed residual).

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
REQUIRED_BUCKETS = [64, 128, 256, 512, 1024, 2048]
# Distinct filler / marker token ids well inside the Llama-3.1 vocab, avoiding
# special tokens (>=128000) and 0.
_FILLER = 1000


def _pow2(n, floor=64):
    f = 1 << (floor - 1).bit_length()
    return f if n <= f else 1 << (n - 1).bit_length()


def _lengths_for(bucket):
    # Two distinct lengths comfortably inside (bucket/2, bucket], away from the
    # boundary so a possible +1 BOS can't spill into the next bucket.
    return bucket * 3 // 4, bucket * 5 // 8


_attrib = re.compile(r"xla-attrib: phase=([a-z_]+) compiled (\d+)")
_checker = re.compile(r"recompilation checker\] (\d+) new XLA program")
_prefill = re.compile(r"#new-token: (\d+), #cached-token: (\d+)")


def parse(lines):
    phase = {p: 0 for p in PHASES}
    checker = 0
    newtok = cached = None
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


def send(input_ids):
    d = json.dumps(
        {"input_ids": input_ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": 4, "ignore_eos": True}}
    ).encode()
    r = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(URL, data=d, headers={"Content-Type": "application/json"}),
            timeout=1800,
        ).read()
    )
    return r["meta_info"].get("completion_tokens")


def build_requests():
    reqs = []  # (label, expected_bucket, row_type, input_ids)
    marker = 2000
    for b in REQUIRED_BUCKETS:
        la, lb = _lengths_for(b)
        cold_ids = [marker] + [_FILLER] * (la - 1)
        marker += 1
        same_ids = [marker] + [_FILLER] * (lb - 1)
        marker += 1
        reqs.append((f"b{b}_cold", b, "cold", cold_ids))
        reqs.append((f"b{b}_same", b, "same_bucket", same_ids))
        reqs.append((f"b{b}_repeat", b, "repeat", list(cold_ids)))  # identical to cold
    return reqs


def main():
    log = sys.argv[1]
    with open(log) as f:
        pos = len(f.readlines())
    hdr = ["request", "input_len", "exp_bkt", "row_type", "newtok", "cached", "obs_bkt",
           "pf_tgt_ext", "pf_mat", "vglue", "kv_app", "ckr", "ct", "VALID?"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    seen = set()
    failures = []
    for label, exp_bkt, row_type, ids in build_requests():
        input_len = len(ids)
        try:
            ct = send(ids)
        except Exception as e:  # bucket may exceed context for the top bucket
            failures.append(f"{label}: request error {e!r}")
            print(f"| {label} | {input_len} | {exp_bkt} | {row_type} | ? | ? | ? | - | - | - | - | - | - | FAIL(req) |")
            continue
        time.sleep(2)
        with open(log) as f:
            lines = f.readlines()
        window, pos = lines[pos:], len(lines)
        phase, checker, newtok, cached = parse(window)
        row_ok = True
        reason = "OK"
        if newtok is None or cached is None:
            row_ok, reason = False, "no-prefill-parse"
        else:
            obs = _pow2(newtok)
            if cached != 0:
                row_ok, reason = False, f"cached={cached}"
            elif obs != exp_bkt:
                row_ok, reason = False, f"obs_bkt={obs}!=exp"
        obs_bkt = _pow2(newtok) if newtok is not None else "?"
        if not row_ok:
            failures.append(f"{label}: {reason}")
        else:
            seen.add((exp_bkt, row_type))
        row = [label, str(input_len), str(exp_bkt), row_type, str(newtok), str(cached), str(obs_bkt),
               str(phase["prefill_target_extend_fwd"]), str(phase["prefill_extend_materialize"]),
               str(phase["verify_glue"]), str(phase["kv_append_post"]),
               str(checker), str(ct), reason]
        print("| " + " | ".join(row) + " |")
    # Coverage: every required (bucket x row_type) present & valid.
    missing = [
        (b, rt) for b in REQUIRED_BUCKETS for rt in ("cold", "same_bucket", "repeat")
        if (b, rt) not in seen
    ]
    print()
    if missing:
        failures.append("missing coverage: " + ", ".join(f"{b}/{rt}" for b, rt in missing))
    if failures:
        print("HARNESS_RESULT: FAIL")
        for f_ in failures:
            print("  - " + f_)
        sys.exit(1)
    print("HARNESS_RESULT: PASS (all required buckets x {cold,same_bucket,repeat} uncached, in-bucket)")


if __name__ == "__main__":
    main()
