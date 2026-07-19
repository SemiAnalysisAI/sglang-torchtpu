#!/usr/bin/env python3
"""AC-9 realistic-workload throughput probe for DFlash-on-TPU.

The synthetic filler-token bench (dflash_ac9_bench.py) is a BEST-CASE for spec
decoding: a repetitive input makes the draft accept ~all draft_token_num tokens
per step. This probe instead sends a natural-language prompt so the DFlash draft
accept length is realistic (typically 2-4 for open-ended generation), giving an
honest steady-state tokens/s. No-spec decode tokens/s is essentially
content-independent, so the no-spec baseline (~9.3 tok/s bs=1) is a valid
reference for THIS DFlash number too.

Usage: dflash_ac9_bench_realistic.py <port> <label>
"""
import json
import sys
import time
import urllib.request

PROMPTS = [
    "Explain, in detail and step by step, how a modern CPU pipeline executes "
    "instructions, covering fetch, decode, execute, memory, and writeback:",
    "Write a thorough essay about the causes and consequences of the "
    "industrial revolution in nineteenth century Europe:",
]
MAX_NEW = 160


def gen(port, prompt, max_new):
    body = json.dumps(
        {
            "text": prompt,
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
    return mi.get("completion_tokens", 0), dt, (
        mi.get("spec_accept_length") or mi.get("accept_length")
    )


def main():
    port = int(sys.argv[1])
    label = sys.argv[2]
    print(f"### AC-9 realistic bench: label={label} port={port} max_new={MAX_NEW}")
    print("| prompt | tok/s | accept_len | completion | s |")
    print("|--------|-------|------------|------------|---|")
    # Warm the decode shape once (discarded).
    gen(port, PROMPTS[0], 24)
    for i, p in enumerate(PROMPTS):
        toks, dt, acc = gen(port, p, MAX_NEW)
        rate = toks / dt if dt > 0 else 0.0
        print(f"| p{i} | {rate:.1f} | "
              f"{('%.2f' % acc) if acc else 'n/a'} | {toks} | {dt:.2f} |")


if __name__ == "__main__":
    main()
