#!/usr/bin/env python3
"""Concurrent per-request greedy-parity harness for DFlash-on-TPU.

Complements the single-stream (bs=1) parity proof: sends N distinct prompts
CONCURRENTLY to one server (forming a real multi-request / bs~N batch), collects
each request's output token stream tagged by prompt, and compares two servers'
dumps per-request. A divergence is exact-fail unless a supplied bf16 tie-break
tolerance manifest (bound to the current dumps) tolerates it.

The comparison core (`compare_concurrent_streams`) is pure + injectable so it is
hermetically CPU-testable without a live server.

Usage:
  dflash_concurrent_parity.py collect <base_url> <out.json> [--max-new-tokens N]
  dflash_concurrent_parity.py compare <dflash.json> <nospec.json> \
      [--tolerance-manifest <m.json>]
"""
import argparse
import concurrent.futures
import importlib.util
import json
import pathlib
import sys
import urllib.request
from typing import Dict, List, Optional

# Distinct natural-language prompts so 16 concurrent requests form a real batch.
DEFAULT_PROMPTS: List[str] = [
    "The capital of France is",
    "List the first five prime numbers:",
    "Q: What is 17 + 25? A:",
    "Once upon a time,",
    "Explain speculative decoding in one sentence:",
    "Water boils at a temperature of",
    "The opposite of hot is",
    "Name three primary colors:",
    "The first president of the United States was",
    "Photosynthesis converts sunlight into",
    "A group of lions is called a",
    "The chemical symbol for gold is",
    "Two plus two equals",
    "The largest planet in our solar system is",
    "Roses are red, violets are",
    "The speed of light is approximately",
]


def _sibling():
    """Load dflash_greedy_parity for its diff / tolerance predicates."""
    p = pathlib.Path(__file__).parent / "dflash_greedy_parity.py"
    spec = importlib.util.spec_from_file_location("dflash_greedy_parity", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(base_url: str, prompt: str, max_new: int):
    """Return ``(output_ids, meta)`` where meta = {completion_tokens, accept_length}."""
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
            "stream": False,
        }
    ).encode()
    r = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                base_url.rstrip("/") + "/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            ),
            timeout=1800,
        ).read()
    )
    mi = r.get("meta_info", {})
    if r.get("output_ids") is not None:
        ids = [int(t) for t in r["output_ids"]]
    else:
        otl = mi.get("output_token_logprobs")
        ids = [int(e[1]) for e in otl] if otl else []
    meta = {
        "completion_tokens": mi.get("completion_tokens"),
        "accept_length": mi.get("spec_accept_length") or mi.get("accept_length"),
    }
    return ids, meta


def collect_concurrent(base_url: str, prompts: List[str], max_new: int):
    """Send all prompts CONCURRENTLY (one request each) so they batch together.

    Returns ``(streams, meta)``: ``streams`` = {prompt: output_ids} (the dump
    `compare` reads), ``meta`` = {prompt: {completion_tokens, accept_length}}."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        results = list(ex.map(lambda p: _generate(base_url, p, max_new), prompts))
    streams = {p: r[0] for p, r in zip(prompts, results)}
    meta = {p: r[1] for p, r in zip(prompts, results)}
    return streams, meta


def summarize_accept_meta(meta: Dict[str, Dict]) -> Dict:
    """Pure summary of per-request metadata: mean accept length over the requests
    that reported one, and the count. CPU-testable (no I/O)."""
    accepts = [
        float(d["accept_length"])
        for d in meta.values()
        if d.get("accept_length") is not None
    ]
    return {
        "n": len(meta),
        "n_with_accept": len(accepts),
        "mean_accept_length": (sum(accepts) / len(accepts)) if accepts else None,
    }


def observed_max_running(server_log_path: str) -> int:
    """Max #running-req observed in a server log (the collect's concurrent batch
    size), via the committed compile-checker parser."""
    p = pathlib.Path(__file__).parent / "dflash_parse_checker_log.py"
    spec = importlib.util.spec_from_file_location("dflash_parse_checker_log", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(server_log_path) as f:
        return mod.parse_checker_log(f.readlines())["max_running"]


def compare_concurrent_streams(
    dflash: Dict[str, List[int]],
    nospec: Dict[str, List[int]],
    manifest: Optional[Dict] = None,
    *,
    diff_fn=None,
    tolerated_fn=None,
) -> Dict:
    """Pure per-request comparison. Each prompt is exact / tolerated / fail /
    missing. Returns counts + per-prompt detail + ``all_ok`` (no untolerated
    divergence AND no missing request). ``diff_fn``/``tolerated_fn`` are injectable
    for hermetic tests; they default to the ratified `dflash_greedy_parity`
    predicates.

    Iterates the UNION of prompt keys so a request missing from EITHER dump fails
    closed (a dropped DFlash request must never be silently omitted and still
    report ``all_ok``)."""
    if diff_fn is None or tolerated_fn is None:
        m = _sibling()
        diff_fn = diff_fn or m.diff_token_streams
        tolerated_fn = tolerated_fn or m._tolerated_divergence
    counts = {"exact": 0, "tolerated": 0, "fail": 0, "missing": 0}
    per_prompt: Dict[str, Dict] = {}
    for prompt in sorted(set(dflash) | set(nospec)):
        in_dflash = prompt in dflash
        in_nospec = prompt in nospec
        if not in_dflash or not in_nospec:
            counts["missing"] += 1
            per_prompt[prompt] = {
                "status": "missing",
                "missing_from": "dflash" if not in_dflash else "nospec",
            }
            continue
        spec, ref = dflash[prompt], nospec[prompt]
        v = diff_fn(spec, ref)
        if v["match"]:
            counts["exact"] += 1
            per_prompt[prompt] = {"status": "exact", "len": v["spec_len"]}
            continue
        idx = int(v["first_divergence"])
        tolerated = False
        if manifest and idx < len(spec) and idx < len(ref):
            tolerated = bool(
                tolerated_fn(manifest, prompt, idx, int(spec[idx]), int(ref[idx]))[0]
            )
        if tolerated:
            counts["tolerated"] += 1
            per_prompt[prompt] = {"status": "tolerated", "index": idx}
        else:
            counts["fail"] += 1
            per_prompt[prompt] = {"status": "fail", "index": idx}
    counts["all_ok"] = counts["fail"] == 0 and counts["missing"] == 0
    return {"counts": counts, "per_prompt": per_prompt}


def _cmd_collect(args) -> int:
    streams, meta = collect_concurrent(
        args.base_url, DEFAULT_PROMPTS, args.max_new_tokens
    )
    with open(args.out_json, "w") as f:
        json.dump(streams, f)  # streams-only dump: `compare` reads this
    print(f"collected {len(streams)} concurrent streams -> {args.out_json}")
    if args.meta_out:
        with open(args.meta_out, "w") as f:
            json.dump(meta, f)
        s = summarize_accept_meta(meta)
        print(
            f"meta -> {args.meta_out} | mean accept_length="
            f"{s['mean_accept_length']} over {s['n_with_accept']}/{s['n']} requests"
        )
    if args.server_log:
        print(f"observed max #running-req (batch): {observed_max_running(args.server_log)}")
    return 0


def _cmd_compare(args) -> int:
    with open(args.dflash_json) as f:
        dflash = json.load(f)
    with open(args.nospec_json) as f:
        nospec = json.load(f)
    manifest = None
    if args.tolerance_manifest:
        with open(args.tolerance_manifest) as f:
            manifest = json.load(f)
    summary = compare_concurrent_streams(dflash, nospec, manifest)
    c = summary["counts"]
    for prompt, d in summary["per_prompt"].items():
        print(f"{d['status'].upper():10s} {prompt!r}")
    print("=" * 60)
    print(
        f"CONCURRENT PER-REQUEST PARITY: exact={c['exact']} "
        f"tolerated={c['tolerated']} fail={c['fail']} missing={c['missing']} "
        f"-> {'PASS' if c['all_ok'] else 'FAIL'}"
    )
    return 0 if c["all_ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    c = sub.add_parser("collect")
    c.add_argument("base_url")
    c.add_argument("out_json")
    c.add_argument("--max-new-tokens", type=int, default=96)
    c.add_argument(
        "--meta-out", help="also write {prompt: {completion_tokens, accept_length}}"
    )
    c.add_argument(
        "--server-log",
        help="server log to report observed max #running-req (concurrent batch)",
    )
    c.set_defaults(func=_cmd_collect)
    cmp = sub.add_parser("compare")
    cmp.add_argument("dflash_json")
    cmp.add_argument("nospec_json")
    cmp.add_argument("--tolerance-manifest")
    cmp.set_defaults(func=_cmd_compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
