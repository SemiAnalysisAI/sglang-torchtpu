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


def _generate(base_url: str, prompt: str, max_new: int) -> List[int]:
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
    if r.get("output_ids") is not None:
        return [int(t) for t in r["output_ids"]]
    otl = r.get("meta_info", {}).get("output_token_logprobs")
    return [int(e[1]) for e in otl] if otl else []


def collect_concurrent(
    base_url: str, prompts: List[str], max_new: int
) -> Dict[str, List[int]]:
    """Send all prompts CONCURRENTLY (one request each) so they batch together."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        streams = list(
            ex.map(lambda p: _generate(base_url, p, max_new), prompts)
        )
    return {p: s for p, s in zip(prompts, streams)}


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
    divergence). ``diff_fn``/``tolerated_fn`` are injectable for hermetic tests;
    they default to the ratified `dflash_greedy_parity` predicates."""
    if diff_fn is None or tolerated_fn is None:
        m = _sibling()
        diff_fn = diff_fn or m.diff_token_streams
        tolerated_fn = tolerated_fn or m._tolerated_divergence
    counts = {"exact": 0, "tolerated": 0, "fail": 0, "missing": 0}
    per_prompt: Dict[str, Dict] = {}
    for prompt, spec in dflash.items():
        if prompt not in nospec:
            counts["missing"] += 1
            per_prompt[prompt] = {"status": "missing"}
            continue
        ref = nospec[prompt]
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
    result = collect_concurrent(args.base_url, DEFAULT_PROMPTS, args.max_new_tokens)
    with open(args.out_json, "w") as f:
        json.dump(result, f)
    print(f"collected {len(result)} concurrent streams -> {args.out_json}")
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
