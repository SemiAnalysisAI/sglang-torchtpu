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
import hashlib
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


def observed_max_running(server_log_path: str, from_line: int = 0) -> int:
    """Max #running-req observed in a server log, over lines ``[from_line:]`` only,
    via the committed compile-checker parser.

    ``from_line`` windows the parse to the concurrent-collect window (the log line
    count captured BEFORE the collect): whole-log parsing (``from_line=0``) can
    over-report a batch size reached by an EARLIER warmup / probe / failed-launch
    window rather than the actual concurrent collection."""
    p = pathlib.Path(__file__).parent / "dflash_parse_checker_log.py"
    spec = importlib.util.spec_from_file_location("dflash_parse_checker_log", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(server_log_path) as f:
        lines = f.readlines()[from_line:]
    return mod.parse_checker_log(lines)["max_running"]


def compare_concurrent_streams(
    dflash: Dict[str, List[int]],
    nospec: Dict[str, List[int]],
    manifest: Optional[Dict] = None,
    *,
    diff_fn=None,
    tolerated_fn=None,
) -> Dict:
    """Pure per-request comparison. Each prompt is exact / tolerated / fail /
    missing / empty. Returns counts + per-prompt detail + ``all_ok`` (no untolerated
    divergence AND no missing request AND no empty stream). ``diff_fn``/``tolerated_fn``
    are injectable for hermetic tests; they default to the ratified
    `dflash_greedy_parity` predicates.

    Iterates the UNION of prompt keys so a request missing from EITHER dump fails
    closed (a dropped DFlash request must never be silently omitted and still
    report ``all_ok``).

    An EMPTY token stream (``[]``) on either side fails closed as ``empty``: two
    empty streams would otherwise diff as ``match`` (an empty prefix of an empty
    stream) and count ``exact``, so a request that errored-to-empty / was rejected /
    truncated to zero tokens on both servers could report a FALSE parity PASS. An
    empty stream is absence of evidence, not proof of parity."""
    if diff_fn is None or tolerated_fn is None:
        m = _sibling()
        diff_fn = diff_fn or m.diff_token_streams
        tolerated_fn = tolerated_fn or m._tolerated_divergence
    counts = {"exact": 0, "tolerated": 0, "fail": 0, "missing": 0, "empty": 0}
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
        if not spec or not ref:
            counts["empty"] += 1
            empty_on = (
                "both" if not spec and not ref else ("dflash" if not spec else "nospec")
            )
            per_prompt[prompt] = {"status": "empty", "empty_on": empty_on}
            continue
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
    counts["all_ok"] = (
        counts["fail"] == 0 and counts["missing"] == 0 and counts["empty"] == 0
    )
    return {"counts": counts, "per_prompt": per_prompt}


def collection_config(prompts: List[str], max_new_tokens: int) -> Dict:
    """Deterministic fingerprint of a collect invocation's config. The two
    sequential collects (dflash server, then no-spec server) MUST share this so
    their dumps are comparable; a mismatch (different max_new_tokens, or a
    harness-version / prompt-set skew) turns into length/content divergences whose
    cause is otherwise invisible. Pure + CPU-testable."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")  # length-independent separator
    return {
        "max_new_tokens": int(max_new_tokens),
        "num_prompts": len(prompts),
        "prompt_sha": h.hexdigest(),
    }


def config_mismatch(cfg_a: Dict, cfg_b: Dict) -> List[str]:
    """Field names that differ between two collection configs (only keys present
    in BOTH are compared; an absent key is not a mismatch, preserving back-compat
    with dumps that carry no config). Empty list => configs are consistent."""
    return sorted(k for k in set(cfg_a) & set(cfg_b) if cfg_a[k] != cfg_b[k])


def _config_path(out_json: str) -> str:
    return out_json + ".config"


def _cmd_collect(args) -> int:
    # Record the server-log line offset BEFORE the collect so observed batching is
    # windowed to the concurrent collection, not any earlier warmup/probe traffic.
    from_line = 0
    if args.server_log:
        with open(args.server_log) as f:
            from_line = sum(1 for _ in f)
    streams, meta = collect_concurrent(
        args.base_url, DEFAULT_PROMPTS, args.max_new_tokens
    )
    with open(args.out_json, "w") as f:
        json.dump(streams, f)  # streams-only dump: `compare` reads this
    print(f"collected {len(streams)} concurrent streams -> {args.out_json}")
    # Companion config (optional consumer): lets `compare` flag a mismatch between
    # the two sequential collects. Does NOT alter the streams dump above.
    cfg = collection_config(DEFAULT_PROMPTS, args.max_new_tokens)
    with open(_config_path(args.out_json), "w") as f:
        json.dump(cfg, f)
    if args.meta_out:
        with open(args.meta_out, "w") as f:
            json.dump(meta, f)
        s = summarize_accept_meta(meta)
        print(
            f"meta -> {args.meta_out} | mean accept_length="
            f"{s['mean_accept_length']} over {s['n_with_accept']}/{s['n']} requests"
        )
    if args.server_log:
        mx = observed_max_running(args.server_log, from_line=from_line)
        print(f"observed max #running-req during collect: {mx}")
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
    # If both dumps carry a companion config, surface a mismatch: the two
    # sequential collects must share max_new_tokens / prompt set, else divergences
    # below may be config artifacts rather than model divergence. Non-fatal.
    df_cfg_path, ns_cfg_path = _config_path(args.dflash_json), _config_path(args.nospec_json)
    if pathlib.Path(df_cfg_path).exists() and pathlib.Path(ns_cfg_path).exists():
        with open(df_cfg_path) as f:
            df_cfg = json.load(f)
        with open(ns_cfg_path) as f:
            ns_cfg = json.load(f)
        diff = config_mismatch(df_cfg, ns_cfg)
        if diff:
            print(
                f"WARNING: collection config mismatch ({', '.join(diff)}) -- "
                f"divergences may be config artifacts, not model divergence"
            )
    summary = compare_concurrent_streams(dflash, nospec, manifest)
    c = summary["counts"]
    for prompt, d in summary["per_prompt"].items():
        print(f"{d['status'].upper():10s} {prompt!r}")
    print("=" * 60)
    print(
        f"CONCURRENT PER-REQUEST PARITY: exact={c['exact']} "
        f"tolerated={c['tolerated']} fail={c['fail']} missing={c['missing']} "
        f"empty={c['empty']} -> {'PASS' if c['all_ok'] else 'FAIL'}"
    )
    return 0 if c["all_ok"] else 1


def build_report(
    dflash: Dict[str, List[int]],
    nospec: Dict[str, List[int]],
    manifest: Optional[Dict] = None,
    dflash_meta: Optional[Dict] = None,
    nospec_meta: Optional[Dict] = None,
    dflash_cfg: Optional[Dict] = None,
    nospec_cfg: Optional[Dict] = None,
) -> Dict:
    """Assemble the reboot-gated concurrent artifact's full evidence set into ONE
    consolidated document, composing the already-tested pure helpers:

      * ``parity``  -> per-request compare counts (exact/tolerated/fail/missing/empty)
      * ``config``  -> {"mismatch": [fields]} between the two collects (if both cfgs)
      * ``accept``  -> per-side accept-length summary (if the meta sidecars are given)
      * ``all_ok``  -> the PARITY verdict only; a config mismatch or accept summary is
                       informational and does NOT flip it (matches ``compare``).

    Pure (no I/O) so it is hermetically CPU-testable."""
    summary = compare_concurrent_streams(dflash, nospec, manifest)
    counts = summary["counts"]
    config = None
    if dflash_cfg is not None and nospec_cfg is not None:
        config = {"mismatch": config_mismatch(dflash_cfg, nospec_cfg)}
    accept = {}
    if dflash_meta is not None:
        accept["dflash"] = summarize_accept_meta(dflash_meta)
    if nospec_meta is not None:
        accept["nospec"] = summarize_accept_meta(nospec_meta)
    return {
        "parity": counts,
        "per_prompt": summary["per_prompt"],
        "config": config,
        "accept": accept,
        "all_ok": bool(counts["all_ok"]),
    }


def _maybe_load(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def _cmd_report(args) -> int:
    with open(args.dflash_json) as f:
        dflash = json.load(f)
    with open(args.nospec_json) as f:
        nospec = json.load(f)
    manifest = _maybe_load(args.tolerance_manifest)
    dflash_meta = _maybe_load(args.dflash_meta)
    nospec_meta = _maybe_load(args.nospec_meta)
    # Auto-pick up the companion collect configs if present (written by `collect`).
    df_cfg_path, ns_cfg_path = _config_path(args.dflash_json), _config_path(args.nospec_json)
    dflash_cfg = _maybe_load(df_cfg_path) if pathlib.Path(df_cfg_path).exists() else None
    nospec_cfg = _maybe_load(ns_cfg_path) if pathlib.Path(ns_cfg_path).exists() else None
    report = build_report(
        dflash, nospec, manifest, dflash_meta, nospec_meta, dflash_cfg, nospec_cfg
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"# wrote consolidated evidence report -> {args.out}")
    c = report["parity"]
    print("=" * 60)
    print("CONSOLIDATED DFLASH CONCURRENT-PARITY EVIDENCE")
    print(
        f"  parity: exact={c['exact']} tolerated={c['tolerated']} fail={c['fail']} "
        f"missing={c['missing']} empty={c['empty']} -> "
        f"{'PASS' if c['all_ok'] else 'FAIL'}"
    )
    if report["config"] is not None:
        mm = report["config"]["mismatch"]
        print(f"  config: {'consistent' if not mm else 'MISMATCH ' + ','.join(mm)}")
    for side, s in report["accept"].items():
        print(
            f"  accept[{side}]: mean_accept_length={s['mean_accept_length']} "
            f"over {s['n_with_accept']}/{s['n']} requests"
        )
    return 0 if report["all_ok"] else 1


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
    rp = sub.add_parser(
        "report", help="one consolidated evidence doc: parity + config + accept"
    )
    rp.add_argument("dflash_json")
    rp.add_argument("nospec_json")
    rp.add_argument("--tolerance-manifest")
    rp.add_argument("--dflash-meta", help="dflash --meta-out sidecar (accept-length)")
    rp.add_argument("--nospec-meta", help="no-spec --meta-out sidecar (accept-length)")
    rp.add_argument("--out", help="write the consolidated report JSON here")
    rp.set_defaults(func=_cmd_report)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
