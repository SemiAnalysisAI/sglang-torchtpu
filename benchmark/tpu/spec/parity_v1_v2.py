#!/usr/bin/env python3
"""Per-request output-token capture for the V1<->V2 spec-decode parity gate.

Sends a fixed prompt set to a running SGLang server at temperature 0 and writes,
for each request, the exact output token-id sequence (the authoritative parity
signal) plus the decoded text. Run once against a V1 server and once against a
V2 server, then diff the two JSON files with --compare.

Usage:
  capture:  python parity_v1_v2.py --url http://127.0.0.1:30010 --out v2.json \
                                    [--parallel 8] [--max-new 128]
  compare:  python parity_v1_v2.py --compare v1.json v2.json
"""

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.request

# Per-request timeout (s). Cold/first-touch compiles of a new shape can take many
# minutes, so allow a generous, env-tunable ceiling (default 10 min).
REQUEST_TIMEOUT = int(os.environ.get("PARITY_REQUEST_TIMEOUT", "600"))

# Fixed, deterministic prompt set. Mixed lengths/topics so the draft/verify loop
# is actually exercised; greedy (temp 0) so output is deterministic and the two
# workers must agree token-for-token.
PROMPTS = [
    "Explain why the sky appears blue, step by step.",
    "Write a short Python function that returns the n-th Fibonacci number.",
    "List three differences between TCP and UDP.",
    "What is the capital of France, and what river runs through it?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Solve: a train travels 60 km in 45 minutes. What is its speed in km/h?",
    "Describe how a hash map works and its average lookup complexity.",
    "Translate 'good morning, how are you?' into Spanish and French.",
]


def _post(url, payload, timeout=REQUEST_TIMEOUT):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/generate", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _output_token_ids(meta):
    """Extract output token ids from meta_info.output_token_logprobs.

    Each entry is [logprob, token_id, text]; token_id is the authoritative id.
    """
    otl = meta.get("output_token_logprobs")
    if not otl:
        return None
    return [int(e[1]) for e in otl]


def capture(url, parallel, max_new):
    def one(idx_prompt):
        idx, prompt = idx_prompt
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new,
            },
            # /generate returns output_ids at the top level; return_logprob is not
            # supported on the TPU backend, so don't request it.
        }
        r = _post(url, payload)
        meta = r.get("meta_info", {})
        # /generate returns output_ids at the top level; fall back to logprobs.
        ids = r.get("output_ids")
        if ids is None:
            ids = _output_token_ids(meta)
        return idx, {
            "prompt": prompt,
            "text": r.get("text", ""),
            "output_ids": ids,
            "completion_tokens": meta.get("completion_tokens"),
        }

    results = [None] * len(PROMPTS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
        for idx, res in ex.map(one, list(enumerate(PROMPTS))):
            results[idx] = res
    return results


def compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    if len(a) != len(b):
        print(f"PARITY FAIL: request count differs ({len(a)} vs {len(b)})")
        return 1
    mismatches = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        ids_a, ids_b = ra.get("output_ids"), rb.get("output_ids")
        if ids_a is None or ids_b is None:
            # Token IDs are the authoritative parity signal (text is diagnostic
            # only). Missing IDs on either side is a hard failure, not a fallback
            # to text comparison, so a tokenization difference cannot be hidden.
            mismatches.append(
                (
                    i,
                    f"missing output_ids (A={ids_a is not None}, B={ids_b is not None})",
                )
            )
            continue
        if ids_a != ids_b:
            # Find first divergence index for a precise report.
            n = min(len(ids_a), len(ids_b))
            j = next((k for k in range(n) if ids_a[k] != ids_b[k]), n)
            mismatches.append(
                (i, f"token ids differ at pos {j} (len {len(ids_a)} vs {len(ids_b)})")
            )
    if mismatches:
        print(f"PARITY FAIL: {len(mismatches)}/{len(a)} requests differ")
        for i, why in mismatches:
            print(f"  req {i}: {why}")
            print(f"    A[{i}] text: {a[i].get('text','')[:120]!r}")
            print(f"    B[{i}] text: {b[i].get('text','')[:120]!r}")
        return 1
    print(f"PARITY PASS: all {len(a)} requests byte-identical (output token ids)")
    return 0


# ----------------------------------------------------------------------------
# Structured bf16 near-tie diagnostic (AC-2 amendment, fail-closed).
#
# Token IDs remain the authoritative parity signal: `compare()` is STRICT (any
# mismatch is nonzero). The amendment only admits a V1!=V2 divergence when it is a
# proven bf16 near-tie, established with DIRECT top-2 logit evidence (a sidecar
# trace emitted by the server's greedy verify path, env SGLANG_TPU_SPEC_TRACE),
# never by positional prose. A near-tie at the first divergence position k of a
# request requires ALL of:
#   - tokens identical for positions [0, k) (cascade precondition),
#   - V1 emitted A and V2 emitted B at k (A != B),
#   - in V1's graph at k: B is V1's 2nd-choice and (logit[A]-logit[B]) <= margin,
#   - in V2's graph at k: A is V2's 2nd-choice and (logit[B]-logit[A]) <= margin.
# Everything after k is autoregressive cascade (different context) and not checked.
# Missing trace data, a wrong-token pair, or a margin above tolerance => FAIL.
# ----------------------------------------------------------------------------

# Default top-1/top-2 logit-gap tolerance (logit units) for "bf16 near-tie". A
# genuinely wrong token wins by a large margin; bf16 ties across two compiled
# graphs sit well under this. The trace records the actual margins observed.
DEFAULT_NEAR_TIE_MARGIN = 1.0


def _load_trace(path):
    """Read a JSONL spec-trace into {rid: [[emitted_tok, [id1,id2], [lg1,lg2]], ...]}.

    Records are appended per request per verify step; concatenating in file order
    reconstructs each request's emitted-token sequence in output order.
    """
    by_rid = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_rid.setdefault(rec["rid"], []).extend(rec["emitted"])
    return by_rid


def _align_trace_to_outputs(outputs, by_rid):
    """Return a list, by prompt index, of the per-position top-2 trace entries.

    A request's reconstructed emitted-token sequence equals (or prefixes) its
    output_ids, so we match each output to the rid whose emitted tokens start with
    that output. Returns None for any prompt with no unique trace match.
    """
    # rid -> emitted token tuple
    rid_seq = {rid: [e[0] for e in entries] for rid, entries in by_rid.items()}
    aligned = []
    for rec in outputs:
        ids = rec.get("output_ids")
        match = None
        if ids is not None:
            for rid, seq in rid_seq.items():
                if seq[: len(ids)] == ids:
                    match = by_rid[rid]
                    break
        aligned.append(match)
    return aligned


def allow_near_tie_compare(path_a, path_b, trace_a, trace_b, margin):
    """Fail-closed near-tie gate. 0 only if every V1!=V2 divergence is a proven tie."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    if len(a) != len(b):
        print(f"NEAR-TIE FAIL: request count differs ({len(a)} vs {len(b)})")
        return 1
    ta = _align_trace_to_outputs(a, _load_trace(trace_a))
    tb = _align_trace_to_outputs(b, _load_trace(trace_b))

    failures, ties = [], []
    for i, (ra, rb) in enumerate(zip(a, b)):
        ia, ib = ra.get("output_ids"), rb.get("output_ids")
        if ia is None or ib is None:
            failures.append((i, "missing output_ids"))
            continue
        if ia == ib:
            continue  # byte-identical: nothing to excuse
        if len(ia) != len(ib):
            failures.append(
                (
                    i,
                    f"length differs ({len(ia)} vs {len(ib)}) — not a same-position tie",
                )
            )
            continue
        k = next((p for p in range(len(ia)) if ia[p] != ib[p]), None)
        if k is None:
            continue
        ea, eb = ta[i], tb[i]
        if ea is None or eb is None or k >= len(ea) or k >= len(eb):
            failures.append((i, f"no trace evidence at divergence pos {k}"))
            continue
        # ea[k] = [emitted, [top1_id, top2_id], [top1_lg, top2_lg]]
        A, (a1, a2), (av1, av2) = ia[k], ea[k][1], ea[k][2]
        B, (b1, b2), (bv1, bv2) = ib[k], eb[k][1], eb[k][2]
        ma, mb = av1 - av2, bv1 - bv2
        ok = (a1 == A and a2 == B and ma <= margin) and (
            b1 == B and b2 == A and mb <= margin
        )
        if ok:
            ties.append((i, k, ma, mb))
        else:
            failures.append(
                (
                    i,
                    f"pos {k}: A={A} B={B} | V1 top2={ea[k][1]} gap={ma:.4f} | "
                    f"V2 top2={eb[k][1]} gap={mb:.4f} (margin={margin}) — not a near-tie",
                )
            )

    for i, k, ma, mb in ties:
        print(f"  near-tie req {i} @pos {k}: V1 gap={ma:.4f}, V2 gap={mb:.4f}")
    if failures:
        print(f"NEAR-TIE FAIL: {len(failures)} request(s) not provable near-ties")
        for i, why in failures:
            print(f"  req {i}: {why}")
        return 1
    print(
        f"NEAR-TIE PASS: all {len(ties)} divergence(s) are proven bf16 near-ties "
        f"(max gap {max([max(t[2], t[3]) for t in ties], default=0):.4f} <= {margin})"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--out")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument(
        "--allow-near-tie",
        nargs=4,
        metavar=("A", "B", "TRACE_A", "TRACE_B"),
        help="fail-closed near-tie check using server top-2 logit traces",
    )
    ap.add_argument("--margin", type=float, default=DEFAULT_NEAR_TIE_MARGIN)
    args = ap.parse_args()

    if args.allow_near_tie:
        sys.exit(allow_near_tie_compare(*args.allow_near_tie, margin=args.margin))

    if args.compare:
        sys.exit(compare(*args.compare))

    if not args.url or not args.out:
        ap.error("capture mode requires --url and --out")
    results = capture(args.url, args.parallel, args.max_new)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    ncap = sum(1 for r in results if r.get("output_ids"))
    print(f"captured {len(results)} requests to {args.out} " f"({ncap} with token ids)")


if __name__ == "__main__":
    main()
