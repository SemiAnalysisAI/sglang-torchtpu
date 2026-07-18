#!/usr/bin/env python
"""DFlash greedy stream-parity harness — token-ID based & sequential.

A single v5e host has only 4 TPU chips, so a DFlash TP=4 server and a non-spec
TP=4 baseline CANNOT run at the same time. This harness is therefore SEQUENTIAL:
you run `collect` against one server, tear it down, run `collect` against the
other, then run `compare` on the two saved JSON dumps OFFLINE. Comparison is by
exact TOKEN ID (greedy determinism), not completion text.

The verdict function ``diff_token_streams`` is pure and CPU-unit-tested in
test/manual/tpu/test_dflash_tpu_units.py::TestDflashStreamParityVerdict.

Usage (after the torch_tpu TP>1 bug is fixed and a server is READY):
    # 1) DFlash server up on :30030
    ./.venv/bin/python test/manual/tpu/dflash_greedy_parity.py collect \
        http://127.0.0.1:30030 /tmp/dflash.json
    # 2) tear it down, bring up the non-spec baseline on :30031
    ./.venv/bin/python test/manual/tpu/dflash_greedy_parity.py collect \
        http://127.0.0.1:30031 /tmp/nospec.json
    # 3) offline token-ID diff
    ./.venv/bin/python test/manual/tpu/dflash_greedy_parity.py compare \
        /tmp/dflash.json /tmp/nospec.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Dict, List, Optional

DEFAULT_PROMPTS = [
    "The capital of France is",
    "List the first five prime numbers:",
    "Q: What is 17 + 25? A:",
    "Once upon a time,",
    "Explain speculative decoding in one sentence:",
]
DEFAULT_MAX_NEW_TOKENS = 64


def diff_token_streams(
    spec_ids: List[int], ref_ids: List[int]
) -> Dict[str, object]:
    """Compare two greedy token-ID streams for exact equality.

    Returns a verdict dict: ``match`` (bool), ``first_divergence`` (int index of
    the first differing position, or the shorter length if one is a prefix of the
    other; ``-1`` when identical), and the two lengths. Greedy decoding is
    deterministic, so a correct DFlash stream must be token-for-token identical to
    the non-speculative target stream.
    """
    n = min(len(spec_ids), len(ref_ids))
    for i in range(n):
        if spec_ids[i] != ref_ids[i]:
            return {
                "match": False,
                "first_divergence": i,
                "spec_len": len(spec_ids),
                "ref_len": len(ref_ids),
            }
    if len(spec_ids) != len(ref_ids):
        return {
            "match": False,
            "first_divergence": n,  # prefix matches; lengths differ
            "spec_len": len(spec_ids),
            "ref_len": len(ref_ids),
        }
    return {
        "match": True,
        "first_divergence": -1,
        "spec_len": len(spec_ids),
        "ref_len": len(ref_ids),
    }


def _generate_token_ids(
    base_url: str, prompt: str, max_new_tokens: int
) -> List[int]:
    """Greedy-generate and return the output token IDs from one SGLang server.

    Uses the native /generate endpoint, which returns generated token IDs in
    the top-level ``output_ids`` field.  Do not request logprobs: DFlash
    intentionally rejects ``return_logprob=True``.
    """
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    if out.get("output_ids") is not None:
        return [int(t) for t in out["output_ids"]]
    meta = out.get("meta_info", {})
    otl = meta.get("output_token_logprobs")
    if otl:
        return [int(entry[1]) for entry in otl]
    if meta.get("output_ids"):
        return [int(t) for t in meta["output_ids"]]
    raise RuntimeError(
        f"could not extract token ids from /generate response keys={list(out.keys())}"
    )


def collect_tokens(
    base_url: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, List[int]]:
    prompts = prompts or DEFAULT_PROMPTS
    return {p: _generate_token_ids(base_url, p, max_new_tokens) for p in prompts}


def _cmd_collect(args) -> int:
    result = collect_tokens(args.base_url, max_new_tokens=args.max_new_tokens)
    with open(args.out_json, "w") as f:
        json.dump(result, f)
    print(f"collected {len(result)} prompt streams from {args.base_url} -> {args.out_json}")
    return 0


def _tolerated_divergence(manifest: Dict[str, object], prompt: str, idx: int) -> bool:
    """True iff the tolerance manifest confirms this exact divergence is a
    bf16 tie-break (rank-1 swap within noise). Any mismatch of prompt/index or
    a non-tie verdict is NOT tolerated."""
    entry = manifest.get(prompt)
    if not isinstance(entry, dict):
        return False
    return entry.get("verdict") == "bf16-tie" and int(entry.get("index", -1)) == idx


def _cmd_compare(args) -> int:
    with open(args.spec_json) as f:
        spec = json.load(f)
    with open(args.ref_json) as f:
        ref = json.load(f)
    manifest: Dict[str, object] = {}
    if getattr(args, "tolerance_manifest", None):
        with open(args.tolerance_manifest) as f:
            manifest = json.load(f)
    all_ok = True
    for prompt in spec:
        if prompt not in ref:
            print(f"MISSING in ref: {prompt!r}")
            all_ok = False
            continue
        v = diff_token_streams(spec[prompt], ref[prompt])
        if v["match"]:
            print(f"PASS  {prompt!r}  (len={v['spec_len']})")
            continue
        idx = int(v["first_divergence"])
        if manifest and _tolerated_divergence(manifest, prompt, idx):
            print(
                f"TOLERATED  {prompt!r}  bf16 tie-break @ {idx} "
                f"(margin={manifest[prompt].get('top2_margin')}, "
                f"dflash=target runner-up)"
            )
            continue
        all_ok = False
        print(
            f"FAIL  {prompt!r}  first token divergence @ {idx} "
            f"(dflash_len={v['spec_len']}, nospec_len={v['ref_len']})"
        )
    print("=" * 60)
    policy = " (with bf16 tie-break tolerance manifest)" if manifest else " (exact)"
    print(f"GREEDY TOKEN-ID STREAM PARITY{policy}:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


# A divergence is classified a bf16 argmax TIE (not a bug) only when BOTH:
#   (1) the DFlash-emitted token is the non-spec RUNNER-UP (rank 1 in the
#       target's own top-k) — i.e. DFlash merely swapped the top-2 (a real
#       verify/materialization bug would emit a token that is NOT the runner-up),
#       and
#   (2) the non-spec top-2 logit margin is within the bf16 forward-noise scale.
# log-softmax preserves differences, so (top-1 logprob - top-2 logprob) == the
# raw top-2 LOGIT margin. Noise scale: bf16 mantissa ~8 bits -> relative error
# ~2^-8; Llama final logits are O(10-20), so each token's logit carries ~0.05-0.08
# absolute bf16 error and a top-2 logit DIFFERENCE carries ~sqrt(2)*0.06 ~ 0.09
# (1 sigma). The verify-batch and decode-batch target forwards disagree at this
# scale, so a top-2 swap within ~2 sigma is a numerically-explained tie-break.
_BF16_LOGIT_DIFF_SIGMA = 0.09
TIE_MARGIN_EPS = 2.0 * _BF16_LOGIT_DIFF_SIGMA  # ~0.18 nats


def _greedy_top_logprobs(
    base_url: str, prompt: str, n_tokens: int, top_k: int
) -> List[List[List[float]]]:
    """Greedy-generate ``n_tokens`` and return per-position top-k output logprobs.

    Returns ``output_top_logprobs`` from the non-spec server: a list (one entry
    per generated position) of ``[[logprob, token_id, ...], ...]`` sorted by
    logprob descending. Requires the target to allow ``return_logprob`` (non-spec
    does; DFlash intentionally rejects it — hence we diagnose against non-spec).
    """
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": n_tokens},
            "stream": False,
            "return_logprob": True,
            "top_logprobs_num": top_k,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    otl = out.get("meta_info", {}).get("output_top_logprobs")
    if not otl:
        raise RuntimeError(
            "non-spec /generate returned no output_top_logprobs; "
            f"response keys={list(out.get('meta_info', {}).keys())}"
        )
    return otl


def classify_divergence(
    ranked: List[List[float]], index: int, spec_token: int
) -> Dict[str, object]:
    """Pure classification of one divergent position (no server).

    ``ranked`` is the target's top-k at ``index`` as ``[(token_id, logprob), ...]``
    sorted by logprob descending. Verdict is ``bf16-tie`` iff the DFlash-emitted
    ``spec_token`` is the target's rank-1 runner-up AND the top-2 margin is within
    the bf16 forward-noise scale (:data:`TIE_MARGIN_EPS`); otherwise ``MISMATCH``.
    CPU-unit-tested in test_dflash_tpu_units.py::TestDflashDivergenceClassifier.
    """
    ranked = [(int(t), float(lp)) for t, lp in ranked]
    top1_id, top1_lp = ranked[0]
    top2_id, top2_lp = ranked[1] if len(ranked) > 1 else (None, float("-inf"))
    margin = top1_lp - top2_lp
    spec_rank = next((r for r, (tid, _) in enumerate(ranked) if tid == spec_token), None)
    spec_lp = next((lp for tid, lp in ranked if tid == spec_token), None)
    is_tie = spec_rank == 1 and margin <= TIE_MARGIN_EPS
    return {
        "index": index,
        "nospec_top1": [top1_id, round(top1_lp, 5)],
        "nospec_top2": [top2_id, round(top2_lp, 5)],
        "top2_margin": round(margin, 6),
        "dflash_token": spec_token,
        "dflash_rank_in_nospec_topk": spec_rank,
        "dflash_logprob": round(spec_lp, 5) if spec_lp is not None else None,
        "verdict": "bf16-tie" if is_tie else "MISMATCH",
    }


def _diagnose_position(
    base_url: str, prompt: str, index: int, spec_token: int, top_k: int
) -> Dict[str, object]:
    """Query the non-spec target for its top-k at ``index`` (given the shared
    greedy prefix, which both runners reproduce deterministically) and classify
    the divergence."""
    otl = _greedy_top_logprobs(base_url, prompt, index + 1, top_k)
    if index >= len(otl):
        return {"error": f"non-spec produced only {len(otl)} tokens; need index {index}"}
    # each entry: [logprob, token_id, (decoded)] sorted desc by logprob
    ranked = [[int(e[1]), float(e[0])] for e in otl[index]]
    return classify_divergence(ranked, index, spec_token)


def _cmd_diagnose(args) -> int:
    with open(args.spec_json) as f:
        spec = json.load(f)
    with open(args.ref_json) as f:
        ref = json.load(f)
    any_mismatch = False
    manifest: Dict[str, Dict[str, object]] = {}
    for prompt in spec:
        if prompt not in ref:
            continue
        v = diff_token_streams(spec[prompt], ref[prompt])
        if v["match"]:
            continue
        idx = int(v["first_divergence"])
        if idx >= len(spec[prompt]):
            d = {"index": idx, "verdict": "LENGTH-ONLY"}
            print(f"LENGTH-ONLY divergence {prompt!r} @ {idx} (no token to diagnose)")
        else:
            spec_token = int(spec[prompt][idx])
            d = _diagnose_position(args.base_url, prompt, idx, spec_token, args.top_k)
            print(f"--- {prompt!r} @ token {idx} ---")
            print(json.dumps(d, indent=2))
        manifest[prompt] = d
        if d.get("verdict") != "bf16-tie":
            any_mismatch = True
    if args.out_manifest:
        with open(args.out_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"# wrote tolerance manifest -> {args.out_manifest}")
    print("=" * 60)
    print(
        f"DIAGNOSE: {len(manifest)} divergence(s); "
        + ("all bf16-tie" if not any_mismatch else "REAL MISMATCH present")
    )
    return 1 if any_mismatch else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="collect greedy token IDs from ONE server")
    c.add_argument("base_url")
    c.add_argument("out_json")
    c.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    c.set_defaults(func=_cmd_collect)
    cmp = sub.add_parser("compare", help="offline token-ID diff of two dumps")
    cmp.add_argument("spec_json")
    cmp.add_argument("ref_json")
    cmp.add_argument(
        "--tolerance-manifest",
        help="optional diagnose manifest; divergences it confirms as bf16 tie-breaks "
        "are reported TOLERATED instead of FAIL. Omit for strict exact parity.",
    )
    cmp.set_defaults(func=_cmd_compare)
    dg = sub.add_parser(
        "diagnose",
        help="for each divergent prompt, query the NON-SPEC server for its top-k "
        "distribution at the divergence and classify bf16-tie vs real mismatch",
    )
    dg.add_argument("base_url", help="NON-SPEC server URL (allows return_logprob)")
    dg.add_argument("spec_json", help="DFlash token dump")
    dg.add_argument("ref_json", help="non-spec token dump")
    dg.add_argument("--top-k", type=int, default=5)
    dg.add_argument(
        "--out-manifest", help="write the tie/mismatch classification JSON here"
    )
    dg.set_defaults(func=_cmd_diagnose)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
