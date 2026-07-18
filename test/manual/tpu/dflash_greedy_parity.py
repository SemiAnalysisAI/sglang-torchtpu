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


def _tolerated_divergence(
    manifest: Dict[str, object],
    prompt: str,
    idx: int,
    spec_tok: int,
    ref_tok: int,
) -> Tuple[bool, str]:
    """Decide tolerance by RECOMPUTING the policy from manifest evidence and
    binding it to the CURRENT dumps — never trust the stored ``verdict``.

    Rejects (returns False + reason) unless ALL hold: the manifest entry exists
    and its ``index`` matches; its recorded ``epsilon`` is not looser than the
    code's ``TIE_MARGIN_EPS``; its ``dflash_token``/``nospec_token`` equal the
    current spec/ref dump tokens at ``idx``; and re-running
    :func:`classify_divergence` on the recorded ``nospec_topk`` + ``verify_top2``
    reproduces a ``bf16-tie`` verdict. So a stale, forged, or looser-epsilon
    manifest cannot make a different current run pass.
    """
    entry = manifest.get(prompt)
    if not isinstance(entry, dict):
        return False, "no manifest entry for prompt"
    if int(entry.get("index", -1)) != idx:
        return False, f"manifest index {entry.get('index')} != current divergence {idx}"
    eps = entry.get("epsilon")
    if eps is None or float(eps) > TIE_MARGIN_EPS + 1e-9:
        return False, f"manifest epsilon {eps} looser than code eps {TIE_MARGIN_EPS}"
    if int(entry.get("dflash_token", -1)) != int(spec_tok):
        return False, f"manifest dflash_token {entry.get('dflash_token')} != current {spec_tok}"
    if int(entry.get("nospec_token", -1)) != int(ref_tok):
        return False, f"manifest nospec_token {entry.get('nospec_token')} != current {ref_tok}"
    topk = entry.get("nospec_topk")
    if not topk:
        return False, "manifest missing nospec_topk evidence"
    recomputed = classify_divergence(
        topk, idx, int(spec_tok), verify_top2=entry.get("verify_top2")
    )
    if recomputed.get("verdict") != "bf16-tie":
        return False, "recomputed verdict is not bf16-tie: " + "; ".join(
            recomputed.get("reasons", [])
        )
    return True, "bf16-tie (recomputed from evidence)"


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
        tolerated = False
        if manifest and idx < len(spec[prompt]) and idx < len(ref[prompt]):
            tolerated, why = _tolerated_divergence(
                manifest, prompt, idx, int(spec[prompt][idx]), int(ref[prompt][idx])
            )
            if tolerated:
                print(f"TOLERATED  {prompt!r}  @ {idx}  ({why})")
                continue
            print(f"FAIL  {prompt!r}  divergence @ {idx} NOT tolerated: {why}")
            all_ok = False
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
    ranked: List[List[float]],
    index: int,
    spec_token: int,
    verify_top2: Optional[List[List[float]]] = None,
) -> Dict[str, object]:
    """Pure, DECISIVE classification of one divergent position (no server).

    ``ranked`` is the NON-SPEC decode top-k at ``index`` as ``[(token_id, logprob),
    ...]`` sorted descending; log-softmax preserves differences, so the top-2
    logprob gap is the raw top-2 LOGIT margin. ``verify_top2`` is DFlash's OWN
    verify-forward top-2 at that position as ``[(token_id, logit), (token_id,
    logit)]`` (from the env-gated verifier trace).

    Verdict is ``bf16-tie`` ONLY when ALL hold (else ``MISMATCH`` with reasons):
      - non-spec: the DFlash-emitted ``spec_token`` is the non-spec RUNNER-UP
        (rank 1) and the non-spec top-2 margin ≤ :data:`TIE_MARGIN_EPS`;
      - verify: DFlash's verify top-1 == the emitted token (correct accept), the
        verify top-2 == the non-spec top-1 (SAME token pair, swapped order), and
        the verify top-2 margin ≤ :data:`TIE_MARGIN_EPS`.
    The verify side rules out a materialization-degraded flip; requiring the
    verify trace means a no-spec-only manifest cannot yield ``bf16-tie``.
    CPU-unit-tested in test_dflash_tpu_units.py::TestDflashDivergenceClassifier.
    """
    ranked = [(int(t), float(lp)) for t, lp in ranked]
    top1_id, top1_lp = ranked[0]
    top2_id, top2_lp = ranked[1] if len(ranked) > 1 else (None, float("-inf"))
    margin = top1_lp - top2_lp
    spec_rank = next((r for r, (tid, _) in enumerate(ranked) if tid == spec_token), None)
    spec_lp = next((lp for tid, lp in ranked if tid == spec_token), None)

    reasons: List[str] = []
    if spec_rank != 1:
        reasons.append(f"dflash token rank in non-spec top-k is {spec_rank}, not 1")
    if margin > TIE_MARGIN_EPS:
        reasons.append(f"non-spec top-2 margin {margin:.4f} > eps {TIE_MARGIN_EPS}")

    out: Dict[str, object] = {
        "index": index,
        "policy": "rank1-runner-up-swap-within-2sigma-bf16-noise",
        "epsilon": TIE_MARGIN_EPS,
        "dflash_token": spec_token,
        "nospec_token": top1_id,
        "nospec_topk": [[t, round(lp, 5)] for t, lp in ranked],
        "top2_margin": round(margin, 6),
        "dflash_rank_in_nospec_topk": spec_rank,
        "dflash_logprob": round(spec_lp, 5) if spec_lp is not None else None,
    }

    if verify_top2 is None:
        reasons.append("no DFlash verify-forward top-2 evidence (verifier trace missing)")
        out["verify_top2"] = None
    else:
        vt = [(int(t), float(v)) for t, v in verify_top2]
        v1_id, v1_v = vt[0]
        v2_id, v2_v = vt[1]
        vmargin = v1_v - v2_v
        out["verify_top2"] = [[v1_id, round(v1_v, 5)], [v2_id, round(v2_v, 5)]]
        out["verify_margin"] = round(vmargin, 6)
        if v1_id != spec_token:
            reasons.append(f"verify top-1 {v1_id} != emitted token {spec_token} (accept bug)")
        if v2_id != top1_id:
            reasons.append(f"verify top-2 {v2_id} != non-spec top-1 {top1_id} (not a swap)")
        if vmargin > TIE_MARGIN_EPS:
            reasons.append(f"verify top-2 margin {vmargin:.4f} > eps {TIE_MARGIN_EPS}")

    out["reasons"] = reasons
    out["verdict"] = "bf16-tie" if not reasons else "MISMATCH"
    return out


def _load_verify_posmap(path: str) -> Dict[int, List[List[float]]]:
    """Load the env-gated DFlash verifier trace (JSONL) into {abs_position:
    verify_top2}. Each record commits tokens at ``start_pos + k`` whose verify
    top-2 is ``verify_top2[k]``. Assumes the trace covers a single request's
    stream (run one prompt with SGLANG_DFLASH_VERIFY_TRACE set)."""
    posmap: Dict[int, List[List[float]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            start = int(rec["start_pos"])
            for k, t2 in enumerate(rec["verify_top2"]):
                posmap[start + k] = t2
    return posmap


def _diagnose_position(
    base_url: str,
    prompt: str,
    index: int,
    spec_token: int,
    top_k: int,
    verify_posmap: Optional[Dict[int, List[List[float]]]] = None,
) -> Dict[str, object]:
    """Query the non-spec target for its top-k at ``index`` (given the shared
    greedy prefix, which both runners reproduce deterministically), merge the
    DFlash verify-forward top-2 from the trace, and classify decisively."""
    otl = _greedy_top_logprobs(base_url, prompt, index + 1, top_k)
    if index >= len(otl):
        return {"error": f"non-spec produced only {len(otl)} tokens; need index {index}"}
    # each entry: [logprob, token_id, (decoded)] sorted desc by logprob
    ranked = [[int(e[1]), float(e[0])] for e in otl[index]]
    verify_top2 = verify_posmap.get(index) if verify_posmap else None
    return classify_divergence(ranked, index, spec_token, verify_top2=verify_top2)


def _cmd_diagnose(args) -> int:
    with open(args.spec_json) as f:
        spec = json.load(f)
    with open(args.ref_json) as f:
        ref = json.load(f)
    verify_posmap = (
        _load_verify_posmap(args.verify_trace) if args.verify_trace else None
    )
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
            d = _diagnose_position(
                args.base_url, prompt, idx, spec_token, args.top_k, verify_posmap
            )
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
    dg.add_argument(
        "--verify-trace",
        help="DFlash verifier trace JSONL (SGLANG_DFLASH_VERIFY_TRACE output from a "
        "single-prompt DFlash run) to merge the verify-forward top-2 evidence",
    )
    dg.set_defaults(func=_cmd_diagnose)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
