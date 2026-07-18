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


def _cmd_compare(args) -> int:
    with open(args.spec_json) as f:
        spec = json.load(f)
    with open(args.ref_json) as f:
        ref = json.load(f)
    all_match = True
    for prompt in spec:
        if prompt not in ref:
            print(f"MISSING in ref: {prompt!r}")
            all_match = False
            continue
        v = diff_token_streams(spec[prompt], ref[prompt])
        if v["match"]:
            print(f"PASS  {prompt!r}  (len={v['spec_len']})")
        else:
            all_match = False
            print(
                f"FAIL  {prompt!r}  first token divergence @ {v['first_divergence']} "
                f"(dflash_len={v['spec_len']}, nospec_len={v['ref_len']})"
            )
    print("=" * 60)
    print("GREEDY TOKEN-ID STREAM PARITY:", "PASS" if all_match else "FAIL")
    return 0 if all_match else 1


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
    cmp.set_defaults(func=_cmd_compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
