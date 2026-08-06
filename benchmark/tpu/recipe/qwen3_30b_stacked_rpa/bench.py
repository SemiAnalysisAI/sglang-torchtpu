#!/usr/bin/env python3
"""Profile skewed long-context decode after preloading the long KV prefixes."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import requests


def make_input_ids(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 30000) for _ in range(length)]


def post(session: requests.Session, url: str, body: dict, timeout: int = 3600):
    response = session.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--long-requests", type=int, default=2)
    parser.add_argument("--short-isl", type=int, default=512)
    parser.add_argument("--long-isl", type=int, default=240_000)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--context-limit", type=int, default=262_144)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--profile-dir", default="/tmp/qwen3-30b-skewed-profile")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--reuse-prefix-cache", action="store_true")
    args = parser.parse_args()

    if not 0 < args.long_requests <= args.concurrency:
        parser.error("long-requests must be between 1 and concurrency")
    if args.long_isl + args.output_len > args.context_limit:
        parser.error("long-isl + output-len exceeds --context-limit")

    session = requests.Session()
    base_url = args.base_url.rstrip("/")
    health = session.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()

    # Distinct long prefixes prevent the two long requests from sharing KV.
    long_inputs = [
        make_input_ids(args.long_isl, seed=10_000 + i)
        for i in range(args.long_requests)
    ]
    short_inputs = [
        make_input_ids(args.short_isl, seed=20_000 + i)
        for i in range(args.concurrency - args.long_requests)
    ]

    if not args.reuse_prefix_cache:
        session.post(f"{base_url}/flush_cache", timeout=60).raise_for_status()
        print(
            f"Preloading {len(long_inputs)} distinct {args.long_isl}-token prefixes..."
        )
        for index, input_ids in enumerate(long_inputs):
            started = time.perf_counter()
            post(
                session,
                f"{base_url}/generate",
                {
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                    },
                },
            )
            print(f"  prefix {index}: {time.perf_counter() - started:.2f}s")

    profile_dir = Path(args.profile_dir).resolve()
    if not args.no_profile:
        profile_dir.mkdir(parents=True, exist_ok=True)
        post(
            session,
            f"{base_url}/start_profile",
            {
                "activities": ["CPU", "TPU"],
                "num_steps": args.profile_steps,
                "output_dir": str(profile_dir),
                "profile_prefix": "qwen3-30b-skewed-decode",
                "record_shapes": True,
            },
            timeout=120,
        )

    inputs = long_inputs + short_inputs
    sampling_params = [
        {
            "temperature": 0,
            "max_new_tokens": args.output_len,
            "ignore_eos": True,
        }
        for _ in inputs
    ]
    print(
        f"Running C{args.concurrency}: {args.long_requests}x{args.long_isl} + "
        f"{len(short_inputs)}x{args.short_isl}, OSL={args.output_len}"
    )
    started = time.perf_counter()
    results = post(
        session,
        f"{base_url}/generate",
        {"input_ids": inputs, "sampling_params": sampling_params},
    )
    duration = time.perf_counter() - started
    if not isinstance(results, list) or len(results) != args.concurrency:
        raise RuntimeError(f"Expected {args.concurrency} results, got {type(results)}")

    output_tokens = sum(
        int(result.get("meta_info", {}).get("completion_tokens", args.output_len))
        for result in results
    )
    output_ids = [result.get("output_ids", []) for result in results]
    first_token_ids = [ids[0] if ids else None for ids in output_ids]
    output_sha256 = hashlib.sha256(
        json.dumps(output_ids, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "concurrency": args.concurrency,
        "long_requests": args.long_requests,
        "long_isl": args.long_isl,
        "short_requests": len(short_inputs),
        "short_isl": args.short_isl,
        "output_len": args.output_len,
        "duration_s": duration,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / duration,
        "first_token_ids": first_token_ids,
        "output_sha256": output_sha256,
        "profile_dir": None if args.no_profile else str(profile_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
