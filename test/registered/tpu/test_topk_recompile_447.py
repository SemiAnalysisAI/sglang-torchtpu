#!/usr/bin/env python3
"""Exercise PR #447 sampling-mode transitions on a frozen TP16 server."""

import json
import os
import subprocess
import urllib.request
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=1800, suite="nightly-tpu-multihost", nightly=True)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/topk_recompile_447_mh.sh",
    "weights_pvc": "model-weights-rox",
}

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("MODEL", "/weights/Qwen3-Coder-480B-A35B-Instruct-FP8")
SERVER_LOG = Path(os.environ.get("SERVER_LOG", "/tmp/server.log"))


def request_json(path: str, payload: dict, timeout: int = 1200) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_batch(label: str, sampling_params: dict) -> None:
    print(f"== {label} ==", flush=True)
    result = request_json(
        "/generate",
        {
            "text": ["Hello"] * 64,
            "sampling_params": {
                "max_new_tokens": 2,
                "ignore_eos": True,
                **sampling_params,
            },
        },
    )
    if not isinstance(result, list) or len(result) != 64:
        raise SystemExit(
            f"{label} returned an unexpected batch response: "
            f"{type(result).__name__}"
        )
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=30) as response:
        print(f"{label} health: HTTP {response.status}", flush=True)


def main() -> None:
    print("== prime Dynamo with one B=1 greedy request ==", flush=True)
    prime = request_json(
        "/generate",
        {
            "text": "Hello",
            "sampling_params": {"temperature": 0, "max_new_tokens": 2},
        },
    )
    print(f"prime response: {json.dumps(prime, sort_keys=True)}", flush=True)

    print("== trigger the B=64 greedy variant ==", flush=True)
    bench_env = os.environ.copy()
    bench_env.update(
        {
            "MODEL": MODEL,
            "DATASET_NAME": "random-ids",
            "TOKENIZE_PROMPT": "1",
            "BATCH_SIZE": "64",
            "NUM_PROMPTS": "64",
            "ISL": "64",
            "OSL": "8",
            "WARMUP_REQUESTS": "0",
            "DUMP_META": "0",
        }
    )
    bench = subprocess.run(
        ["bash", "benchmark/tpu/bench_serving.sh"],
        env=bench_env,
        check=False,
    )

    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=30) as response:
        print(f"post-benchmark health: HTTP {response.status}", flush=True)

    if bench.returncode:
        raise SystemExit(f"B=64 benchmark failed with rc={bench.returncode}")

    request_batch(
        "exercise TP16 distributed TopK=50 at B=64",
        {"temperature": 0.7, "top_k": 50},
    )
    request_batch(
        "switch fixed candidate path to TopK=7 at B=64",
        {"temperature": 0.7, "top_k": 7},
    )
    request_batch(
        "switch to default full-vocabulary sampling at B=64",
        {"temperature": 0.7},
    )

    log = SERVER_LOG.read_text(errors="replace") if SERVER_LOG.exists() else ""
    failure_signatures = (
        "Detected recompile when torch.compile stance is 'fail_on_recompile'",
        "args[2]._tpu_real_b ==",
        "RuntimeUnexpectedCoreHalt",
        "UserFatal",
        "RuntimeBufferAllocationFailure",
    )
    found = [signature for signature in failure_signatures if signature in log]
    print(
        "failure scan: "
        + (f"FOUND {found}" if found else "clean (no recompile/halt signature)"),
        flush=True,
    )
    if found:
        raise SystemExit(f"serving regression reproduced: {found}")

    required_path_markers = (
        "path=candidate:1 candidate_width=64 logical_batch=64 physical_batch=64",
        "path=candidate:50 candidate_width=64 logical_batch=64 physical_batch=64",
        "path=candidate:7 candidate_width=64 logical_batch=64 physical_batch=64",
        "path=full_vocab:1073741824 candidate_width=64 logical_batch=64 "
        "physical_batch=64",
    )
    missing = [marker for marker in required_path_markers if marker not in log]
    if missing:
        raise SystemExit(f"missing post-forward path evidence: {missing}")
    print("path scan: candidate and full-vocabulary branches confirmed", flush=True)

    print(
        "PASS: B=1 prime and B=64 greedy/TopK/default transitions completed",
        flush=True,
    )


if __name__ == "__main__":
    main()
