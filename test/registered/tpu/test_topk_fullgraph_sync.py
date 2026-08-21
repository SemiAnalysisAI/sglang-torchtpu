"""Exercise both intact, precompiled logits contracts on TP16."""

import json
import os
import subprocess
import urllib.request
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=1800, suite="topk-fullgraph-validation", nightly=True)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/topk_fullgraph_sync_mh.sh",
    "weights_pvc": "model-weights-rox",
}

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("MODEL", "/weights/Qwen3-Coder-480B-A35B-Instruct-FP8")
SERVER_LOG = Path(os.environ.get("SERVER_LOG", "/tmp/server.log"))


def request_json(payload: dict, timeout: int = 1200):
    request = urllib.request.Request(
        f"{BASE_URL}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_health(label: str) -> None:
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=30) as response:
        print(f"{label}: HTTP {response.status}", flush=True)


def request_batch(label: str, batch_size: int, sampling_params: dict) -> None:
    print(f"== {label} ==", flush=True)
    response = request_json(
        {
            "text": ["Hello"] * batch_size,
            "sampling_params": sampling_params,
        }
    )
    if not isinstance(response, list) or len(response) != batch_size:
        raise RuntimeError(
            f"{label} returned {type(response).__name__}, "
            f"expected {batch_size} rows"
        )
    check_health(f"post-{label} health")


def main() -> None:
    print("== Prime Dynamo with the reported B=1 greedy request ==", flush=True)
    request_json(
        {
            "text": "Hello",
            "sampling_params": {"temperature": 0, "max_new_tokens": 2},
        }
    )
    check_health("post-B=1-prime health")

    print("== Run the reported B=64 greedy trigger ==", flush=True)
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
    subprocess.run(
        ["bash", "benchmark/tpu/bench_serving.sh"], env=bench_env, check=True
    )
    check_health("post-B=64-greedy-trigger health")

    # Exercise every candidate boundary and vary logical batch size within the
    # same B=64 physical bucket after the exact regression sequence passes.
    for batch_size, top_k in ((33, 2), (63, 7), (64, 50), (64, 64)):
        request_batch(
            f"B={batch_size} TopK-{top_k}/fixed-candidates",
            batch_size,
            {"temperature": 1, "top_k": top_k, "max_new_tokens": 8},
        )
    request_batch(
        "B=33 greedy/full-vocabulary",
        33,
        {"temperature": 0, "max_new_tokens": 8},
    )
    request_batch(
        "B=63 unrestricted/full-vocabulary",
        63,
        {"temperature": 1, "max_new_tokens": 8},
    )

    # Cross the contract boundary again after both variants have executed.
    request_batch(
        "B=64 TopK-50 replay/fixed-candidates",
        64,
        {"temperature": 1, "top_k": 50, "max_new_tokens": 8},
    )
    request_batch(
        "B=64 greedy replay/full-vocabulary",
        64,
        {"temperature": 0, "max_new_tokens": 8},
    )

    log = SERVER_LOG.read_text(errors="replace") if SERVER_LOG.exists() else ""
    failure_signatures = (
        "Detected recompile when torch.compile stance is 'fail_on_recompile'",
        "RuntimeUnexpectedCoreHalt",
        "SparseCoreSequencer",
    )
    found = [signature for signature in failure_signatures if signature in log]
    if found:
        raise RuntimeError(f"server log contains failure signatures: {found}")

    print(
        "PASS: exact B=1 -> B=64 greedy repro and both intact TP16 logits "
        "contracts completed without a recompile or SparseCore halt",
        flush=True,
    )


if __name__ == "__main__":
    main()
