"""Exercise both static logits modes in one TP16 compiled model graph."""

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


def main() -> None:
    print("== B=1 greedy, full-vocabulary graph ==", flush=True)
    request_json(
        {
            "text": "Hello",
            "sampling_params": {"temperature": 0, "max_new_tokens": 2},
        }
    )

    print("== B=64 greedy, fixed-candidate graph ==", flush=True)
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
    check_health("post-greedy health")

    for top_k in (7, 50, 64):
        print(f"== B=64 TopK-{top_k}, same fixed-candidate graph ==", flush=True)
        response = request_json(
            {
                "text": ["Hello"] * 64,
                "sampling_params": {
                    "temperature": 1,
                    "top_k": top_k,
                    "max_new_tokens": 2,
                },
            }
        )
        if not isinstance(response, list) or len(response) != 64:
            raise RuntimeError(
                f"TopK-{top_k} returned {type(response).__name__}, expected 64 rows"
            )

    print(
        "== B=64 unconstrained sampling, prewarmed full-vocabulary graph ==", flush=True
    )
    response = request_json(
        {
            "text": ["Hello"] * 64,
            "sampling_params": {"temperature": 1, "max_new_tokens": 2},
        }
    )
    if not isinstance(response, list) or len(response) != 64:
        raise RuntimeError(
            f"full-vocabulary sampling returned {type(response).__name__}, expected 64 rows"
        )
    check_health("final health")

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
        "PASS: unified TP16 graph served greedy, TopK-7/50/64, and full-vocab modes",
        flush=True,
    )


if __name__ == "__main__":
    main()
