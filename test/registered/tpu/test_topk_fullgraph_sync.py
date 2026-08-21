"""Exercise both intact, precompiled logits contracts on TP16."""

import json
import os
import subprocess
import time
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
FREEZE_SIGNATURE = "froze torch.compile via set_stance('fail_on_recompile')"
RECOMPILE_SIGNATURES = (
    "Detected recompile when torch.compile stance is 'fail_on_recompile'",
    "new XLA program(s) compiled at serving time",
)
HALT_SIGNATURES = (
    "RuntimeUnexpectedCoreHalt",
    "SparseCoreSequencer",
)


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


def read_server_log() -> str:
    return SERVER_LOG.read_text(errors="replace") if SERVER_LOG.exists() else ""


def wait_for_log(signature: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if signature in read_server_log():
            return
        time.sleep(0.5)
    raise RuntimeError(f"server log did not contain {signature!r} within {timeout}s")


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
    # Startup intentionally builds every configured executable. Only messages
    # emitted after this boundary can represent serving-time recompilation.
    initial_log = read_server_log()
    serving_log_offset = len(initial_log)
    if FREEZE_SIGNATURE in initial_log:
        print("Dynamo was frozen by server warmup before the B=1 request", flush=True)

    print("== Prime Dynamo with the reported B=1 greedy request ==", flush=True)
    request_json(
        {
            "text": "Hello",
            "sampling_params": {"temperature": 0, "max_new_tokens": 2},
        }
    )
    check_health("post-B=1-prime health")
    wait_for_log(FREEZE_SIGNATURE)

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
            # The literal user command is not run under SGLANG_IS_IN_CI; avoid
            # the benchmark's CI-only /flush_cache request here as well.
            "SGLANG_IS_IN_CI": "false",
        }
    )
    subprocess.run(
        ["bash", "benchmark/tpu/bench_serving.sh"], env=bench_env, check=True
    )
    check_health("post-B=64-greedy-trigger health")

    # A single list request makes all 64 greedy rows visible to the scheduler at
    # once, independently of client arrival jitter in bench_serving.
    request_batch(
        "B=64 greedy/full-vocabulary",
        64,
        {"temperature": 0, "max_new_tokens": 8},
    )

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

    log = read_server_log()
    serving_log = log[serving_log_offset:]
    found = [
        signature for signature in RECOMPILE_SIGNATURES if signature in serving_log
    ]
    found.extend(signature for signature in HALT_SIGNATURES if signature in log)
    if found:
        raise RuntimeError(f"server log contains failure signatures: {found}")

    print(
        "PASS: exact B=1 -> B=64 greedy repro and both intact TP16 logits "
        "contracts completed without a recompile or SparseCore halt",
        flush=True,
    )


if __name__ == "__main__":
    main()
