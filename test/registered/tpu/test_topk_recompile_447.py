"""Exercise TP16 distributed TopK after a B=1 Dynamo prime."""

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

    log = SERVER_LOG.read_text(errors="replace") if SERVER_LOG.exists() else ""
    guard_signatures = (
        "Detected recompile when torch.compile stance is 'fail_on_recompile'",
        "args[2]._tpu_real_b ==",
    )
    found = [signature for signature in guard_signatures if signature in log]
    print(
        "guard scan: " + (f"FOUND {found}" if found else "clean (no recompile guard)"),
        flush=True,
    )

    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=30) as response:
        print(f"post-benchmark health: HTTP {response.status}", flush=True)

    if bench.returncode:
        raise SystemExit(f"B=64 benchmark failed with rc={bench.returncode}")
    if found:
        raise SystemExit(f"recompile regression reproduced: {found}")

    print(
        "PASS: B=1 prime followed by B=64 completed without a Dynamo recompile",
        flush=True,
    )


if __name__ == "__main__":
    main()
