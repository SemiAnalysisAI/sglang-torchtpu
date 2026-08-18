"""Run matched GPQA and AIME25 evaluations on Qwen3-Coder-480B."""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "/weights/Qwen3-Coder-480B-A35B-Instruct-FP8"
BASE_URL = "http://127.0.0.1:30000"
SERVER_LOG = Path("/tmp/qwen3-coder-480b-server.log")


def _wait_for_server(process: subprocess.Popen, timeout: int = 1800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode}\n"
                f"{SERVER_LOG.read_text(errors='replace')[-20000:]}"
            )
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"server was not healthy after {timeout}s")


def _run_eval(backend: str, eval_name: str) -> dict:
    output_dir = Path(f"/tmp/qwen480-rpa-eval/{backend}/{eval_name}")
    env = os.environ | {
        "MODEL": MODEL,
        "BASE_URL": BASE_URL,
        "EVAL_NAME": eval_name,
        "NUM_THREADS": "8",
        "OUTPUT_DIR": str(output_dir),
        "HEALTH_TIMEOUT": "30",
        "EXTRA_ARGS": "--max-tokens 4096 --temperature 0 --thinking-mode qwen-3",
    }
    subprocess.run(["bash", "benchmark/tpu/run_eval.sh"], env=env, check=True)
    result_files = list(output_dir.glob("*.json"))
    if len(result_files) != 1:
        raise RuntimeError(
            f"expected one result JSON in {output_dir}, found {result_files}"
        )
    return json.loads(result_files[0].read_text())


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=("pallas_tpu_rpa_v3", "pallas_tpu_rpa_v4"),
    )
    args = parser.parse_args()

    server_env = os.environ | {
        "MODEL": MODEL,
        "TP_SIZE": "8",
        "EP_SIZE": "8",
        "PAGE_SIZE": "128",
        "CONTEXT_LEN": "8192",
        "CHUNKED_PREFILL_SIZE": "4096",
        "MAX_PREFILL_TOKENS": "4096",
        "MAX_TOTAL_TOKENS": "65536",
        "MEM_FRACTION_STATIC": "0.8",
        "MAX_RUNNING": "8",
        "TORCH_COMPILE_MAX_BS": "8",
        "TPU_COMPILE_TOKENS": "64 128 256 512 1024 2048 4096",
        "QUANTIZATION": "fp8",
        "KV_CACHE_DTYPE": "fp8_e4m3",
        "ATTENTION_BACKEND": args.backend,
        "EXTRA_ARGS": (
            "--tpu-fp8-runtime channelwise --sampling-defaults openai "
            "--enable-fused-qk-norm-rope --reasoning-parser qwen3"
        ),
    }
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SERVER_LOG.open("w") as log_file:
        process = subprocess.Popen(
            ["bash", "benchmark/tpu/launch_server.sh"],
            env=server_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(process)
            results = {
                eval_name: _run_eval(args.backend, eval_name)
                for eval_name in ("gpqa", "aime25")
            }
        except Exception:
            print(SERVER_LOG.read_text(errors="replace")[-20000:])
            raise
        finally:
            _stop_server(process)

    print(
        f"RPA_EVAL_RESULTS backend={args.backend} {json.dumps(results, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
