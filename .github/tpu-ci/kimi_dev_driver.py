"""Run one full MMMU-Pro pass on the user's two dev hosts; persist artifacts."""

import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/sgl-workspace/sglang")
OUT = Path("/results")
BASE = "http://127.0.0.1:30000"
BUCKET = "sglang-torchtpu-gcs-shared"
PREFIX = "evals/kimi-dev-mmmu-20260923-r1"
RANK = int(os.environ["TPU_WORKER_ID"])
LAST_ARCHIVE = 0


def upload(path, checkpoint):
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        token = json.load(response)["access_token"]
    name = f"{PREFIX}/rank{RANK}/{checkpoint}/{path.name}"
    query = urllib.parse.urlencode(
        {"uploadType": "media", "name": name, "ifGenerationMatch": "0"}
    )
    request = urllib.request.Request(
        f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o?{query}",
        data=path.read_bytes(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        json.load(response)


def snapshot(stage, final=False):
    global LAST_ARCHIVE
    checkpoint = str(time.time_ns())
    rows = sum(
        sum(1 for _ in path.open())
        for path in (OUT / "eval").glob("sgl_eval_mmmu_pro_*/output-rs0.jsonl")
    )
    state = {
        "stage": stage,
        "rank": RANK,
        "responses": rows,
        "time": time.time(),
        "artifacts": f"gs://{BUCKET}/{PREFIX}/rank{RANK}/{checkpoint}/",
    }
    (OUT / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    print("KIMI_DEV_STATUS " + json.dumps(state), flush=True)
    upload(OUT / "status.json", checkpoint)
    for name in ("server.log", "eval.log"):
        path = OUT / name
        if not path.exists():
            continue
        with path.open("rb") as src:
            src.seek(max(0, path.stat().st_size - 150000))
            tail = OUT / (name + ".tail.gz")
            tail.write_bytes(gzip.compress(src.read()))
        upload(tail, checkpoint)
        if final:
            compressed = OUT / (name + ".gz")
            with path.open("rb") as src, gzip.open(compressed, "wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            upload(compressed, checkpoint)
    if (
        RANK == 0
        and (OUT / "eval").exists()
        and (final or time.monotonic() - LAST_ARCHIVE >= 300)
    ):
        archive = OUT / "results.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(OUT / "eval", arcname="eval")
        upload(archive, checkpoint)
        LAST_ARCHIVE = time.monotonic()


def safe_snapshot(stage):
    try:
        snapshot(stage)
    except Exception as error:
        print(f"Artifact checkpoint failed: {type(error).__name__}", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    os.chdir(ROOT)
    peers = os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",")
    if len(peers) != 2 or RANK not in (0, 1):
        raise RuntimeError(f"Expected two TPU hosts; rank={RANK}, peers={peers}")
    gmm = ROOT / "python/sglang/srt/layers/moe/pallas/gmm_v2.py"
    actual = hashlib.sha256(gmm.read_bytes()).hexdigest()
    if actual != os.environ["EXPECTED_GMM_SHA256"]:
        raise RuntimeError("Container GMM does not match the validated pinned source")
    print(f"KIMI_DEV_START rank={RANK} peers={peers} gmm_sha256={actual}", flush=True)
    # Fail before loading the model if durable result storage is unavailable.
    snapshot("starting")
    env = {
        **os.environ,
        "HF_HOME": "/tmp/hf",
        "SGLANG_PLATFORM": "tpu",
        "SGLANG_IS_IN_CI": "true",
        "PYTHONUNBUFFERED": "1",
        "HOST": "127.0.0.1",
        "PORT": "30000",
        "ENABLE_COMPILE_CACHE": "0",
        "TORCH_TPU_TIER3_ROOT": "/dev/shm/torch_tpu_tier3",
        "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torchinductor",
        "RUN_DIR": "/results/launch",
    }
    stage = "server_startup"
    evaluator = None
    with (OUT / "server.log").open("w") as log:
        server = subprocess.Popen(
            ["bash", "benchmark/tpu/recipe/kimi_k26_mmmu.sh"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        if RANK == 1:
            while server.poll() is None:
                safe_snapshot("worker_serving")
                time.sleep(60)
            snapshot("worker_stopped", final=True)
            return server.returncode
        deadline = time.monotonic() + 3600
        next_snapshot = 0
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(
                    f"Server exited before readiness: {server.returncode}"
                )
            try:
                with urllib.request.urlopen(BASE + "/health_generate", timeout=10):
                    break
            except OSError:
                pass
            if time.monotonic() >= next_snapshot:
                safe_snapshot(stage)
                next_snapshot = time.monotonic() + 60
            time.sleep(10)
        else:
            raise TimeoutError("Server did not become ready within one hour")
        stage = "evaluating"
        safe_snapshot(stage)
        with (OUT / "eval.log").open("w") as log:
            evaluator = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "sglang.test.tpu.mmmu_full",
                    "--base-url",
                    BASE,
                    "--shard",
                    "0",
                    "--shards",
                    "1",
                    "--output",
                    str(OUT / "eval"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        while evaluator.poll() is None:
            if server.poll() is not None:
                raise RuntimeError(
                    f"Server exited during evaluation: {server.returncode}"
                )
            safe_snapshot(stage)
            time.sleep(60)
        if evaluator.returncode:
            raise RuntimeError(f"Evaluation failed: {evaluator.returncode}")
        metrics = list((OUT / "eval").glob("sgl_eval_mmmu_pro_*/metrics.json"))
        if len(metrics) != 1:
            raise RuntimeError("Expected one full-set metrics file")
        result = json.loads(metrics[0].read_text())
        if result["num_examples"] != 1730 or result["aggregate"]["error_rate"] != 0:
            raise RuntimeError("Incomplete evaluation or request errors")
        print("KIMI_DEV_RESULT " + json.dumps(result), flush=True)
        snapshot("complete", final=True)
        return 0
    except Exception:
        snapshot("failed_" + stage, final=True)
        print((OUT / "server.log").read_text(errors="replace")[-16000:], flush=True)
        if (OUT / "eval.log").exists():
            print((OUT / "eval.log").read_text(errors="replace")[-8000:], flush=True)
        raise
    finally:
        if evaluator is not None and evaluator.poll() is None:
            evaluator.terminate()
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main())
