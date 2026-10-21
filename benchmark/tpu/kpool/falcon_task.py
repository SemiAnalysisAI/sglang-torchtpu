# SPDX-License-Identifier: Apache-2.0
"""Render a native Falcon task that starts from a pinned public Git revision."""

import argparse
import json
import re
import shlex


def manifest(
    revision,
    cluster,
    wheelhouse_bucket=None,
    wheelhouse_prefix=None,
    mode="performance",
    hold_seconds=0,
    aot_cases=None,
    benchmark_cases=None,
):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full Git commit SHA")
    if bool(wheelhouse_bucket) != bool(wheelhouse_prefix):
        raise ValueError("Set both wheelhouse bucket and prefix, or neither")
    if (
        mode not in ("performance", "decode-performance", "aot", "profile-analysis", "device-profile")
        or not 0 <= hold_seconds <= 1800
    ):
        raise ValueError("Invalid task mode or artifact retrieval window")
    install = "python -m pip install -r benchmark/tpu/kpool/requirements.txt"
    if wheelhouse_bucket:
        install += " --no-index --find-links=/wheelhouse"
    workload = """echo PHASE_BENCHMARK
git fetch --depth=1 origin 985b4f8c9f2151cfae7c110156f7433892e1ee60
git fetch --depth=1 origin 61edd0e421a877158f7bfdfdfd2ae6133173cbfc
git fetch --depth=1 origin 880e8ace2f2f3f06cc66eb9a1235b6e6d7bf2335
compiler="${ARTIFACT_LOCAL_DIR:?}/rank-${FALCON_RANK:-0}/compiler"
mkdir -p "$compiler/llo" "$compiler/mosaic"
export LIBTPU_INIT_ARGS="--xla_jf_dump_to=$compiler/llo --xla_mosaic_dump_to=$compiler/mosaic --xla_jf_dump_hlo_text=true --xla_jf_dump_llo_text=true --xla_mosaic_enable_llo_source_annotations=true --xla_mosaic_enable_dump_debug_info=true --xla_jf_emit_annotations=true --xla_jf_collect_llo_stack_trace=true"
timeout 1800 python benchmark/tpu/kpool/benchmark.py \\
  --output "${ARTIFACT_LOCAL_DIR:?}/rank-${FALCON_RANK:-0}/benchmark"
"""
    if mode == "decode-performance":
        # Full LLO is collected by the separate AOT task. Keep compiler-dump
        # I/O out of this bounded same-device latency comparison.
        lines = workload.splitlines()
        lines = [
            line for line in lines if not line.startswith("export LIBTPU_INIT_ARGS=")
        ]
        workload = (
            "\n".join(lines)
            + " --cases decode --decode-pipeline --decode-query-lengths 1 6\n"
        )
        workload = workload.replace(
            "echo PHASE_BENCHMARK",
            "echo PHASE_BENCHMARK\ngit fetch --depth=1 origin de72e465f066024c2e678b63de7d4cd73212a404",
        )
    if benchmark_cases and mode != "decode-performance":
        workload = (
            workload.rstrip()
            + " --cases "
            + " ".join(shlex.quote(case) for case in benchmark_cases)
            + "\n"
        )
    if mode == "profile-analysis":
        workload = "bash benchmark/tpu/kpool/profile_analysis_task.sh\n"
    if mode == "aot":
        cases = ""
        if aot_cases:
            cases = " --cases " + " ".join(shlex.quote(case) for case in aot_cases)
        workload = f"""echo PHASE_AOT
git fetch --depth=1 origin 75a03373e46643a9373cf9bcd1ed7fdfd794385d
git fetch --depth=1 origin de72e465f066024c2e678b63de7d4cd73212a404
out="${{ARTIFACT_LOCAL_DIR:?}}/rank-${{FALCON_RANK:-0}}/aot"
python benchmark/tpu/kpool/check_isolation.py \\
  --baseline de72e465f066024c2e678b63de7d4cd73212a404 --output "$out/isolation.json"
timeout 3600 python benchmark/tpu/kpool/aot.py --output "$out"{cases} \\
  --archive /tmp/tpu_logs/kpool-aot.tar.gz
sleep {hold_seconds}
"""
    script = f"""set -euo pipefail
ulimit -c 0
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export JAX_PLATFORMS={"cpu" if mode in ("aot", "profile-analysis") else "tpu"}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
echo PHASE_FETCH_SOURCE
task_root=$(mktemp -d /tmp/tpu_logs/kpool.XXXXXX)
git init "$task_root/source"
git -C "$task_root/source" remote add origin https://github.com/sgl-project/sglang-torchtpu.git
git -C "$task_root/source" fetch --depth=1 origin {revision}
git -C "$task_root/source" checkout --detach FETCH_HEAD
cd "$task_root/source"
test "$(git rev-parse HEAD)" = {revision}
echo PHASE_INSTALL
python3 -m venv "$task_root/venv"
export PATH="$task_root/venv/bin:$PATH"
{install}
{workload}
"""
    task = {
        "image": "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1",
        "replica": 1,
        "device_type": "v7x",
        "device_count": 8,
        "device_topo": "2x2x1",
        "command": "bash -c " + shlex.quote(script),
    }
    if mode in ("aot", "profile-analysis"):
        task.update(
            device_type="cpu",
            device_count=1,
            resources={
                "cpu_request": "8",
                "cpu_limit": "8",
                "memory_request": "16Gi",
                "memory_limit": "16Gi",
            },
        )
        del task["device_topo"]
    if wheelhouse_bucket:
        task["mounts"] = [
            {
                "name": "wheelhouse",
                "type": "gcs",
                "bucket": wheelhouse_bucket,
                "prefix": wheelhouse_prefix,
                "mount_path": "/wheelhouse",
                "read_only": True,
                "mount_options": ["implicit-dirs"],
            }
        ]
    if mode == "profile-analysis":
        task.setdefault("mounts", []).append({
            "name": "profiles", "type": "gcs",
            "bucket": "tpu-for-training-falcon-logs", "prefix": "experiments",
            "mount_path": "/profiles", "read_only": True,
            "mount_options": ["implicit-dirs"],
        })
    return {
        "name": f"kpool-{mode}-{revision[:12]}",
        "cluster_name": cluster,
        "exp_type": "BENCHMARK",
        "artifact_type": "GCS",
        "config": json.dumps(
            {"source_revision": revision, "accuracy": "deferred_to_checkpoint"}
        ),
        "role_to_task_spec": {"worker": task},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--wheelhouse-bucket")
    parser.add_argument("--wheelhouse-prefix")
    parser.add_argument(
        "--mode",
        choices=("performance", "decode-performance", "aot", "profile-analysis", "device-profile"),
        default="performance",
    )
    parser.add_argument("--hold-seconds", type=int, default=0)
    parser.add_argument("--aot-cases", nargs="+")
    parser.add_argument("--benchmark-cases", nargs="+", choices=("decode", "prefill"))
    args = parser.parse_args()
    print(
        json.dumps(
            manifest(
                args.revision,
                args.cluster,
                args.wheelhouse_bucket,
                args.wheelhouse_prefix,
                args.mode,
                args.hold_seconds,
                args.aot_cases,
                args.benchmark_cases,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
