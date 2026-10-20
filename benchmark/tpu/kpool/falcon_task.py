# SPDX-License-Identifier: Apache-2.0
"""Render a native Falcon task that starts from a pinned public Git revision."""

import argparse
import json
import re
import shlex


def manifest(revision, cluster, wheelhouse_bucket=None, wheelhouse_prefix=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full Git commit SHA")
    if bool(wheelhouse_bucket) != bool(wheelhouse_prefix):
        raise ValueError("Set both wheelhouse bucket and prefix, or neither")
    install = "python -m pip install -r benchmark/tpu/kpool/requirements.txt"
    if wheelhouse_bucket:
        install += " --no-index --find-links=/wheelhouse"
    script = f"""set -euo pipefail
ulimit -c 0
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export JAX_PLATFORMS=tpu
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
echo PHASE_BENCHMARK
timeout 1800 python benchmark/tpu/kpool/benchmark.py \\
  --output "${{ARTIFACT_LOCAL_DIR:?}}/rank-${{FALCON_RANK:-0}}/benchmark"
"""
    task = {
        "image": "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1",
        "replica": 1,
        "device_type": "v7x",
        "device_count": 8,
        "device_topo": "2x2x1",
        "command": "bash -c " + shlex.quote(script),
    }
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
    return {
        "name": f"kpool-perf-{revision[:12]}",
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
    args = parser.parse_args()
    print(
        json.dumps(
            manifest(
                args.revision,
                args.cluster,
                args.wheelhouse_bucket,
                args.wheelhouse_prefix,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
