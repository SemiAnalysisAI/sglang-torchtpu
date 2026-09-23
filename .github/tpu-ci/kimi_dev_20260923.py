"""Inspect the user's dev capacity before placing the authorized Kimi retry."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

NAME = "kimi-dev-mmmu-20260923-r1"
POOL = "tpu7x-2x2x2-qinghan"
IMAGE = "us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu@sha256:631fffcab00e5e8bff9ff2ab303dea3314109a3d3405633eee25df8aacd67531"


def read_kube(*args):
    return json.loads(
        subprocess.check_output(["kubectl", *args, "-o", "json"], timeout=60)
    )


def main():
    nodes = read_kube("get", "nodes")["items"]
    dev_nodes = {}
    for node in nodes:
        labels = node["metadata"]["labels"]
        if "qinghan" in labels.get("cloud.google.com/gke-nodepool", ""):
            name = node["metadata"]["name"]
            dev_nodes[name] = node
            print(
                "DEV_NODE "
                + json.dumps(
                    {
                        "name": name,
                        "labels": labels,
                        "allocatable": node["status"].get("allocatable"),
                        "conditions": node["status"].get("conditions"),
                        "unschedulable": node["spec"].get("unschedulable", False),
                    }
                ),
                flush=True,
            )
    occupied = []
    for pod in read_kube("-n", "tpu-ci", "get", "pods")["items"]:
        if pod["spec"].get("nodeName") not in dev_nodes:
            continue
        if pod["status"].get("phase") in ("Succeeded", "Failed"):
            continue
        print(
            "DEV_POD "
            + json.dumps(
                {
                    "name": pod["metadata"]["name"],
                    "namespace": pod["metadata"]["namespace"],
                    "node": pod["spec"].get("nodeName"),
                    "phase": pod["status"].get("phase"),
                    "labels": pod["metadata"].get("labels"),
                    "resources": [
                        c.get("resources", {}) for c in pod["spec"]["containers"]
                    ],
                }
            ),
            flush=True,
        )
        if any(
            int(c.get("resources", {}).get("limits", {}).get("google.com/tpu", 0))
            for c in pod["spec"]["containers"]
        ):
            occupied.append(pod["metadata"]["name"])
    if occupied:
        raise SystemExit(f"Dev TPUs occupied; leaving those jobs untouched: {occupied}")
    if len(dev_nodes) != 2:
        raise SystemExit(
            f"Expected one two-host dev slice; found {len(dev_nodes)} hosts"
        )
    for node in dev_nodes.values():
        if (
            node["metadata"]["labels"].get("cloud.google.com/gke-nodepool") != POOL
            or node["spec"].get("unschedulable", False)
            or not any(
                c["type"] == "Ready" and c["status"] == "True"
                for c in node["status"]["conditions"]
            )
        ):
            raise SystemExit("Dev slice is not ready")
    command = [
        sys.executable,
        ".github/tpu-ci/render_jobset.py",
        "--image",
        IMAGE,
        "--hosts",
        "2",
        "--lane",
        "nightly",
        "--exec",
        "python3 -u /campaign/driver.py",
        "--weights-pvc",
        "model-weights-rox",
        "--service-account",
        "perf-metrics-writer",
        "--name",
        NAME,
    ]
    job = yaml.safe_load(subprocess.check_output(command, text=True))
    # The user explicitly moved this run to their dev slice, outside CI queues.
    # Full TPU requests and preemptionPolicy=Never protect existing workloads.
    job["metadata"] = {
        "name": NAME,
        "namespace": "tpu-ci",
        "labels": {"experiment": NAME, "owner": "qinghan-kimi-eval"},
    }
    job["spec"]["suspend"] = False
    job["spec"]["ttlSecondsAfterFinished"] = 604800
    child = job["spec"]["replicatedJobs"][0]["template"]["spec"]
    child["activeDeadlineSeconds"] = 86400
    pod = child["template"]["spec"]
    pod.pop("affinity", None)
    pod["nodeSelector"]["cloud.google.com/gke-nodepool"] = POOL
    pod["preemptionPolicy"] = "Never"
    for volume in pod["volumes"]:
        if volume["name"] == "dshm":
            volume["emptyDir"]["sizeLimit"] = "256Gi"
    container = pod["containers"][0]
    driver = Path(".github/tpu-ci/kimi_dev_driver.py").read_text()
    container["args"] = [
        "set -euo pipefail\ncd /sgl-workspace/sglang\n"
        "cat > /tmp/kimi_dev_driver.py <<'KIMI_DEV_DRIVER_PY'\n"
        + driver
        + "\nKIMI_DEV_DRIVER_PY\n"
        "exec python3 -u /tmp/kimi_dev_driver.py\n"
    ]
    source = Path("python/sglang/srt/layers/moe/pallas/gmm_v2.py")
    container["env"].append(
        {
            "name": "EXPECTED_GMM_SHA256",
            "value": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    )
    rendered = Path("/tmp/kimi-dev-jobset.yaml")
    rendered.write_text(yaml.safe_dump(job, sort_keys=False))
    # Use the CI account's existing JobSet permissions; no ConfigMap is needed.
    subprocess.run(
        ["kubectl", "create", "--dry-run=server", "-f", str(rendered)], check=True
    )
    subprocess.run(["kubectl", "create", "-f", str(rendered)], check=True)
    summary = {
        "jobset": NAME,
        "nodes": sorted(dev_nodes),
        "image": IMAGE,
        "TP": 16,
        "EP": 16,
        "DP": 1,
        "questions": 1730,
        "passes": 1,
        "concurrency": 128,
        "generation_cap": 98304,
        "artifacts": "gs://sglang-torchtpu-gcs-shared/evals/" + NAME,
    }
    print("KIMI_DEV_SUBMITTED " + json.dumps(summary), flush=True)
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
        "Submitted persistent Kimi MMMU-Pro evaluation on the user's dev slice.\n\n"
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )


if __name__ == "__main__":
    main()
