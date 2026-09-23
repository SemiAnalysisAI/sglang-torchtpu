"""Inspect the user's dev capacity before placing the authorized Kimi retry."""

import json
import subprocess


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
    for pod in read_kube("get", "pods", "-A")["items"]:
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
    print("Inspection only; no workloads created or modified.", flush=True)


if __name__ == "__main__":
    main()
