"""Temporary retry gate: wait for two free nightly slices before building."""

import json
import os
import subprocess
import time
from pathlib import Path


def read_json(command):
    return json.loads(subprocess.check_output(command, text=True))


def available_slices(queue):
    reservations = {
        flavor["name"]: int(resource["total"])
        for flavor in queue.get("status", {}).get("flavorsReservation", [])
        for resource in flavor["resources"]
        if resource["name"] == "google.com/tpu"
    }
    return sum(
        max(int(resource["nominalQuota"]) - reservations.get(flavor["name"], 0), 0) // 8
        for group in queue["spec"]["resourceGroups"]
        for flavor in group["flavors"]
        for resource in flavor["resources"]
        if resource["name"] == "google.com/tpu"
    )


def refresh_token():
    # The auth action's external-account credentials can mint a fresh token;
    # the static token written by get-gke-credentials expires after one hour.
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()
    config = read_json(["kubectl", "config", "view", "--raw", "-o", "json"])
    context = next(
        item["context"]
        for item in config["contexts"]
        if item["name"] == config["current-context"]
    )
    user = next(item for item in config["users"] if item["name"] == context["user"])
    user["user"] = {"token": token}
    path = Path(os.environ["KUBECONFIG"])
    path.write_text(json.dumps(config))
    path.chmod(0o600)


def main():
    deadline = time.monotonic() + 330 * 60
    refresh_at = 0
    while time.monotonic() < deadline:
        if time.monotonic() >= refresh_at:
            refresh_token()
            refresh_at = time.monotonic() + 40 * 60
        queue = read_json(
            ["kubectl", "get", "clusterqueue", "tpu-nightly-2h", "-o", "json"]
        )
        free = available_slices(queue)
        pending = queue.get("status", {}).get("pendingWorkloads", 0)
        print(
            f"Nightly 2x2x2: {free} free slices, {pending} queued workloads", flush=True
        )
        if free >= 2 and pending == 0:
            return
        time.sleep(60)
    raise SystemExit("No two free nightly slices within the retry wait budget")


if __name__ == "__main__":
    main()
