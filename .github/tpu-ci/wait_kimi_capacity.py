"""Temporary retry gate: wait for two free nightly slices before building."""

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
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


def request_json(url, *, headers=None, data=None):
    headers = dict(headers or {})
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode()
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def refresh_token():
    # Match _tpu-test.yml: exchange a fresh GitHub OIDC token. gcloud may
    # return the same cached access token, so calling it early cannot renew
    # the one-hour kubeconfig token reliably.
    provider = os.environ["WIF_PROVIDER"]
    audience = "https://iam.googleapis.com/" + provider
    oidc = request_json(
        os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
        + "&"
        + urllib.parse.urlencode({"audience": audience}),
        headers={
            "Authorization": "bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
        },
    )["value"]
    sts = request_json(
        "https://sts.googleapis.com/v1/token",
        data={
            "audience": "//iam.googleapis.com/" + provider,
            "grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
            "requestedTokenType": "urn:ietf:params:oauth:token-type:access_token",
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "subjectTokenType": "urn:ietf:params:oauth:token-type:jwt",
            "subjectToken": oidc,
        },
    )["access_token"]
    credentials = request_json(
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
        + os.environ["CI_SA"]
        + ":generateAccessToken",
        headers={"Authorization": "Bearer " + sts},
        data={
            "scope": [
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/userinfo.email",
            ],
            "lifetime": "3600s",
        },
    )
    token = credentials["accessToken"]
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
    print(
        "Renewed cluster credentials; expires " + credentials["expireTime"], flush=True
    )


def main():
    deadline = time.monotonic() + 330 * 60
    refresh_at = 0
    failures = 0
    while time.monotonic() < deadline:
        try:
            if time.monotonic() >= refresh_at:
                refresh_token()
                refresh_at = time.monotonic() + 40 * 60
            queue = read_json(
                ["kubectl", "get", "clusterqueue", "tpu-nightly-2h", "-o", "json"]
            )
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
            failures += 1
            if failures >= 3:
                raise SystemExit("Cluster access failed three times; aborting wait")
            print(
                f"Cluster check failed ({type(error).__name__}); renewing credentials",
                flush=True,
            )
            refresh_at = 0
            time.sleep(30)
            continue
        failures = 0
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
