"""One-shot Kimi dispatch after the user's current nightly has finished."""

import json
import os
import subprocess
import time
from pathlib import Path

REPO = "sgl-project/sglang-torchtpu"
NIGHTLY_RUN = 35815394025
PREVIOUS_RETRY = 35840663944
NIGHTLY_QUIET_SECONDS = 5 * 60
TARGET_BRANCH = "test/kimi-mmmu-vmem-20260923"
TARGET_SHA = "8fe64967eff0bcbfa6b06a237d93579d4a05cfc8"


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", *args], text=True, timeout=60))


def active_nightly_runs(runs):
    return [run["databaseId"] for run in runs if run["status"] != "completed"]


def main():
    deadline = time.monotonic() + 330 * 60
    quiet_since = None
    while time.monotonic() < deadline:
        try:
            nightly = gh_json("api", f"repos/{REPO}/actions/runs/{NIGHTLY_RUN}")
            runs = gh_json(
                "run",
                "list",
                "--repo",
                REPO,
                "--workflow",
                "nightly-test-tpu.yml",
                "--limit",
                "30",
                "--json",
                "databaseId,status",
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            quiet_since = None
            print(
                f"Nightly status query failed ({type(error).__name__}); retrying",
                flush=True,
            )
            time.sleep(60)
            continue
        active = active_nightly_runs(runs)
        print(
            f"Nightly {NIGHTLY_RUN}: {nightly['status']}; active nightly runs: {active}",
            flush=True,
        )
        if nightly["status"] == "completed" and not active:
            # Allow time for a failed nightly to be rerun before taking TPU capacity.
            now = time.monotonic()
            if quiet_since is None:
                quiet_since = now
            if now - quiet_since >= NIGHTLY_QUIET_SECONDS:
                break
        else:
            quiet_since = None
        time.sleep(60)
    else:
        raise SystemExit("Nightly has not finished; no Kimi evaluation submitted")

    # Idempotent if this watcher is rerun after a successful dispatch.
    existing = gh_json(
        "run",
        "list",
        "--repo",
        REPO,
        "--workflow",
        "accuracy-test-tpu.yml",
        "--branch",
        TARGET_BRANCH,
        "--limit",
        "30",
        "--json",
        "databaseId,headSha",
    )
    for run in existing:
        if run["headSha"] == TARGET_SHA and run["databaseId"] > PREVIOUS_RETRY:
            print(f"Kimi retry already submitted: {run['databaseId']}", flush=True)
            return
    branch = gh_json("api", f"repos/{REPO}/git/ref/heads/{TARGET_BRANCH}")
    if branch["object"]["sha"] != TARGET_SHA:
        raise SystemExit(
            "Kimi test branch changed; refusing to dispatch unreviewed code"
        )
    output = subprocess.check_output(
        [
            "gh",
            "workflow",
            "run",
            "accuracy-test-tpu.yml",
            "--repo",
            REPO,
            "--ref",
            TARGET_BRANCH,
            "-f",
            f"ref={TARGET_SHA}",
            "-f",
            "model=kimi-k26",
            "-f",
            "update-dashboard=false",
        ],
        text=True,
        timeout=90,
    )
    print(output, flush=True)
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
        f"Nightly {NIGHTLY_RUN} finished ({nightly['conclusion']}).\n\n"
        f"Submitted Kimi-only MMMU-Pro evaluation at `{TARGET_SHA}`.\n\n{output}"
    )


if __name__ == "__main__":
    main()
