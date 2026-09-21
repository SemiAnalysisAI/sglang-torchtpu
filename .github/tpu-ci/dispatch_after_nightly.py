"""Wait for a scheduled nightly, then dispatch branch accuracy without merging.

Runs inside GitHub Actions. Before the hosted-runner time limit, hand the wait
to another dispatch, carrying the same nightly ID and original cutoff timestamp.
"""

import datetime as dt
import json
import os
import subprocess
import time


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True).strip()


def select_nightly(runs, since):
    candidates = [run for run in runs if run["createdAt"] >= since]
    return min(candidates, key=lambda run: run["createdAt"]) if candidates else None


def dispatch(repo, ref, model, test_filter, wait_id="", since=""):
    args = [
        "workflow",
        "run",
        "accuracy-test-tpu.yml",
        "--repo",
        repo,
        "--ref",
        ref,
        "-f",
        f"ref={ref}",
        "-f",
        f"model={model}",
        "-f",
        f"test_filter={test_filter}",
        "-f",
        "update-dashboard=false",
        "-f",
        f"after-run-id={wait_id}",
        "-f",
        f"after-created-at={since}",
    ]
    result = gh(*args)
    print(result, flush=True)
    return result


def main():
    repo = os.environ["GH_REPO"]
    ref = os.environ["TARGET_REF"]
    model = os.environ["TARGET_MODEL"]
    test_filter = os.environ.get("TEST_FILTER", "")
    wait_id = os.environ["WAIT_RUN_ID"]
    since = os.environ.get("AFTER_CREATED_AT") or dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff = dt.datetime.fromisoformat(since.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("after-created-at must include a timezone")
    if wait_id != "next" and not wait_id.isdigit():
        raise ValueError("after-run-id must be a run ID or 'next'")
    started = time.monotonic()
    while True:
        if dt.datetime.now(dt.timezone.utc) - cutoff > dt.timedelta(days=2):
            raise TimeoutError("No completed selected nightly within two days")
        if wait_id == "next":
            runs = json.loads(
                gh(
                    "run",
                    "list",
                    "--repo",
                    repo,
                    "--workflow",
                    "nightly-test-tpu.yml",
                    "--event",
                    "schedule",
                    "--limit",
                    "30",
                    "--json",
                    "databaseId,status,createdAt",
                )
            )
            run = select_nightly(runs, since)
            if run:
                wait_id = str(run["databaseId"])
                print(f"Selected nightly {wait_id}", flush=True)
        if wait_id != "next":
            run = json.loads(gh("api", f"repos/{repo}/actions/runs/{wait_id}"))
            if (
                run["event"] != "schedule"
                or run["path"] != ".github/workflows/nightly-test-tpu.yml"
            ):
                raise ValueError("Selected run is not a scheduled TPU nightly")
            if run["status"] == "completed":
                print(
                    f"Nightly {wait_id} completed ({run['conclusion']}); dispatching {ref}",
                    flush=True,
                )
                result = dispatch(repo, ref, model, test_filter)
                with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
                    summary.write(
                        f"Nightly [{wait_id}](https://github.com/{repo}/actions/runs/{wait_id}) finished. Accuracy dispatch: {result}\n"
                    )
                return
        if time.monotonic() - started > 5 * 3600:
            print(
                "Handing the wait to another workflow before runner timeout", flush=True
            )
            dispatch(repo, ref, model, test_filter, wait_id, since)
            return
        print(f"Waiting for {wait_id}; cutoff={since}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
