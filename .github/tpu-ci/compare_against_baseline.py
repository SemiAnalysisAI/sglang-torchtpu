#!/usr/bin/env python3
"""Compare current GitHub Actions run job-level pass/fail results against the baseline run from main."""

from __future__ import annotations

import argparse
import json
import os
import subprocess


def _gh_api(path: str, token: str | None = None, allow_escape: bool = False) -> bytes:
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    cmd = ["gh", "api", path]
    if allow_escape:
        cmd.append("--allow-escape-sequences")
    try:
        return subprocess.check_output(cmd, env=env, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        if b"unknown flag" in e.stderr and allow_escape:
            return subprocess.check_output(["gh", "api", path], env=env)
        if b"--allow-escape-sequences" in e.stderr:
            return subprocess.check_output(
                ["gh", "api", path, "--allow-escape-sequences"], env=env
            )
        raise


def generate_comparison(
    baseline_path: str,
    current_run_id: int | str,
    repo: str,
    token: str | None = None,
) -> str:
    with open(baseline_path, "r") as fp:
        baseline = json.load(fp)

    baseline_jobs = baseline.get("jobs", {})
    baseline_run_id = baseline.get("run_id")
    baseline_url = baseline.get(
        "run_url", f"https://github.com/{repo}/actions/runs/{baseline_run_id}"
    )
    baseline_sha = baseline.get("head_sha", "unknown")[:10]

    # Fetch current run jobs
    jobs_raw = _gh_api(
        f"repos/{repo}/actions/runs/{current_run_id}/jobs?per_page=100", token
    ).decode("utf-8")
    current_data = json.loads(jobs_raw)
    current_jobs_list = current_data.get("jobs", [])

    current_jobs = {}
    for j in current_jobs_list:
        name = j.get("name")
        conclusion = j.get("conclusion")
        if name and conclusion and conclusion != "skipped":
            # Skip self-reporting job if present
            if "report-baseline-diff" in name or "compare" in name.lower():
                continue
            current_jobs[name] = conclusion

    all_job_names = sorted(set(baseline_jobs.keys()) | set(current_jobs.keys()))

    regressions = []
    fixes = []
    consistent_pass = []
    consistent_fail = []
    only_current = []
    only_baseline = []

    for name in all_job_names:
        base_res = baseline_jobs.get(name)
        curr_res = current_jobs.get(name)

        if base_res and curr_res:
            if base_res == "success" and curr_res != "success":
                regressions.append((name, base_res, curr_res))
            elif base_res != "success" and curr_res == "success":
                fixes.append((name, base_res, curr_res))
            elif curr_res == "success":
                consistent_pass.append((name, base_res, curr_res))
            else:
                consistent_fail.append((name, base_res, curr_res))
        elif curr_res:
            only_current.append((name, curr_res))
        else:
            only_baseline.append((name, base_res))

    # Format Markdown summary
    lines = []
    lines.append("# TPU Test vs Nightly `main` Baseline Comparison")
    lines.append("")
    lines.append(
        f"- **Baseline Run**: [#{baseline_run_id} ({baseline_sha})]({baseline_url})"
    )
    lines.append(
        f"- **Current Run**: [#{current_run_id}](https://github.com/{repo}/actions/runs/{current_run_id})"
    )
    lines.append("")
    lines.append("## High-Level Summary")
    lines.append(f"- **Total Executed Jobs in Current Run**: `{len(current_jobs)}`")
    lines.append(
        f"- **Passed**: `{len([c for c in current_jobs.values() if c == 'success'])}`"
    )
    lines.append(
        f"- **Failed**: `{len([c for c in current_jobs.values() if c != 'success'])}`"
    )
    lines.append(
        f"- **Regressions (Passed in Baseline -> Failed in Current)**: `{len(regressions)}`"
    )
    lines.append(
        f"- **Fixes (Failed in Baseline -> Passed in Current)**: `{len(fixes)}`"
    )
    lines.append("")

    if regressions:
        lines.append("### 🚨 Regressions")
        lines.append("| Job Name | Baseline (`main`) | Current Run |")
        lines.append("| :--- | :---: | :---: |")
        for name, b_res, c_res in regressions:
            lines.append(f"| `{name}` | ✅ {b_res} | ❌ {c_res} |")
        lines.append("")

    if fixes:
        lines.append("### 🚀 Improvements / Fixes")
        lines.append("| Job Name | Baseline (`main`) | Current Run |")
        lines.append("| :--- | :---: | :---: |")
        for name, b_res, c_res in fixes:
            lines.append(f"| `{name}` | ❌ {b_res} | ✅ {c_res} |")
        lines.append("")

    lines.append("### Full Job Results Comparison")
    lines.append("| Job Name | Baseline (`main`) | Current Run | Status |")
    lines.append("| :--- | :---: | :---: | :---: |")

    for name, b_res, c_res in regressions:
        lines.append(f"| `{name}` | ✅ {b_res} | ❌ {c_res} | 🚨 REGRESSION |")
    for name, b_res, c_res in fixes:
        lines.append(f"| `{name}` | ❌ {b_res} | ✅ {c_res} | 🚀 FIXED |")
    for name, b_res, c_res in consistent_fail:
        lines.append(f"| `{name}` | ❌ {b_res} | ❌ {c_res} | ⚠️ PERSISTENT FAILURE |")
    for name, b_res, c_res in consistent_pass:
        lines.append(f"| `{name}` | ✅ {b_res} | ✅ {c_res} | ✅ PASSED |")
    for name, c_res in only_current:
        lines.append(
            f"| `{name}` | *(not in baseline)* | {'✅' if c_res == 'success' else '❌'} {c_res} | NEW |"
        )
    for name, b_res in only_baseline:
        lines.append(
            f"| `{name}` | {'✅' if b_res == 'success' else '❌'} {b_res} | *(not run)* | NOT RUN |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-json", required=True, help="Path to baseline JSON file"
    )
    parser.add_argument(
        "--current-run-id", required=True, help="Current GitHub Actions run ID"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang-torchtpu"),
        help="GitHub repository (owner/repo)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        help="GitHub API token",
    )
    parser.add_argument(
        "--output-markdown",
        default="",
        help="Path to save Markdown summary",
    )
    args = parser.parse_args()

    md = generate_comparison(
        args.baseline_json, args.current_run_id, args.repo, args.token
    )

    if args.output_markdown:
        with open(args.output_markdown, "w") as fp:
            fp.write(md)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fp:
            fp.write("\n" + md + "\n")

    print(md)


if __name__ == "__main__":
    main()
