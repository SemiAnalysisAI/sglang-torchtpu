#!/usr/bin/env python3
"""Plan the shards of a single-host TPU suite once, for a GitHub matrix.

The shard count is fixed here and handed to every leg as a job output, so a
re-run of a failed leg reuses it: auto_partition is a pure function of the
commit's registrations and the count, and the ref is pinned, hence a re-run
leg runs exactly the files it ran before. Never derive the count inside a leg.

Stdlib only (see ci_register_loader.py); render_jobset.py imports suite_tests.

Usage: plan_partitions.py --suite <name> [--target-seconds S] [--max-partitions N]
Prints {"include": [{"id", "size", "label"}, ...]}; writes the per-shard file
list to $GITHUB_STEP_SUMMARY when set.
"""

import argparse
import glob
import json
import math
import os

from ci_register_loader import ci_register as _ci_register


def suite_tests(suite, tests_dir="test/registered", *, nightly=False, test_file=""):
    """Enabled TPU tests of a suite, filtered the way run_suite.py filters them."""
    files = [
        path
        for path in glob.glob(os.path.join(tests_dir, "**", "*.py"), recursive=True)
        if os.path.basename(path) not in ("conftest.py", "__init__.py")
    ]
    tests = [
        test
        for test in _ci_register.collect_tests(files, sanity_check=False)
        if test.backend == _ci_register.HWBackend.TPU
        and test.effective_suite == suite
        and test.nightly == nightly
        and test.disabled is None
    ]
    if test_file:
        requested = os.path.normpath(os.path.abspath(test_file))
        tests = [
            test
            for test in tests
            if os.path.normpath(os.path.abspath(test.filename)) == requested
        ]
    return tests


def partition_count(tests, target_seconds, max_partitions):
    total = sum(test.est_time for test in tests)
    return max(1, min(max_partitions, math.ceil(total / target_seconds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--tests-dir", default="test/registered")
    # Registered est_time per shard; the poll and admission budgets scale with it.
    ap.add_argument("--target-seconds", type=float, default=3600)
    # Upper bound on shards; more legs than slices only queue in Kueue.
    ap.add_argument("--max-partitions", type=int, default=4)
    a = ap.parse_args()
    if a.target_seconds <= 0 or a.max_partitions <= 0:
        ap.error("--target-seconds and --max-partitions must be positive")

    tests = suite_tests(a.suite, a.tests_dir, nightly=a.nightly)
    if not tests:
        ap.error(f"suite {a.suite!r} has no enabled TPU tests")
    size = partition_count(tests, a.target_seconds, a.max_partitions)
    shards = [_ci_register.auto_partition(tests, rank, size) for rank in range(size)]

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"## {a.suite}: {size} shard(s)\n\n")
            for rank, shard in enumerate(shards):
                est = sum(test.est_time for test in shard)
                f.write(f"<details><summary>{rank + 1}/{size} ")
                f.write(f"({len(shard)} files, est {est:.0f}s)</summary>\n\n")
                for test in sorted(shard, key=lambda t: t.filename):
                    f.write(f"- `{test.filename}`\n")
                f.write("\n</details>\n\n")

    print(
        json.dumps(
            {
                "include": [
                    {"id": rank, "size": size, "label": f"{rank + 1}-of-{size}"}
                    for rank in range(size)
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
