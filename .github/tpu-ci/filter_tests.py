#!/usr/bin/env python3
"""Narrow a discover_tests.py matrix to the tests matching a path substring.

Reads the JSON row list on stdin and writes the kept rows to stdout, so it
composes as a pipe stage in the workflow's discover step.
"""

import argparse
import json
import sys


def filter_rows(rows: list[dict], needle: str) -> list[dict]:
    """Rows whose `file` contains `needle`.

    Raises when nothing matches: an empty matrix makes GitHub skip the job, and
    a skipped job reads as a pass, so a typo'd filter would otherwise produce a
    green run that tested nothing.
    """
    kept = [row for row in rows if needle in row["file"]]
    if not kept:
        raise ValueError(f"filter {needle!r} matched none of the {len(rows)} tests")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", required=True, help="substring of a test file path")
    args = parser.parse_args()
    try:
        kept = filter_rows(json.load(sys.stdin), args.filter)
    except ValueError as exc:
        sys.exit(str(exc))
    json.dump(kept, sys.stdout)


if __name__ == "__main__":
    main()
