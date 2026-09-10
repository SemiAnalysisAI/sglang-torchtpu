#!/usr/bin/env python3
"""Narrow a discovery matrix to the rows matching a path substring.

Reads the JSON row list on stdin and writes the kept rows to stdout, so it
composes as a pipe stage in a workflow's discover step. Shape-agnostic: a
discover_tests.py row names its test in `file`, a discover_multihost.py launch
group in `files`.

Matching nothing is not an error here -- a filter naming a single-host test
legitimately matches no multi-host group. Only the workflow sees every matrix
it filtered, so it owns the rule that the filter must match somewhere; see
feature-validation-tpu.yml's `filter-matched` job.
"""

import argparse
import json
import sys


def _row_path(row: dict) -> str:
    """The row's test file: `file` on a test row, `files` on a launch group."""
    return row.get("file") or row.get("files", "")


def _filter_rows(rows: list[dict], needle: str) -> list[dict]:
    return [row for row in rows if needle in _row_path(row)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", required=True, help="substring of a test file path")
    args = parser.parse_args()
    json.dump(_filter_rows(json.load(sys.stdin), args.filter), sys.stdout)


if __name__ == "__main__":
    main()
