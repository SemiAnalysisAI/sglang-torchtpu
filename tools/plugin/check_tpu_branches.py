# SPDX-License-Identifier: Apache-2.0
"""Ratchet on TPU debt in SGLang's shared core tree.

Two numbers, both of which may only go down:

1. **TPU branches in shared core.** References to ``is_tpu`` / ``_is_tpu`` in
   ``python/sglang/srt``, excluding TPU-owned directories. Each one is a place
   core has to know TPU exists, and each is work for waves 1-3.

2. **TPU modules outside the plugin.** Python files under ``python/sglang``
   whose path names TPU but which do not live in the in-tree TPU package. Each
   is a file wave 4 has to relocate.

Neither is forbidden -- the repo has 300-odd of the first today. What is
forbidden is *adding* to either without taking the decision deliberately. New
TPU code belongs in ``hardware_backend/tpu/`` and reaches core through a
platform method or a registration, never a new branch (see that package's README
and the wave-0 handoff).

Usage
-----
    python tools/plugin/check_tpu_branches.py              # check against baseline
    python tools/plugin/check_tpu_branches.py --write-baseline
    python tools/plugin/check_tpu_branches.py --files a.py b.py   # pre-commit

Counts are tracked **per file**, not as one global total, so a removal in one
module cannot pay for an addition in another.

When a count legitimately rises -- a TPU fix that genuinely cannot be expressed
through a seam -- rerun with ``--write-baseline`` and say why in the PR. The
baseline is reviewed like code; that review is the control, not this script.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Iterator, NamedTuple, Optional

#: Paths are relative to the repo root throughout, with forward slashes.
BASELINE_PATH = os.path.join("tools", "plugin", "tpu_debt_baseline.json")

#: Where TPU branches are counted.
BRANCH_SCAN_ROOT = "python/sglang/srt"

#: Where TPU-named modules are looked for.
MODULE_SCAN_ROOT = "python/sglang"

#: The in-tree TPU package: TPU-owned already, and the subject of wave 4 as a
#: whole rather than file by file.
IN_TREE_TPU_PACKAGE = "python/sglang/srt/hardware_backend/tpu/"

#: Platform definitions. `is_tpu` is *defined* here; counting its definition as
#: debt would make the ratchet unfixable.
PLATFORM_PACKAGE = "python/sglang/srt/platforms/"

#: Names whose appearance in shared core is the debt being tracked.
_TPU_NAMES = frozenset({"is_tpu", "_is_tpu"})

#: "tpu" as a path component or a snake_case word, not the "tpu" inside "output".
_TPU_IN_PATH = re.compile(r"(?:^|[/_.-])tpu(?:[/_.-]|$)", re.IGNORECASE)


class Debt(NamedTuple):
    """The two tracked numbers, keyed by file.

    A NamedTuple rather than the repo-standard ``msgspec.Struct``: pre-commit
    runs this script with ``language: system``, so it gets the bare interpreter
    on the runner and may import nothing outside the standard library. Treat the
    defaults as read-only -- they are shared across instances, as always with
    NamedTuple, and nothing here mutates a field in place.
    """

    #: path -> count of is_tpu / _is_tpu references in shared core
    branches: dict[str, int] = {}
    #: paths of TPU-named modules outside the in-tree TPU package
    modules: list[str] = []


def _rel(path: str, repo_root: str) -> str:
    return os.path.relpath(path, repo_root).replace(os.sep, "/")


def names_tpu(rel_path: str) -> bool:
    """True when the path itself declares the file to be TPU code."""
    return bool(_TPU_IN_PATH.search(rel_path))


def _is_tpu_owned(rel_path: str) -> bool:
    return (
        rel_path.startswith(IN_TREE_TPU_PACKAGE)
        or rel_path.startswith(PLATFORM_PACKAGE)
        or names_tpu(rel_path)
    )


def _iter_python_files(root: str) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def count_tpu_branches(path: str) -> int:
    """References to ``is_tpu`` / ``_is_tpu``, ignoring imports and definitions.

    An import is how a file gets the symbol, not a branch, and the definition is
    not debt. Everything else -- a call, an attribute read, a truth test -- is a
    place core behaves differently because the device is a TPU.
    """
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:  # pragma: no cover - CI lints syntax separately
        return 0

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The definition itself is not a branch; its body is walked anyway.
            continue
        if isinstance(node, ast.Name) and node.id in _TPU_NAMES:
            count += 1
        elif isinstance(node, ast.Attribute) and node.attr in _TPU_NAMES:
            count += 1
    return count


def collect(repo_root: str, only: Optional[list[str]] = None) -> Debt:
    """Measure both numbers. ``only`` restricts the branch scan to those files."""
    restrict = set(only) if only else None

    branches: dict[str, int] = {}
    branch_root = os.path.join(repo_root, BRANCH_SCAN_ROOT)
    if os.path.isdir(branch_root):
        for path in _iter_python_files(branch_root):
            rel = _rel(path, repo_root)
            if _is_tpu_owned(rel):
                continue
            if restrict is not None and rel not in restrict:
                continue
            count = count_tpu_branches(path)
            if count:
                branches[rel] = count

    modules: list[str] = []
    module_root = os.path.join(repo_root, MODULE_SCAN_ROOT)
    if os.path.isdir(module_root):
        for path in _iter_python_files(module_root):
            rel = _rel(path, repo_root)
            if rel.startswith(IN_TREE_TPU_PACKAGE) or not names_tpu(rel):
                continue
            if restrict is not None and rel not in restrict:
                continue
            modules.append(rel)

    return Debt(branches=branches, modules=sorted(modules))


def load_baseline(repo_root: str) -> Debt:
    path = os.path.join(repo_root, BASELINE_PATH)
    if not os.path.exists(path):
        return Debt()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Debt(
        branches=dict(payload.get("branches", {})),
        modules=list(payload.get("modules", [])),
    )


def write_baseline(repo_root: str, debt: Debt) -> str:
    path = os.path.join(repo_root, BASELINE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "branches": dict(sorted(debt.branches.items())),
        "modules": sorted(debt.modules),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def compare(
    current: Debt, baseline: Debt, partial: bool
) -> tuple[list[str], list[str]]:
    """Return ``(violations, improvements)`` as human-readable lines."""
    violations: list[str] = []
    improvements: list[str] = []

    for rel, count in sorted(current.branches.items()):
        allowed = baseline.branches.get(rel, 0)
        if count > allowed:
            violations.append(
                f"{rel}: {count} is_tpu reference(s), baseline {allowed}. "
                f"New TPU branches belong behind a platform method or a "
                f"registration in hardware_backend/tpu/."
            )
        elif count < allowed:
            improvements.append(f"{rel}: {allowed} -> {count}")

    added = set(current.modules) - set(baseline.modules)
    for rel in sorted(added):
        violations.append(
            f"{rel}: new TPU module outside {IN_TREE_TPU_PACKAGE}. "
            f"New TPU code goes in hardware_backend/tpu/."
        )

    # Absence is only evidence of progress on a full scan; under --files, a path
    # missing from `current` is one that was not staged, not one that went away.
    if not partial:
        for rel, allowed in sorted(baseline.branches.items()):
            if rel not in current.branches and allowed:
                improvements.append(f"{rel}: {allowed} -> 0")
        removed = set(baseline.modules) - set(current.modules)
        for rel in sorted(removed):
            improvements.append(f"{rel}: relocated or removed")

    return violations, improvements


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=_repo_root())
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current numbers as the new ceiling",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="only check these files (pre-commit passes the staged set)",
    )
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)

    if args.write_baseline:
        debt = collect(repo_root)
        path = write_baseline(repo_root, debt)
        total = sum(debt.branches.values())
        print(
            f"wrote {_rel(path, repo_root)}: {total} is_tpu reference(s) in "
            f"{len(debt.branches)} file(s), {len(debt.modules)} TPU module(s) "
            f"outside the in-tree package"
        )
        return 0

    # Pre-commit hands absolute or repo-relative paths; normalise, and ignore
    # anything outside the scanned root so the hook can be registered broadly.
    only = None
    if args.files is not None:
        only = []
        for raw in args.files:
            rel = _rel(os.path.abspath(raw), repo_root)
            if rel.startswith(MODULE_SCAN_ROOT) and rel.endswith(".py"):
                only.append(rel)
        if not only:
            return 0

    current = collect(repo_root, only=only)
    baseline = load_baseline(repo_root)
    violations, improvements = compare(current, baseline, partial=only is not None)

    for line in improvements:
        print(f"improved: {line}")
    if improvements:
        print(
            "\nTPU debt went down. Run "
            "`python tools/plugin/check_tpu_branches.py --write-baseline` "
            "to lock the gain in.\n"
        )

    if violations:
        print("TPU debt increased:\n", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nIf the addition is genuinely unavoidable, rerun with "
            "--write-baseline and justify it in the PR description.",
            file=sys.stderr,
        )
        return 1

    scope = f"{len(current.branches)} file(s)" if only else "shared core"
    print(f"TPU debt check passed ({scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
