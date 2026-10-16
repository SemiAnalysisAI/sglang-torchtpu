# SPDX-License-Identifier: Apache-2.0
"""Require an SPDX license identifier at the top of TPU-owned Python files.

The check covers the TPU plugin and the fork-only TPU directories under
python/sglang/srt. Upstream-owned files are left alone so upstream syncs stay
clean. A file passes when one of its first HEADER_WINDOW lines carries
``SPDX-License-Identifier:``; empty files are skipped.

Usage:
    python tools/plugin/check_spdx_headers.py               # check every covered file
    python tools/plugin/check_spdx_headers.py --fix         # insert the default header
    python tools/plugin/check_spdx_headers.py --files a.py  # pre-commit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Iterable, List

#: Fork-owned TPU paths, relative to the repo root.
COVERED = re.compile(
    r"^(sglang_tpu/"
    r"|python/sglang/srt/hardware_backend/tpu/"
    r"|python/sglang/srt/layers/attention/pallas/"
    r"|python/sglang/srt/layers/moe/pallas/"
    r"|python/sglang/srt/disaggregation/raiden/"
    r").*\.py$"
)

#: Vendored files put the source URL and the original copyright above the
#: identifier, so look a few lines down instead of requiring line 1.
HEADER_WINDOW = 10

DEFAULT_HEADER = "# SPDX-License-Identifier: Apache-2.0\n"


def covered_files() -> List[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], check=True, capture_output=True, text=True
    ).stdout
    return [p for p in out.splitlines() if COVERED.match(p)]


def has_header(lines: List[str]) -> bool:
    return any("SPDX-License-Identifier:" in line for line in lines[:HEADER_WINDOW])


def insert_header(lines: List[str]) -> List[str]:
    # Keep a shebang and a PEP 263 encoding line first; both must stay on top.
    at = 0
    if at < len(lines) and lines[at].startswith("#!"):
        at += 1
    if at < len(lines) and re.match(r"^#.*coding[:=]", lines[at]):
        at += 1
    return lines[:at] + [DEFAULT_HEADER] + lines[at:]


def run(paths: Iterable[str], fix: bool) -> int:
    missing = []
    for path in paths:
        if not COVERED.match(path):
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if not lines or has_header(lines):
            continue
        missing.append(path)
        if fix:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(insert_header(lines))

    if not missing:
        return 0
    if fix:
        print(f"Added '{DEFAULT_HEADER.strip()}' to {len(missing)} file(s).")
        return 0
    print("Missing SPDX license header (add it as the first line):")
    for path in missing:
        print(f"  {path}")
    print(
        f"\nRun: python tools/plugin/check_spdx_headers.py --fix\n"
        f"Code adapted from another project keeps that project's license and "
        f"copyright; see NOTICE."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="insert missing headers")
    parser.add_argument("--files", nargs="*", help="check only these paths")
    args = parser.parse_args()
    paths = args.files if args.files is not None else covered_files()
    return run(paths, args.fix)


if __name__ == "__main__":
    sys.exit(main())
