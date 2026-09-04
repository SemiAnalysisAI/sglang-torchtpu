"""Generate the sglang dist-info for Bazel runfiles from pyproject_tpu.toml.

importlib.metadata must find sglang's package metadata in runfiles (platform
plugin discovery via the sglang.srt.platforms entry-point group, and
version("sglang")), but the in-tree package is never pip-installed, so no
dist-info exists. The //python:dist_info genrule runs this to derive one from
the same pyproject the wheel is built from -- nothing is duplicated by hand.

Usage: gen_dist_info.py <pyproject_tpu.toml> <out_dir>
Writes <out_dir>/sglang-0.0.0.dev0.dist-info/{METADATA,entry_points.txt}.
The version matches the setuptools-scm fallback_version in pyproject_tpu.toml.
"""

import sys
import tomllib
from pathlib import Path

_VERSION = "0.0.0.dev0"  # setuptools-scm fallback_version


def main() -> None:
    pyproject_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    project = tomllib.loads(pyproject_path.read_text())["project"]

    dist_info = out_dir / f"sglang-{_VERSION}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)

    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {_VERSION}\n"
    )

    lines = []
    for group, entries in project.get("entry-points", {}).items():
        lines.append(f"[{group}]")
        lines += [f"{k} = {v}" for k, v in entries.items()]
        lines.append("")
    scripts = project.get("scripts", {})
    if scripts:
        lines.append("[console_scripts]")
        lines += [f"{k} = {v}" for k, v in scripts.items()]
        lines.append("")
    (dist_info / "entry_points.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
