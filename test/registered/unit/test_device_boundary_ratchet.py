"""Core reaches device code through the platform interface, not direct imports.

The TPU backend is being reshaped into an out-of-tree plugin. Two couplings have
to stop growing for that to be possible, and this ratchet caps both:

* **Core importing the device package.** After the plugin ships as its own
  distribution, ``sglang.srt.hardware_backend.tpu`` is not importable from core
  at all -- stock sglang does not ship it, and a plugin cannot be a dependency
  of the thing it plugs into. Core should receive device classes rather than
  name them: a platform factory (``current_platform.get_mha_kv_pool_cls()``), a
  registry (``register_attention_backend``), or a dispatch table
  (``BaseFusedOp.register_oot_forward``), all populated from ``init_backend()``.
* **Module-scope device imports**, which are the same thing but worse: they run
  on every platform, so they break CUDA and CPU users at import rather than at
  use. These are tracked separately and should reach zero first.
* **``is_tpu()`` branches in core**, which are what keeps a file forked. A
  branch that calls a device function looked up at call time is an accepted
  middle ground; a new one still needs a linked upstream seam issue.

Device code that merely *lives* in the core tree (the Pallas trees, raiden, the
TPU graph runner) is not coupling -- it relocates with the plugin -- so it is
excluded from both counts.

The baseline is per file, so a new offending file fails, an existing file
growing fails, and a file that improves without the baseline being lowered also
fails. Regenerate after a genuine change with::

    SGLANG_UPDATE_DEVICE_BOUNDARY_BASELINE=1 python test/registered/unit/test_device_boundary_ratchet.py

The diff is the record of what moved, in either direction.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import ast
import json
import os
import re
import unittest
from pathlib import Path

import sglang
from sglang.test.test_utils import CustomTestCase

_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))
_BASELINE_PATH = Path(__file__).with_name("device_boundary_baseline.json")

# Device code living in the core tree. It moves to the plugin as a unit, so its
# own imports and branches are internal, not a boundary violation.
_DEVICE_TREES = (
    "srt/hardware_backend/tpu/",
    "srt/layers/attention/pallas/",
    "srt/layers/moe/pallas/",
    "srt/disaggregation/tpu/",
    "srt/disaggregation/raiden/",
    "srt/mem_cache/storage/umbp/",
    "srt/model_executor/tpu_graph_runner.py",
)

# Import roots that name device code. The second is the plugin distribution the
# in-tree package becomes; listed now so the ratchet keeps working across the
# move instead of silently measuring nothing.
_DEVICE_PACKAGES = ("sglang.srt.hardware_backend.tpu", "sglang_tpu")

# A dotted device path inside a string literal -- importlib.import_module(...)
# and friends -- is the obvious way around an AST import scan.
_DYNAMIC_IMPORT = re.compile(
    r"""["'](?:%s)[."']""" % "|".join(re.escape(p) for p in _DEVICE_PACKAGES)
)

_IS_TPU = re.compile(r"\b_is_tpu\b|\bis_tpu\s*\(")


def _is_device_tree(rel: str) -> bool:
    return rel.startswith(_DEVICE_TREES)


def _names_device_package(dotted: str) -> bool:
    return any(
        dotted == pkg or dotted.startswith(pkg + ".") for pkg in _DEVICE_PACKAGES
    )


def _scan() -> dict[str, dict[str, int]]:
    """Per-file counts for each tracked coupling, over core files only."""
    imports: dict[str, int] = {}
    module_scope: dict[str, int] = {}
    branches: dict[str, int] = {}

    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if _is_device_tree(rel):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        n_branch = sum(1 for line in source.splitlines() if _IS_TPU.search(line))
        if n_branch:
            branches[rel] = n_branch

        if not any(pkg in source for pkg in _DEVICE_PACKAGES):
            continue

        n_import = len(_DYNAMIC_IMPORT.findall(source))
        n_module_scope = 0
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Unparseable file: fall back to counting textual import lines so a
            # syntax error cannot be used to hide a new import.
            n_import += sum(
                1
                for line in source.splitlines()
                if re.match(r"\s*(from|import)\s+(%s)" % "|".join(
                    re.escape(p) for p in _DEVICE_PACKAGES), line)
            )
        else:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    dotted = node.module or ""
                elif isinstance(node, ast.Import):
                    dotted = next(
                        (a.name for a in node.names if _names_device_package(a.name)),
                        "",
                    )
                else:
                    continue
                if not _names_device_package(dotted):
                    continue
                n_import += 1
                if node.col_offset == 0:
                    n_module_scope += 1

        if n_import:
            imports[rel] = n_import
        if n_module_scope:
            module_scope[rel] = n_module_scope

    return {
        "device_imports": imports,
        "module_scope_device_imports": module_scope,
        "is_tpu_branches": branches,
    }


def _load_baseline() -> dict[str, dict[str, int]]:
    if not _BASELINE_PATH.exists():
        return {k: {} for k in ("device_imports", "module_scope_device_imports", "is_tpu_branches")}
    return json.loads(_BASELINE_PATH.read_text())["counters"]


def _write_baseline(counters: dict[str, dict[str, int]]) -> None:
    totals = {k: sum(v.values()) for k, v in counters.items()}
    _BASELINE_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "Ratchet baseline for test_device_boundary_ratchet.py. "
                    "Per-file counts; every number may only go down. Regenerate with "
                    "SGLANG_UPDATE_DEVICE_BOUNDARY_BASELINE=1."
                ),
                "totals": totals,
                "counters": {k: dict(sorted(v.items())) for k, v in counters.items()},
            },
            indent=2,
        )
        + "\n"
    )


def _diff(counter: str, current: dict[str, int], baseline: dict[str, int]):
    """(regressions, improvements) for one counter."""
    regressions, improvements = [], []
    for rel, n in sorted(current.items()):
        allowed = baseline.get(rel, 0)
        if n > allowed:
            where = "new file" if rel not in baseline else f"was {allowed}"
            regressions.append(f"  {rel}: {n} ({where})")
    for rel, allowed in sorted(baseline.items()):
        n = current.get(rel, 0)
        if n < allowed:
            improvements.append(f"  {rel}: {allowed} -> {n}")
    return regressions, improvements


_GUIDANCE = {
    "device_imports": (
        "core must not import the device package. Hand the class to core instead: "
        "a platform factory (get_*_cls), a registry (register_attention_backend), "
        "or BaseFusedOp.register_oot_forward, registered from init_backend()."
    ),
    "module_scope_device_imports": (
        "a module-scope device import runs on EVERY platform and breaks stock "
        "sglang at import, not at use. Move it inside the function that needs it "
        "at minimum, and prefer a registry lookup."
    ),
    "is_tpu_branches": (
        "a new device branch in core keeps that file forked. Use a platform "
        "capability or a registry; if a branch is genuinely the right answer for "
        "now, link the upstream seam issue in the PR before updating the baseline."
    ),
}


class TestDeviceBoundaryRatchet(CustomTestCase):
    """Each coupling may shrink, never grow."""

    @classmethod
    def setUpClass(cls):
        cls.current = _scan()
        cls.baseline = _load_baseline()
        if os.environ.get("SGLANG_UPDATE_DEVICE_BOUNDARY_BASELINE"):
            _write_baseline(cls.current)
            print(f"baseline rewritten: {_BASELINE_PATH}")

    def _check(self, counter: str):
        regressions, improvements = _diff(
            counter, self.current[counter], self.baseline.get(counter, {})
        )
        self.assertFalse(
            regressions,
            f"{counter} grew -- {_GUIDANCE[counter]}\n" + "\n".join(regressions),
        )
        self.assertFalse(
            improvements,
            f"{counter} shrank, which is the point -- lower the baseline so it "
            "cannot grow back:\n  SGLANG_UPDATE_DEVICE_BOUNDARY_BASELINE=1 python "
            f"{Path(__file__).name}\n" + "\n".join(improvements),
        )

    def test_core_does_not_import_the_device_package(self):
        self._check("device_imports")

    def test_no_module_scope_device_imports(self):
        self._check("module_scope_device_imports")

    def test_no_new_is_tpu_branches_in_core(self):
        self._check("is_tpu_branches")

    def test_baseline_is_present_and_nonempty(self):
        """A missing or emptied baseline would make every other test vacuous."""
        self.assertTrue(
            _BASELINE_PATH.exists(), f"baseline file missing: {_BASELINE_PATH}"
        )
        self.assertTrue(
            any(self.baseline.get(k) for k in _GUIDANCE),
            "baseline has no entries; the ratchet would pass on anything",
        )


if __name__ == "__main__":
    unittest.main()
