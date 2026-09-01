"""Bookkeeping guards for the external TPU benchmark matrix (no TPU needed).

The matrix is stitched together by convention across three places -- the
test files' MULTIHOST_LAUNCH dicts (AST-scanned), the launch wrappers under
benchmark/tpu/recipe/external_bench/, and config.sh's (model, recipe,
workload) axes -- and nothing else in CI checks the stitching without a TPU
slice. Each case here answers "someone extended X without updating Y":

  * a test file naming a launch wrapper that does not exist ships a group
    whose JobSet dies at `bash: no such file` after weights are mounted;
  * the two workload files of a (model, recipe) cell drifting apart in
    (suite, launch_script, weights) silently splits their shared server into
    two full bring-ups;
  * a recipe registering against the wrong slice shape (a pd cell on the
    2-host suite) only fails at engine rendezvous, an hour into the run;
  * config.sh losing a sweep axis (the task's {1,16,64} x {0,1,4} grid) would
    keep every downstream gate green while measuring a smaller matrix.
"""

import ast
import glob
import importlib.util
import os
import subprocess
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3))
E2E = os.path.join(REPO, "test", "registered", "tpu", "e2e")
EXT_BENCH = os.path.join(REPO, "benchmark", "tpu", "recipe", "external_bench")

MODELS = ("gpt_oss_120b", "qwen3_coder_480b")
RECIPES = {
    "1host_pc": "external-nightly-2x2x2",
    "1host_pc_spec_kvoff": "external-nightly-2x2x2",
    "pd_2p2d_pc_spec_kvoff": "external-nightly-2x2x4",
}
WORKLOADS = ("agentic", "8k1k")


def _scan_file(path):
    """(suite, MULTIHOST_LAUNCH, disabled) the CI scanner would read, via the
    same AST contract (literal kwargs / literal dict)."""
    tree = ast.parse(open(path).read())
    suite, launch, disabled = None, None, None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "register_tpu_ci"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            if "suite" in kw:
                suite = ast.literal_eval(kw["suite"])
            if "disabled" in kw:
                disabled = ast.literal_eval(kw["disabled"])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "MULTIHOST_LAUNCH":
                    launch = ast.literal_eval(node.value)
    return suite, launch, disabled


class TestExternalBenchMatrix(CustomTestCase):
    def _cell_files(self):
        files = sorted(glob.glob(os.path.join(E2E, "test_tpu_ext_*.py")))
        self.assertEqual(
            len(files),
            len(MODELS) * len(RECIPES) * len(WORKLOADS),
            "the matrix is 2 models x 3 recipes x 2 workloads; a missing or "
            "extra test_tpu_ext_* file means a cell was added or dropped "
            "without updating this inventory",
        )
        return files

    def test_every_cell_names_an_existing_launch_wrapper(self):
        for path in self._cell_files():
            _, launch, _ = _scan_file(path)
            self.assertIsNotNone(launch, f"{path}: no literal MULTIHOST_LAUNCH")
            script = os.path.join(REPO, "benchmark", "tpu", launch["launch_script"])
            self.assertTrue(
                os.path.isfile(script),
                f"{path}: launch_script {launch['launch_script']} does not exist",
            )

    def test_workload_pairs_share_one_launch_group(self):
        """Files sharing (suite, launch_script, weights) share one server; a
        drifted field silently doubles the bring-up cost of the cell."""
        for model in MODELS:
            for recipe in RECIPES:
                keys = set()
                for workload in WORKLOADS:
                    path = os.path.join(
                        E2E, f"test_tpu_ext_{model}_{recipe}_{workload}.py"
                    )
                    suite, launch, _ = _scan_file(path)
                    keys.add((suite, launch["launch_script"], launch["weights"]))
                self.assertEqual(
                    len(keys),
                    1,
                    f"{model}/{recipe}: workload files disagree on the launch "
                    f"group key: {keys}",
                )

    def test_recipe_registers_against_its_slice_shape(self):
        """A pd cell on the 2-host suite (or a 1-host cell on the 4-host one)
        fails only at engine rendezvous, an hour into the run."""
        for path in self._cell_files():
            suite, _, _ = _scan_file(path)
            recipe = next(r for r in RECIPES if f"_{r}_" in os.path.basename(path))
            self.assertEqual(
                suite,
                RECIPES[recipe],
                f"{path}: recipe {recipe} must register to {RECIPES[recipe]}",
            )

    def test_discovery_buckets_one_group_per_slice(self):
        """The even-split contract: 4 single-host groups over the four 2x2x2
        slices, 2 pd groups over the two 2x2x4 slices, none serialized."""
        spec = importlib.util.spec_from_file_location(
            "render_external_jobset",
            os.path.join(
                REPO, ".github", "tpu-ci", "external", "render_external_jobset.py"
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.load_config(
            os.path.join(
                REPO, ".github", "tpu-ci", "external", "external-topology.yaml"
            )
        )
        slices = mod.list_groups(
            cfg, os.path.join(REPO, "test", "registered"), nightly_only=True
        )
        per_suite = {}
        for sl in slices:
            ours = [
                g
                for g in sl["groups"]
                if g["launch_script"].startswith("recipe/external_bench/")
            ]
            self.assertLessEqual(
                len(ours),
                1,
                f"{sl['name']}: {len(ours)} external_bench groups serialize on "
                f"one slice -- the matrix is sized to fill each slice exactly "
                f"once ({[g['name'] for g in ours]})",
            )
            per_suite.setdefault(sl["suite"], 0)
            per_suite[sl["suite"]] += len(ours)
            for g in ours:
                self.assertEqual(
                    g["timeout_per_file"],
                    5400,
                    f"{g['name']}: discovery must carry the files' declared "
                    "timeout_per_file (the JobSet deadline is sized from it)",
                )
        # Expected group counts follow the files themselves: a cell disabled
        # with a reason (e.g. the raiden P/D transfer finding) is skipped by
        # discovery, and this test must not force it back on.
        expected = {}
        for path in self._cell_files():
            suite, launch, disabled = _scan_file(path)
            if disabled is not None:
                continue
            key = (suite, launch["launch_script"])
            expected.setdefault(suite, set()).add(key)
        self.assertEqual(
            per_suite.get("external-nightly-2x2x2", 0),
            len(expected.get("external-nightly-2x2x2", ())),
        )
        self.assertEqual(
            per_suite.get("external-nightly-2x2x4", 0),
            len(expected.get("external-nightly-2x2x4", ())),
        )

    def test_config_resolves_the_tasks_sweep_grid(self):
        """config.sh must keep the task's grid: concurrency {1,16,64} x long
        requests {0,1,4} in the ci profile, for every (model, recipe)."""
        for model in MODELS:
            for recipe in RECIPES:
                out = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source ./config.sh && echo "BS=$SWEEP_BS|LONG=$SWEEP_LONG"',
                    ],
                    cwd=EXT_BENCH,
                    env={
                        **os.environ,
                        "BENCH_MODEL": model,
                        "BENCH_RECIPE": recipe,
                        "WORKLOAD": "agentic",
                        "PROFILE": "ci",
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(out.returncode, 0, f"{model}/{recipe}: {out.stderr}")
                self.assertIn("BS=1 16 64|LONG=0 1 4", out.stdout)


if __name__ == "__main__":
    unittest.main()
