"""pr-gate checks for .github/tpu-ci/filter_tests.py (no TPU).

The filter narrows a nightly dispatch to one test. Its failure mode is silent:
a matrix that came back empty makes GitHub skip the job, and a skipped job
reads as a pass, so a filter that matched nothing would produce a green run
that tested nothing. feature-validation-tpu.yml's `filter-matched` job is what
turns that into a red run; these checks cover the matching itself, including
the shape-agnostic row lookup that lets one filter serve both the single-host
matrix (`file`) and the multi-host launch groups (`files`).
"""

import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FILTER_TESTS = (
    Path(__file__).resolve().parents[3] / ".github" / "tpu-ci" / "filter_tests.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("filter_tests", FILTER_TESTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SINGLE_HOST = [
    {"name": "a", "file": "test/registered/tpu/e2e/test_tpu_alpha_spec.py"},
    {"name": "b", "file": "test/registered/tpu/e2e/test_tpu_alpha_spec_dpa.py"},
    {"name": "c", "file": "test/registered/tpu/e2e/test_tpu_beta.py"},
]
# discover_multihost.py names a group's test in `files`, space-separated, and
# the dashboard contract holds it to exactly one entry.
GROUPS = [{"name": "g", "files": "test/registered/tpu/e2e/test_tpu_beta_mh.py"}]


class TestFilterRows(CustomTestCase):
    def setUp(self):
        self.mod = _load()

    def test_substring_keeps_matching_rows(self):
        kept = self.mod._filter_rows(SINGLE_HOST, "alpha_spec")
        self.assertEqual([row["name"] for row in kept], ["a", "b"])

    def test_py_suffix_pins_one_row(self):
        kept = self.mod._filter_rows(SINGLE_HOST, "alpha_spec.py")
        self.assertEqual([row["name"] for row in kept], ["a"])

    def test_launch_groups_match_on_files(self):
        # Shape-agnostic: a group row has `files`, not `file`. Reading only
        # `file` would raise here and break the multi-host matrix.
        self.assertEqual(self.mod._filter_rows(GROUPS, "beta_mh"), GROUPS)

    def test_no_match_is_empty_not_an_error(self):
        # A filter naming a single-host test legitimately matches no group, so
        # the helper cannot decide fatality; the workflow checks across both.
        self.assertEqual(self.mod._filter_rows(GROUPS, "alpha_spec"), [])


if __name__ == "__main__":
    unittest.main()
