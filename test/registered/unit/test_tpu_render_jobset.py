"""render_jobset.py: the bring-up budget of a multi-host launch group (no TPU).

A file's MULTIHOST_LAUNCH["bringup_timeout"] sets how long rank 0 polls
/health_generate and pads the JobSet deadline; files without it keep the
300-poll / 3600 s allowance every existing group was tuned against.
"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[3]
RENDER = REPO / ".github" / "tpu-ci" / "render_jobset.py"
TIMEOUT_PER_FILE = 1800


class TestLaunchBringupBudget(CustomTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _test_file(self, name, **launch):
        path = self.root / f"{name}.py"
        launch = {"hosts": 2, "launch_script": "recipe/x.py", **launch}
        path.write_text(f"MULTIHOST_LAUNCH = {launch!r}\n")
        return str(path)

    def _run(self, files, *args):
        return subprocess.run(
            [
                sys.executable,
                str(RENDER),
                "--launch-script",
                "recipe/x.py",
                "--files",
                " ".join(files),
                "--timeout-per-file",
                str(TIMEOUT_PER_FILE),
                *args,
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def _render(self, *files):
        """(health polls, activeDeadlineSeconds) of the rendered JobSet."""
        jobset = self._run(files, "--image", "img", "--hosts", "2", "--lane", "nightly")
        polls = re.search(r"for i in \$\(seq 1 (\d+)\)", jobset)
        deadline = re.search(r"activeDeadlineSeconds: (\d+)", jobset)
        return int(polls.group(1)), int(deadline.group(1))

    def _wait_seconds(self, *files):
        return int(self._run(files, "--print-wait-seconds"))

    def test_undeclared_budget_keeps_the_default(self):
        f = self._test_file("plain")
        self.assertEqual(self._render(f), (300, 3600 + TIMEOUT_PER_FILE))

    def test_declared_budget_sets_polls_and_deadline(self):
        f = self._test_file("slow", bringup_timeout=5400)
        self.assertEqual(self._render(f), (540, 5400 + 600 + TIMEOUT_PER_FILE))

    def test_budget_below_the_default_keeps_the_default(self):
        f = self._test_file("fast", bringup_timeout=600)
        self.assertEqual(self._render(f), (300, 3600 + TIMEOUT_PER_FILE))

    def test_group_takes_its_longest_budget(self):
        files = (
            self._test_file("plain"),
            self._test_file("slow", bringup_timeout=4500),
        )
        self.assertEqual(self._render(*files), (450, 4500 + 600 + 2 * TIMEOUT_PER_FILE))

    def test_print_wait_seconds_is_the_rendered_deadline(self):
        for launch in ({}, {"bringup_timeout": 5400}):
            with self.subTest(launch=launch):
                f = self._test_file("case", **launch)
                self.assertEqual(self._wait_seconds(f), self._render(f)[1])

    def test_print_wait_seconds_needs_the_files(self):
        # Without them the deadline would count no file timeout at all.
        result = subprocess.run(
            [
                sys.executable,
                str(RENDER),
                "--launch-script",
                "recipe/x.py",
                "--print-wait-seconds",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs --files", result.stderr)


if __name__ == "__main__":
    unittest.main()
