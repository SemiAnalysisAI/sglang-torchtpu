import subprocess
import unittest
from unittest.mock import patch

from sglang.srt.hardware_backend.tpu.topology import probe_topological_node_rank
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestTpuTopologyProbe(unittest.TestCase):
    @patch("sglang.srt.hardware_backend.tpu.topology.subprocess.run")
    def test_parses_marked_result_from_disposable_client(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "unrelated runtime output\n"
                'SGLANG_TPU_TOPOLOGY={"node_rank":2,"coords":[4,0,0],'
                '"topology":"8x4x1"}\n'
            ),
            stderr="",
        )

        self.assertEqual(
            probe_topological_node_rank(),
            {"node_rank": 2, "coords": [4, 0, 0], "topology": "8x4x1"},
        )
        self.assertIn("--json", run.call_args.args[0])

    @patch("sglang.srt.hardware_backend.tpu.topology.subprocess.run")
    def test_reports_client_failure_without_falling_back_to_worker_id(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="coordinate API unavailable"
        )

        with self.assertRaisesRegex(RuntimeError, "coordinate API unavailable"):
            probe_topological_node_rank()


if __name__ == "__main__":
    unittest.main()
