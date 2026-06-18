"""Server-level negative startup tests for the NGram-on-TPU capability gate (AC-1).

Each unsupported configuration must fail fast at startup (the gate runs during
ServerArgs validation, before any model load) with a clear capability error.

Usage:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_ngram_tpu_startup.py
"""

import os
import subprocess
import sys
import unittest

from _ngram_tpu_common import MODEL, skip_if_no_tpu

_BASE = [
    "--model-path", MODEL,
    "--device", "tpu",
    "--dtype", "bfloat16",
    "--page-size", "1",
    "--grammar-backend", "none",
    "--mem-fraction-static", "0.4",
    "--speculative-algorithm", "NGRAM",
    "--speculative-num-draft-tokens", "4",
    "--host", "127.0.0.1",
    "--port", "21099",
]


def _launch_expect_failure(extra_args):
    env = dict(os.environ)
    env["SGLANG_PLATFORM"] = "tpu"
    cmd = [sys.executable, "-m", "sglang.launch_server", *_BASE, *extra_args]
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300
    )
    return proc.returncode, proc.stdout + proc.stderr


@unittest.skipIf(skip_if_no_tpu(), "torch_tpu not available")
class TestNgramTpuStartupGate(unittest.TestCase):
    def test_breadth_gt_one_rejected(self):
        rc, out = _launch_expect_failure(["--speculative-ngram-max-bfs-breadth", "4"])
        self.assertNotEqual(rc, 0, "server should not start with breadth>1")
        self.assertIn("speculative_ngram_max_bfs_breadth=1", out)

    def test_non_rpa_v3_backend_rejected(self):
        rc, out = _launch_expect_failure(["--attention-backend", "pallas_tpu_flash"])
        self.assertNotEqual(rc, 0, "server should not start with non-rpa_v3 backend")
        self.assertIn("pallas_tpu_rpa_v3", out)

    def test_torch_compile_rejected(self):
        rc, out = _launch_expect_failure(["--enable-torch-compile"])
        self.assertNotEqual(rc, 0, "server should not start with --enable-torch-compile")
        self.assertIn("eager mode only", out)


if __name__ == "__main__":
    unittest.main()
