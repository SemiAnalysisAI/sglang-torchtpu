# SPDX-License-Identifier: Apache-2.0
"""``source_fingerprint`` names a kernel's sources in its custom-op name, so an
edited kernel or block-size table never matches a stale compiled graph."""

import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.pallas_export import source_fingerprint

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSourceFingerprint(CustomTestCase):
    def test_digest_follows_file_contents_not_argument_order(self):
        with tempfile.TemporaryDirectory() as directory:
            kernel, table = Path(directory, "kernel.py"), Path(directory, "table.py")
            kernel.write_text("def kernel(): pass\n")
            table.write_text("BLOCK = 128\n")
            digest = source_fingerprint(kernel, table)
            self.assertRegex(digest, r"^[0-9a-f]{12}$")
            self.assertEqual(source_fingerprint(table, kernel), digest)
            self.assertEqual(source_fingerprint(str(kernel), str(table)), digest)
            table.write_text("BLOCK = 256\n")
            self.assertNotEqual(source_fingerprint(kernel, table), digest)


if __name__ == "__main__":
    unittest.main()
