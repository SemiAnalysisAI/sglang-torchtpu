"""Unit test for the non-CUDA (TPU) DeviceTimer no-op.

DeviceTimer/GapTimer create CUDA timing events; on TPU (no CUDA) wrap() must
short-circuit to a plain no-op. Runs on CPU CI where CUDA is absent.
"""

import unittest

import torch

from sglang.srt.utils.device_timer import DeviceTimer, GapTimer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNonCudaDeviceTimer(CustomTestCase):
    def test_device_timer_wrap_is_noop_without_cuda(self):
        if torch.cuda.is_available():
            self.skipTest("non-CUDA no-op only exercised when CUDA is absent")
        timer = DeviceTimer(reporter=lambda *a, **k: None)
        with timer.wrap({"tag": "x"}):
            pass
        # No CUDA events were created -> no recorded intervals.
        self.assertEqual(len(timer._intervals), 0)

    def test_gap_timer_wrap_is_noop_without_cuda(self):
        if torch.cuda.is_available():
            self.skipTest("non-CUDA no-op only exercised when CUDA is absent")
        timer = GapTimer(reporter=lambda *a, **k: None)
        with timer.wrap({"tag": "x"}):
            pass
        with timer.wrap({"tag": "y"}):
            pass
        self.assertEqual(len(timer._intervals), 0)
        self.assertIsNone(timer._pending)


if __name__ == "__main__":
    unittest.main()
