# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU device module compatibility bridge.

Verifies that calls resolving the TPU device module through
`torch.get_device_module("tpu")` or SGLang's `get_device_module()` satisfy
the full device module protocol backed by `torch.accelerator.*`.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.patches import (
    _TpuDeviceContext,
    bridge_tpu_device_module,
)
from sglang.srt.utils import get_device_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuDeviceModuleBridge(CustomTestCase):
    """Test TPU device module bridging to torch.accelerator."""

    def test_bridge_attaches_all_methods(self):
        """Simulate a stripped TPU device module and verify all methods are attached."""
        mock_tpu = SimpleNamespace()
        bridged = bridge_tpu_device_module(mock_tpu)

        self.assertIs(bridged, mock_tpu)
        self.assertTrue(callable(bridged.synchronize))
        self.assertTrue(callable(bridged.is_available))
        self.assertTrue(callable(bridged.device_count))
        self.assertTrue(callable(bridged.current_device))
        self.assertTrue(callable(bridged.current_device_index))
        self.assertTrue(callable(bridged.set_device))
        self.assertTrue(callable(bridged.set_device_index))
        self.assertTrue(callable(bridged.empty_cache))
        self.assertTrue(callable(bridged.memory_allocated))
        self.assertTrue(callable(bridged.max_memory_allocated))
        self.assertTrue(callable(bridged.memory_reserved))
        self.assertTrue(callable(bridged.max_memory_reserved))
        self.assertTrue(callable(bridged.reset_peak_memory_stats))
        self.assertTrue(callable(bridged.is_current_stream_capturing))
        self.assertFalse(bridged.is_current_stream_capturing())
        self.assertTrue(hasattr(bridged, "device"))

    def test_delegation_to_torch_accelerator(self):
        """Verify bridged methods forward to torch.accelerator."""
        mock_tpu = SimpleNamespace()
        bridge_tpu_device_module(mock_tpu)

        with (
            patch.object(torch.accelerator, "synchronize") as mock_sync,
            patch.object(
                torch.accelerator, "current_device_index", return_value=2
            ) as mock_curr,
            patch.object(torch.accelerator, "set_device_index") as mock_set,
            patch.object(
                torch.accelerator, "is_available", return_value=True
            ) as mock_avail,
            patch.object(
                torch.accelerator, "device_count", return_value=4
            ) as mock_count,
        ):
            mock_tpu.synchronize()
            mock_sync.assert_called_once_with(None)

            mock_tpu.synchronize("tpu:1")
            mock_sync.assert_called_with("tpu:1")

            self.assertEqual(mock_tpu.current_device(), 2)
            self.assertEqual(mock_tpu.current_device_index(), 2)
            mock_curr.assert_called()

            mock_tpu.set_device(3)
            mock_set.assert_called_with(3)

            mock_tpu.set_device_index(1)
            mock_set.assert_called_with(1)

            self.assertTrue(mock_tpu.is_available())
            mock_avail.assert_called_once()

            self.assertEqual(mock_tpu.device_count(), 4)
            mock_count.assert_called_once()

    def test_device_context_manager(self):
        """Verify _TpuDeviceContext correctly enters and restores previous device."""
        with (
            patch.object(torch.accelerator, "current_device_index", return_value=1),
            patch.object(torch.accelerator, "set_device_index") as mock_set,
        ):
            ctx = _TpuDeviceContext(3)
            with ctx:
                mock_set.assert_called_with(3)
            mock_set.assert_called_with(1)

        # Context manager with torch.device and string
        with (
            patch.object(torch.accelerator, "current_device_index", return_value=0),
            patch.object(torch.accelerator, "set_device_index") as mock_set,
        ):
            with _TpuDeviceContext(torch.device("tpu:2")):
                mock_set.assert_called_with(2)
            mock_set.assert_called_with(0)

        # No-op with None
        with patch.object(torch.accelerator, "set_device_index") as mock_set:
            with _TpuDeviceContext(None):
                pass
            mock_set.assert_not_called()

    def test_torch_get_device_module_returns_bridged_tpu(self):
        """Verify torch.get_device_module('tpu') has all bridged methods."""
        mod = torch.get_device_module("tpu")
        self.assertIsNotNone(mod)
        self.assertTrue(callable(mod.synchronize))
        self.assertTrue(callable(mod.current_device))
        self.assertTrue(callable(mod.set_device))
        self.assertTrue(callable(mod.is_available))
        self.assertTrue(callable(mod.device_count))
        self.assertTrue(callable(mod.empty_cache))
        self.assertTrue(callable(mod.memory_allocated))
        self.assertTrue(callable(mod.is_current_stream_capturing))

    def test_sglang_get_device_module_tpu(self):
        """Verify SGLang's get_device_module() returns the bridged module under TPU."""
        with patch("sglang.srt.utils.common.is_tpu", return_value=True):
            # Clear lru_cache to re-evaluate
            get_device_module.cache_clear()
            try:
                mod = get_device_module()
                self.assertIsNotNone(mod)
                self.assertTrue(callable(mod.synchronize))
                self.assertTrue(callable(mod.current_device))
                self.assertTrue(callable(mod.set_device))
            finally:
                get_device_module.cache_clear()


if __name__ == "__main__":
    unittest.main()
