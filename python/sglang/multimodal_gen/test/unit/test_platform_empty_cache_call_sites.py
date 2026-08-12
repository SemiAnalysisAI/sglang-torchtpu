"""Regression tests: converted empty_cache call sites resolve and execute.

Catches the class of bug where a call site is converted to
``current_platform.empty_cache()`` without importing ``current_platform``
(NameError only surfaces on the memory-release runtime path, not at import).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.nn as nn


class TestEmptyCacheCallSites(unittest.TestCase):
    def test_component_manager_release_path_executes(self):
        from sglang.multimodal_gen.runtime.managers.memory_managers import (
            component_manager as cm,
        )

        manager = object.__new__(cm.ComponentResidencyManager)
        use = SimpleNamespace(memory_intensive=True)
        module = nn.Linear(2, 2)  # on CPU => "released from device"
        platform = MagicMock()
        with patch.object(cm, "current_platform", platform):
            cm.ComponentResidencyManager._empty_cache_after_large_release(
                manager,
                use=use,
                strategy=None,
                module=module,
                was_on_cuda=True,
            )
        platform.empty_cache.assert_called_once_with()

    def test_lora_pipeline_references_platform(self):
        from sglang.multimodal_gen.runtime.pipelines_core import lora_pipeline

        self.assertTrue(hasattr(lora_pipeline, "current_platform"))

    def test_no_raw_device_module_memory_calls_in_runtime(self):
        """Sweep: memory-stat calls go through the platform hooks.

        ``TpuDeviceModule`` implements none of the CUDA memory-stat API, so a
        raw ``torch.get_device_module().empty_cache()`` (or reset/max stats)
        raises AttributeError at runtime on TPU. The hooks in
        ``platforms/interface.py`` are the one sanctioned caller."""
        import pathlib
        import re

        import sglang.multimodal_gen as mg

        root = pathlib.Path(mg.__file__).parent / "runtime"
        pattern = re.compile(
            r"get_device_module\(\)\s*\.\s*"
            r"(empty_cache|reset_peak_memory_stats|memory_allocated|"
            r"memory_reserved|max_memory_allocated|max_memory_reserved)"
        )
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "interface.py" and path.parent.name == "platforms":
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(root)}:{lineno}")
        self.assertEqual(
            offenders,
            [],
            "raw device-module memory calls found (route through "
            f"current_platform hooks): {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
