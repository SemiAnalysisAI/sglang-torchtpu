"""The persistent torch.compile cache is split per topology where the platform asks."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.platforms.interface import Platform
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform


def _configure(namespace):
    platform = MagicMock()
    platform.get_compile_cache_namespace.return_value = namespace
    worker = SimpleNamespace(server_args=SimpleNamespace(num_gpus=4, nnodes=1))
    with (
        tempfile.TemporaryDirectory() as root,
        patch.object(gpu_worker, "current_platform", platform),
        patch.object(gpu_worker.envs, "SGLANG_DIFFUSION_CACHE_ROOT", root),
        patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        os.environ.pop("TRITON_CACHE_DIR", None)
        gpu_worker.GPUWorker._configure_persistent_torch_compile_cache(worker)
        platform.get_compile_cache_namespace.assert_called_once_with(4, 1)
        return os.path.relpath(os.environ["TORCHINDUCTOR_CACHE_DIR"], root)


class TestCompileCacheNamespace(unittest.TestCase):
    def test_tpu_splits_the_cache_per_topology_and_flags(self):
        tpu = TpuPlatform()
        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--a=1"}):
            one, four = (tpu.get_compile_cache_namespace(n, 1) for n in (1, 4))
            self.assertTrue(four.startswith("tpu_4x1_"))
            self.assertNotEqual(one, four)
            self.assertEqual(four, tpu.get_compile_cache_namespace(4, 1))
        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--a=2"}):
            self.assertNotEqual(four, tpu.get_compile_cache_namespace(4, 1))

    def test_other_platforms_share_one_cache(self):
        self.assertIsNone(Platform.get_compile_cache_namespace(None, 4, 1))

    def test_worker_nests_the_cache_under_the_namespace(self):
        self.assertEqual(
            _configure("tpu_4x1"), os.path.join("torch_compile_cache", "tpu_4x1", "inductor")
        )
        self.assertEqual(_configure(None), os.path.join("torch_compile_cache", "inductor"))


if __name__ == "__main__":
    unittest.main()
