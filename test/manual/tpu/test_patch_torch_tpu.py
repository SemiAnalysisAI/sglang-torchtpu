"""Unit tests for monkey_patch_torch_tpu_env.

Runs anywhere (no TPU required) — exercises pure Python env-var logic.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils.patch_torch import monkey_patch_torch_tpu_env
from sglang.test.test_utils import CustomTestCase

_TORCHRUN_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE",
    "TORCH_TPU_INTERNAL_DETECT_REPEATED_OPS",
    "SGLANG_TPU_HLO_DUMP_DIR",
    "XLA_FLAGS",
)


class TestMonkeyPatchTorchTpuEnv(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _TORCHRUN_ENV_VARS}
        for k in _TORCHRUN_ENV_VARS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_skips_when_device_not_tpu(self):
        args = SimpleNamespace(device="cuda", tp_size=8, nnodes=1, node_rank=0)
        monkey_patch_torch_tpu_env(args, tp_rank=3)
        for k in _TORCHRUN_ENV_VARS:
            self.assertNotIn(k, os.environ)

    def test_sets_torchrun_vars_for_tpu(self):
        args = SimpleNamespace(device="tpu", tp_size=4, nnodes=2, node_rank=1)
        monkey_patch_torch_tpu_env(args, tp_rank=2)
        # Current launchers pass a global tp_rank and global tp_size.  Derive
        # only the host-local rank/size here; do not apply node_rank twice.
        self.assertEqual(os.environ["RANK"], "2")
        self.assertEqual(os.environ["LOCAL_RANK"], "0")
        self.assertEqual(os.environ["WORLD_SIZE"], "4")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "2")
        self.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(os.environ["MASTER_PORT"], "29500")

    def test_single_node_rank0(self):
        args = SimpleNamespace(device="tpu", tp_size=8, nnodes=1, node_rank=0)
        monkey_patch_torch_tpu_env(args, tp_rank=0)
        self.assertEqual(os.environ["RANK"], "0")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "8")
        self.assertEqual(
            os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"], "default_r0"
        )
        self.assertEqual(os.environ["TORCH_TPU_INTERNAL_DETECT_REPEATED_OPS"], "0")

    def test_rank_private_cache_preserves_custom_base(self):
        os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "dflash"
        args = SimpleNamespace(device="tpu", tp_size=4, nnodes=1, node_rank=0)
        monkey_patch_torch_tpu_env(args, tp_rank=3)
        self.assertEqual(
            os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"], "dflash_r3"
        )

    def test_master_addr_port_respect_existing(self):
        os.environ["MASTER_ADDR"] = "10.0.0.1"
        os.environ["MASTER_PORT"] = "12345"
        args = SimpleNamespace(device="tpu", tp_size=1, nnodes=1, node_rank=0)
        monkey_patch_torch_tpu_env(args, tp_rank=0)
        self.assertEqual(os.environ["MASTER_ADDR"], "10.0.0.1")
        self.assertEqual(os.environ["MASTER_PORT"], "12345")


if __name__ == "__main__":
    unittest.main()
