"""Regression tests for TPU Pallas-backend registration and pool selection.

The selectable TPU attention backends include:

* ``pallas_tpu_rpa_v3`` (default) — wraps the
  :mod:`sglang.srt.layers.attention.pallas.ragged_paged_attention_v3` kernel
  and needs the fused 5D KV layout from
  :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`.
* ``pallas_tpu_rpa_seq`` — wraps the separately registered sequence-layout
  kernel and uses :class:`TpuSeqAlongLaneKVPool`.
* ``pallas_tpu_rpa_batched`` — wraps the separately registered batched kernel.
* ``pallas_tpu_flash`` — wraps ``flash_attention`` for prefill/extend and
  ``jax.experimental.pallas.ops.tpu.paged_attention`` for decode. Reads
  separate K/V from the standard :class:`MHATokenToKVPool`.

The behaviour being guarded here:

  * Every name is registered with a distinct factory.
  * The TPU platform selects :class:`TpuFusedMHATokenToKVPool` for the RPA
    backends and the standard pool for flash and torch-native attention.

These tests run on CPU (no torch_tpu / JAX needed) — they only exercise
the patch-and-dispatch glue.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import types
import unittest
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase


class TestPallasBackendRegistration(CustomTestCase):
    """Pallas backends must be registered as distinct selectable factories."""

    def test_all_backends_registered_with_distinct_factories(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )

        names = (
            "pallas_tpu_flash",
            "pallas_tpu_rpa_v3",
            "pallas_tpu_rpa_seq",
            "pallas_tpu_rpa_batched",
        )
        factories = [ATTENTION_BACKENDS[name] for name in names]
        self.assertTrue(all(callable(factory) for factory in factories))
        self.assertEqual(len(set(factories)), len(names))

    def test_rpa_seq_is_a_server_arg_choice(self):
        from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

        self.assertIn("pallas_tpu_rpa_seq", ATTENTION_BACKEND_CHOICES)


class TestMHAKVPoolSelection(CustomTestCase):
    def _pool_cls_for(self, attention_backend):
        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform

        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=types.SimpleNamespace(attention_backend=attention_backend),
        ):
            return TpuSRTPlatform().get_mha_kv_pool_cls()

    def test_fused_pool_for_rpa_backends(self):
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.tpu.kv_pool import (
            TpuFusedMHATokenToKVPool,
        )

        with envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(False):
            for backend in ("pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched"):
                with self.subTest(backend=backend):
                    self.assertIs(self._pool_cls_for(backend), TpuFusedMHATokenToKVPool)

    def test_rpa_seq_uses_seq_pool(self):
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuSeqAlongLaneKVPool

        with envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(False):
            self.assertIs(
                self._pool_cls_for("pallas_tpu_rpa_seq"), TpuSeqAlongLaneKVPool
            )

    def test_flat_pool_for_non_rpa_backends(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        for backend in ("pallas_tpu_flash", "torch_native"):
            with self.subTest(backend=backend):
                self.assertIs(self._pool_cls_for(backend), MHATokenToKVPool)


if __name__ == "__main__":
    unittest.main()
