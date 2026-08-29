# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable integration tests for the stacked-RPA decode backend."""

import types
import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")


class TestStackedRpaRegistration(CustomTestCase):
    def test_backend_is_registered(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )

        self.assertIn("pallas_tpu_rpa_stacked", ATTENTION_BACKENDS)
        self.assertTrue(callable(ATTENTION_BACKENDS["pallas_tpu_rpa_stacked"]))

    def test_backend_declares_decode_only_seq_layout(self):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_stacked import (
            RPA_STACKED_SPEC,
        )

        self.assertEqual(RPA_STACKED_SPEC.extend_distribution_mask, (0, 0, 0))
        self.assertIsNone(RPA_STACKED_SPEC.min_page_size)
        self.assertEqual(RPA_STACKED_SPEC.seq_along_lane_min_page_size, 128)
        self.assertTrue(RPA_STACKED_SPEC.seq_along_lane_only)
        self.assertEqual(RPA_STACKED_SPEC.kv_cache_rank, 4)
        self.assertEqual(
            RPA_STACKED_SPEC.import_kernel().__module__,
            "sglang.srt.layers.attention.pallas.stacked_rpa.wrapper",
        )

    def test_backend_requires_rpa_seq_prefill(self):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
            BaseTpuPallasRpaAttnBackend,
        )
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_stacked import (
            TpuPallasRpaStackedAttnBackend,
        )

        runner = types.SimpleNamespace(
            prefill_attention_backend_str="pallas_tpu_rpa_v3"
        )
        with self.assertRaisesRegex(ValueError, "pallas_tpu_rpa_seq"):
            TpuPallasRpaStackedAttnBackend(runner)

        runner.prefill_attention_backend_str = "pallas_tpu_rpa_seq"
        with patch.object(BaseTpuPallasRpaAttnBackend, "__init__", return_value=None):
            TpuPallasRpaStackedAttnBackend(runner)

    def test_backend_rejects_stacked_prefill_configuration(self):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_stacked import (
            TpuPallasRpaStackedAttnBackend,
        )

        runner = types.SimpleNamespace(
            prefill_attention_backend_str="pallas_tpu_rpa_stacked"
        )
        with self.assertRaisesRegex(ValueError, "decode-only"):
            TpuPallasRpaStackedAttnBackend(runner)

    def test_backend_rejects_prefill(self):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_stacked import (
            TpuPallasRpaStackedAttnBackend,
        )

        backend = object.__new__(TpuPallasRpaStackedAttnBackend)
        with self.assertRaisesRegex(NotImplementedError, "decode-only"):
            backend.forward_extend(None, None, None, None, None)


class TestStackedRpaSharedCache(CustomTestCase):
    def test_rpa_seq_prefill_selects_native_rank4_pool(self):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuRpaSeqKVPool
        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform

        args = types.SimpleNamespace(
            attention_backend="pallas_tpu_rpa_seq",
            decode_attention_backend="pallas_tpu_rpa_stacked",
        )
        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=args,
        ):
            pool_cls = TpuSRTPlatform().get_mha_kv_pool_cls()

        self.assertIs(pool_cls, TpuRpaSeqKVPool)

    def test_native_pool_uses_rank4_shape(self):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuRpaSeqKVPool

        pool = object.__new__(TpuRpaSeqKVPool)
        pool.num_blocks = 3
        pool.head_num = 2
        pool.page_size = 128
        pool._head_dim_lanes = Mock(return_value=192)

        self.assertEqual(pool._buffer_shape(), (3, 4, 192, 128))


if __name__ == "__main__":
    unittest.main()
