"""Unit tests for the TPU Pallas RPA v4 kernel wrapper."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v4 import (
    RPA_V4_SPEC,
    _invoke_v4_kernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAttentionPallasRpaV4(unittest.TestCase):
    def test_kernel_invocation_adapts_metadata_and_tpu7_blocks(self):
        kernel = Mock(return_value=("output", "cache"))
        cache = SimpleNamespace(shape=(1024, 64, 1, 4, 128))

        result = _invoke_v4_kernel(
            kernel,
            queries="q",
            keys="k",
            values="v",
            kv_cache_fused=cache,
            kv_lens="kv_lens",
            page_indices="pages",
            cu_q_lens="cu_q",
            cu_kv_lens="unused",
            distribution="distribution",
            attention_sink="sink",
            sm_scale=0.125,
            sliding_window=2047,
            soft_cap=None,
            k_scale=0.5,
            v_scale=0.25,
        )

        self.assertEqual(result, ("output", "cache"))
        kernel.assert_called_once_with(
            "q",
            "k",
            "v",
            cache,
            "kv_lens",
            "pages",
            "cu_q",
            "distribution",
            "sink",
            sm_scale=0.125,
            sliding_window=2047,
            soft_cap=None,
            k_scale=0.5,
            v_scale=0.25,
            num_kv_pages_per_block=64,
            num_queries_per_block=32,
        )

    def test_rejects_page_size_that_does_not_divide_tpu7_block(self):
        with self.assertRaisesRegex(ValueError, "page size to divide 4096"):
            _invoke_v4_kernel(
                Mock(),
                queries=None,
                keys=None,
                values=None,
                kv_cache_fused=SimpleNamespace(shape=(1, 96, 1, 4, 128)),
                kv_lens=None,
                page_indices=None,
                cu_q_lens=None,
                cu_kv_lens=None,
                distribution=None,
                attention_sink=None,
                sm_scale=1.0,
                sliding_window=None,
                soft_cap=None,
                k_scale=None,
                v_scale=None,
            )

    def test_spec_uses_unique_custom_op(self):
        self.assertEqual(
            RPA_V4_SPEC.op_name,
            "sglang_tpu_pallas::ragged_paged_attention_v4",
        )
        self.assertEqual(RPA_V4_SPEC.extend_distribution_mask, (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
