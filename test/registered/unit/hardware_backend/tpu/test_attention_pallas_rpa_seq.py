"""Unit tests for the TPU Pallas rpa_seq kernel wrapper."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_seq import (
    RPA_SEQ_SPEC,
    TpuPallasRpaSeqAttnBackend,
    _invoke_rpa_seq_kernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAttentionPallasRpaSeq(unittest.TestCase):
    def test_registered_factory_builds_rpa_seq_backend(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )

        runner = SimpleNamespace()
        with patch(
            "sglang.srt.hardware_backend.tpu.attention_pallas_rpa_seq."
            "TpuPallasRpaSeqAttnBackend",
            return_value="backend",
        ) as backend_cls:
            backend = ATTENTION_BACKENDS["pallas_tpu_rpa_seq"](runner)

        self.assertEqual(backend, "backend")
        backend_cls.assert_called_once_with(runner)

    def test_backend_uses_rpa_seq_kernel_spec(self):
        self.assertIs(TpuPallasRpaSeqAttnBackend.kernel_spec, RPA_SEQ_SPEC)

    def test_kernel_always_uses_seq_layout(self):
        kernel = Mock(return_value=("output", "cache"))
        cache = SimpleNamespace(shape=(1024, 2, 16, 4, 128))

        with envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(False):
            result = _invoke_rpa_seq_kernel(
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
            num_kv_pages_per_block=32,
            num_queries_per_block=32,
        )

    def test_seq_layout_contract_selects_pool_and_kernel_layout(self):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuSeqAlongLaneKVPool
        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform

        kernel = Mock(return_value=("output", "cache"))
        cache = SimpleNamespace(shape=(1024, 2, 16, 4, 128))
        with (
            envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(False),
            patch(
                "sglang.srt.server_args.get_global_server_args",
                return_value=SimpleNamespace(attention_backend="pallas_tpu_rpa_seq"),
            ),
        ):
            self.assertIs(TpuSRTPlatform().get_mha_kv_pool_cls(), TpuSeqAlongLaneKVPool)
            _invoke_rpa_seq_kernel(
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
                sliding_window=None,
                soft_cap=None,
                k_scale=0.5,
                v_scale=0.25,
            )

        self.assertNotIn("kv_layout", kernel.call_args.kwargs)
        self.assertEqual(kernel.call_args.kwargs["num_kv_pages_per_block"], 32)

    def test_rejects_page_size_that_does_not_divide_tpu7_block(self):
        with self.assertRaisesRegex(ValueError, "page size to divide 4096"):
            _invoke_rpa_seq_kernel(
                Mock(),
                queries=None,
                keys=None,
                values=None,
                kv_cache_fused=SimpleNamespace(shape=(1, 2, 16, 4, 96)),
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
            RPA_SEQ_SPEC.op_name,
            "sglang_tpu_pallas::ragged_paged_attention_seq",
        )
        self.assertEqual(RPA_SEQ_SPEC.extend_distribution_mask, (0, 1, 1))
        self.assertEqual(RPA_SEQ_SPEC.seq_along_lane_min_page_size, 128)
        self.assertTrue(RPA_SEQ_SPEC.seq_along_lane_only)


if __name__ == "__main__":
    unittest.main()
