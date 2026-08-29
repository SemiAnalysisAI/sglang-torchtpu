"""Unit tests for the TPU Pallas rpa_seq kernel wrapper."""

import importlib.util
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
        cache = SimpleNamespace(shape=(1024, 2, 64, 128))

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
                attention_sink=None,
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
            None,
            sm_scale=0.125,
            sliding_window=2047,
            soft_cap=None,
            k_scale=0.5,
            v_scale=0.25,
        )

    def test_seq_layout_contract_selects_pool_and_kernel_layout(self):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuRpaSeqKVPool
        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform

        kernel = Mock(return_value=("output", "cache"))
        cache = SimpleNamespace(shape=(1024, 2, 64, 128))
        with (
            envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV.override(False),
            patch(
                "sglang.srt.server_args.get_global_server_args",
                return_value=SimpleNamespace(
                    attention_backend="pallas_tpu_rpa_seq",
                ),
            ),
        ):
            self.assertIs(TpuSRTPlatform().get_mha_kv_pool_cls(), TpuRpaSeqKVPool)
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
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=None,
                soft_cap=None,
                k_scale=0.5,
                v_scale=0.25,
            )

        self.assertNotIn("kv_layout", kernel.call_args.kwargs)

    def test_rank5_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "native rank-4"):
            _invoke_rpa_seq_kernel(
                Mock(),
                queries=None,
                keys=None,
                values=None,
                kv_cache_fused=SimpleNamespace(shape=(8, 4, 16, 4, 128)),
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

    def test_rejects_page_size_that_is_not_lane_aligned(self):
        with self.assertRaisesRegex(ValueError, "positive multiple of 128"):
            _invoke_rpa_seq_kernel(
                Mock(),
                queries=None,
                keys=None,
                values=None,
                kv_cache_fused=SimpleNamespace(shape=(1, 2, 64, 96)),
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
            "sglang_tpu_pallas::ragged_paged_attention_seq_prefill_v1",
        )
        self.assertEqual(RPA_SEQ_SPEC.extend_distribution_mask, (0, 1, 1))
        self.assertEqual(RPA_SEQ_SPEC.seq_along_lane_min_page_size, 128)
        self.assertTrue(RPA_SEQ_SPEC.seq_along_lane_only)
        self.assertFalse(RPA_SEQ_SPEC.supports_attention_sink)
        self.assertEqual(RPA_SEQ_SPEC.kv_cache_rank, 4)

    @unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is not installed")
    def test_specialized_kernel_declares_rank4_cache(self):
        from sglang.srt.layers.attention.pallas.ragged_paged_attention_seq import (
            kernel,
        )

        with patch.object(
            kernel.pltpu,
            "get_tpu_info",
            return_value=SimpleNamespace(num_sublanes=8),
        ):
            shape = kernel.get_kv_cache_shape(4, 512, 1, 64, kernel.jnp.float8_e4m3fn)

        self.assertEqual(shape, (4, 2, 64, 512))

    @unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is not installed")
    def test_prefill_uses_the_shape_tuned_tile(self):
        from sglang.srt.layers.attention.pallas.ragged_paged_attention_seq import (
            tuned_block_sizes,
        )

        with patch.object(
            tuned_block_sizes.tpu_info,
            "get_device_kind",
            return_value="TPU7x",
        ):
            blocks = tuned_block_sizes.resolve_block_sizes(
                tuned_block_sizes.jnp.bfloat16,
                tuned_block_sizes.jnp.float8_e4m3fn,
                8,
                1,
                64,
                512,
                8192,
                256,
            )

        self.assertEqual(blocks, (1, 256))

    def test_decode_fails_loudly(self):
        backend = object.__new__(TpuPallasRpaSeqAttnBackend)
        with self.assertRaisesRegex(NotImplementedError, "prefill-only"):
            backend.forward_decode()

    def test_backend_requires_stacked_decode(self):
        runner = SimpleNamespace(decode_attention_backend_str="pallas_tpu_rpa_v3")
        with self.assertRaisesRegex(ValueError, "prefill-only"):
            TpuPallasRpaSeqAttnBackend(runner)


if __name__ == "__main__":
    unittest.main()
