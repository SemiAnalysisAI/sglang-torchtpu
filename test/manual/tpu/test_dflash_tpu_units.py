"""CPU-runnable unit tests for DFlash-on-TPU host-side helpers.

Covers ``validate_dflash_request`` TPU greedy-only admission — non-greedy
(temperature/top-k/top-p) requests must be rejected on TPU rather than silently
degrading to greedy argmax, while greedy (top_k == 1) is admitted. Needs no JAX
/ TPU device. The non-causal draft-block "allow" mask is a fixed-shape all-ones
cache inside the RPA-v3 backend and is validated on TPU in
``test_ragged_paged_attention_v3.py``.
"""

import types
import unittest

import sglang.srt.speculative.dflash_utils as du
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.dflash_worker import (
    resolve_dflash_draft_attention_backend,
)
from sglang.test.test_utils import CustomTestCase


class TestResolveDflashDraftBackend(CustomTestCase):
    """DFlash draft attention backend selection (the ENCODER_ONLY path)."""

    def test_tpu_defaults_to_rpa_v3(self):
        self.assertEqual(
            resolve_dflash_draft_attention_backend(None, is_tpu=True, is_hip=False),
            "pallas_tpu_rpa_v3",
        )

    def test_tpu_explicit_rpa_v3_ok(self):
        self.assertEqual(
            resolve_dflash_draft_attention_backend(
                "pallas_tpu_rpa_v3", is_tpu=True, is_hip=False
            ),
            "pallas_tpu_rpa_v3",
        )

    def test_tpu_rejects_gpu_backends(self):
        for bad in ("flashinfer", "fa3", "fa4", "triton", "trtllm_mha"):
            with self.subTest(backend=bad):
                with self.assertRaises(ValueError):
                    resolve_dflash_draft_attention_backend(
                        bad, is_tpu=True, is_hip=False
                    )

    def test_gpu_defaults_and_fallbacks(self):
        self.assertEqual(
            resolve_dflash_draft_attention_backend(None, is_tpu=False, is_hip=False),
            "flashinfer",
        )
        self.assertEqual(
            resolve_dflash_draft_attention_backend(None, is_tpu=False, is_hip=True),
            "triton",
        )
        # trtllm_mha (causal-only) and unknown backends fall back on GPU.
        self.assertEqual(
            resolve_dflash_draft_attention_backend(
                "trtllm_mha", is_tpu=False, is_hip=False
            ),
            "flashinfer",
        )
        self.assertEqual(
            resolve_dflash_draft_attention_backend(
                "unknown_backend", is_tpu=False, is_hip=False
            ),
            "flashinfer",
        )

    def test_gpu_supported_backends_passthrough(self):
        for good in ("flashinfer", "fa3", "fa4", "triton"):
            with self.subTest(backend=good):
                self.assertEqual(
                    resolve_dflash_draft_attention_backend(
                        good, is_tpu=False, is_hip=False
                    ),
                    good,
                )


def _make_req(top_k: int, return_logprob: bool = False):
    """Minimal Req stand-in exposing only the fields validate_dflash_request reads."""
    sampling_params = types.SimpleNamespace(
        top_k=top_k,
        json_schema=None,
        regex=None,
        ebnf=None,
        structural_tag=None,
    )
    return types.SimpleNamespace(
        return_logprob=return_logprob,
        sampling_params=sampling_params,
    )


class TestValidateDflashRequestTPUGreedyOnly(CustomTestCase):
    def setUp(self):
        # Force the TPU code path regardless of the host running the test.
        self._saved = du._IS_TPU
        du._IS_TPU = True

    def tearDown(self):
        du._IS_TPU = self._saved

    def test_greedy_request_admitted(self):
        # top_k == 1 is greedy after SamplingParams normalization.
        self.assertIsNone(validate_dflash_request(_make_req(top_k=1)))

    def test_topk_sampling_rejected(self):
        err = validate_dflash_request(_make_req(top_k=50))
        self.assertIsInstance(err, str)
        self.assertIn("greedy", err.lower())

    def test_top_k_all_rejected(self):
        # top_k == -1 normalizes to TOP_K_ALL (a large int > 1): a sampling
        # request (any temperature/top_p sampling) -> rejected on TPU.
        from sglang.srt.sampling.sampling_params import TOP_K_ALL

        self.assertIsNotNone(validate_dflash_request(_make_req(top_k=TOP_K_ALL)))

    def test_return_logprob_still_rejected_first(self):
        err = validate_dflash_request(_make_req(top_k=1, return_logprob=True))
        self.assertIn("return_logprob", err)


class TestValidateDflashRequestNonTPU(CustomTestCase):
    def setUp(self):
        self._saved = du._IS_TPU
        du._IS_TPU = False

    def tearDown(self):
        du._IS_TPU = self._saved

    def test_non_tpu_admits_sampling(self):
        # Off TPU, sampling verify is available; non-greedy is not rejected here.
        self.assertIsNone(validate_dflash_request(_make_req(top_k=50)))


class TestDflashKvCellSizing(CustomTestCase):
    """The shared target+draft KV pool must be sized from the DRAFT config, not
    by assuming the draft's per-layer KV bytes equal the target's.
    """

    def test_explicit_draft_cell_size_is_summed(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        got = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000,
            target_num_layers=32,
            draft_num_layers=5,
            draft_cell_size_per_token=200,
        )
        self.assertEqual(got, 1200)

    def test_layer_count_fallback_when_draft_cell_size_none(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        # Layer-count scaling: ceil(1000 * (32+5) / 32) = 1157.
        got = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000,
            target_num_layers=32,
            draft_num_layers=5,
        )
        self.assertEqual(got, 1157)

    def test_wider_draft_not_undersized_vs_layer_count(self):
        # A draft with larger per-token KV (e.g. more KV heads / bigger head_dim)
        # must be sized from its real bytes, which exceed the layer-count estimate
        # that assumes draft==target per-layer.
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        layer_count = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000, target_num_layers=32, draft_num_layers=5
        )
        exact_wide = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000,
            target_num_layers=32,
            draft_num_layers=5,
            draft_cell_size_per_token=2000,  # a wide draft
        )
        self.assertGreater(exact_wide, layer_count)


class TestDflashDraftCellSizeIntegration(CustomTestCase):
    """Integration-level guard for ModelRunner._compute_dflash_draft_cell_size:
    it must produce a real (non-None) draft KV cell size for a mismatched draft,
    and it must run only after the KV cache dtype is resolved. This is what the
    pure-function tests above did NOT catch: an earlier version ran before
    configure_kv_cache_dtype() and silently fell back to None.
    """

    @staticmethod
    def _fake_draft_cfg(kv_heads, head_dim, v_head_dim=None, mla=False):
        return types.SimpleNamespace(
            use_mla_backend=mla,
            head_dim=head_dim,
            v_head_dim=head_dim if v_head_dim is None else v_head_dim,
            get_num_kv_heads=lambda tp, _n=kv_heads: _n,
        )

    def _fake_runner(self, draft_cfg, kv_cache_dtype, num_layers=5):
        return types.SimpleNamespace(
            _dflash_draft_model_config=draft_cfg,
            dflash_draft_num_layers=num_layers,
            kv_cache_dtype=kv_cache_dtype,
            dflash_draft_cell_size_per_token=None,
        )

    def test_computes_real_value_after_dtype_set(self):
        import torch
        from unittest import mock

        from sglang.srt.model_executor.model_runner import ModelRunner

        runner = self._fake_runner(
            self._fake_draft_cfg(kv_heads=8, head_dim=128), torch.bfloat16
        )
        with mock.patch(
            "sglang.srt.layers.dp_attention.get_attention_tp_size", return_value=1
        ):
            ModelRunner._compute_dflash_draft_cell_size(runner)
        # 8 kv heads * (128 + 128) * 5 layers * 2 bytes (bf16).
        self.assertEqual(
            runner.dflash_draft_cell_size_per_token, 8 * (128 + 128) * 5 * 2
        )

    def test_bug_regression_requires_dtype_resolved(self):
        # Running before configure_kv_cache_dtype() (kv_cache_dtype unset) must
        # fail loudly rather than silently fall back to None -- guarding the
        # init-order fix.
        from unittest import mock

        from sglang.srt.model_executor.model_runner import ModelRunner

        runner = self._fake_runner(
            self._fake_draft_cfg(kv_heads=8, head_dim=128), kv_cache_dtype=None
        )
        with mock.patch(
            "sglang.srt.layers.dp_attention.get_attention_tp_size", return_value=1
        ):
            with self.assertRaises(Exception):
                ModelRunner._compute_dflash_draft_cell_size(runner)

    def test_mla_draft_left_none(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        runner = self._fake_runner(
            self._fake_draft_cfg(kv_heads=8, head_dim=128, mla=True),
            kv_cache_dtype=None,
        )
        ModelRunner._compute_dflash_draft_cell_size(runner)
        self.assertIsNone(runner.dflash_draft_cell_size_per_token)


class TestDflashVerifyMaskPolicyAvoidsTriton(CustomTestCase):
    """Verify-routing guard: on the TPU RPA-v3 backend, DFlash verify skips its custom mask,
    so it uses RPA-v3's own metadata builder and never reaches
    DFlashVerifyInput.generate_attn_arg_prefill (the only caller of the Triton
    create_flashinfer_kv_indices_triton kernel).
    """

    @staticmethod
    def _named(name):
        cls = type(name, (), {})
        return cls()

    def test_rpa_v3_backend_skips_custom_mask(self):
        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_verify_mask_policy,
        )

        backend = self._named("TpuPallasRpaV3AttnBackend")
        backend_name, build_custom_mask = resolve_dflash_verify_mask_policy(backend)
        self.assertEqual(backend_name, "TpuPallasRpaV3AttnBackend")
        self.assertFalse(
            build_custom_mask,
            "RPA-v3 verify must skip the DFlash custom mask so it uses RPA-v3's "
            "own metadata (not the flashinfer/Triton kv-indices path).",
        )

    def test_unknown_backend_still_builds_custom_mask(self):
        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_verify_mask_policy,
        )

        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self._named("SomeGpuAttnBackend")
        )
        self.assertTrue(build_custom_mask)

    def test_triton_kv_indices_only_reachable_via_generate_attn_arg_prefill(self):
        # The Triton kernel is only invoked from generate_attn_arg_prefill, so
        # skipping custom mask + using RPA-v3 metadata provably avoids it.
        import inspect

        import sglang.srt.speculative.dflash_info as di

        src = inspect.getsource(di)
        # It is imported and called exactly once, inside generate_attn_arg_prefill.
        self.assertEqual(src.count("create_flashinfer_kv_indices_triton["), 1)
        gen_src = inspect.getsource(di.DFlashVerifyInput.generate_attn_arg_prefill)
        self.assertIn("create_flashinfer_kv_indices_triton[", gen_src)


if __name__ == "__main__":
    unittest.main()
