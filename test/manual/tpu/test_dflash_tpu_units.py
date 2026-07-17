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


if __name__ == "__main__":
    unittest.main()
