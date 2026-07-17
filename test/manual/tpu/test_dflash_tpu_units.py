"""CPU-runnable unit tests for DFlash-on-TPU host-side helpers.

Covers:
* ``build_dflash_draft_block_allow_mask`` — the ragged all-ones "allow" mask for
  the non-causal draft block (1 == allow), whose semantics are validated against
  the RPA-v3 kernel on TPU in ``test_ragged_paged_attention_v3.py``.
* ``validate_dflash_request`` TPU greedy-only admission — non-greedy
  (temperature/top-k/top-p) requests must be rejected on TPU rather than
  silently degrading to greedy argmax, while greedy (top_k == 1) is admitted.

These need no JAX / TPU device.
"""

import types
import unittest

import torch

import sglang.srt.speculative.dflash_utils as du
from sglang.srt.speculative.dflash_utils import (
    build_dflash_draft_block_allow_mask,
    validate_dflash_request,
)
from sglang.test.test_utils import CustomTestCase


class TestBuildDraftBlockAllowMask(CustomTestCase):
    def test_length_dtype_and_all_ones(self):
        # kv_lens = prefix + block per sequence.
        kv_lens = torch.tensor([21, 8, 80], dtype=torch.int32)
        block_size = 4
        mask = build_dflash_draft_block_allow_mask(kv_lens, block_size)
        self.assertEqual(mask.dtype, torch.int32)
        self.assertEqual(mask.numel(), block_size * int(kv_lens.sum()))
        self.assertTrue(torch.all(mask == 1))

    def test_single_sequence(self):
        mask = build_dflash_draft_block_allow_mask(
            torch.tensor([10], dtype=torch.int32), block_size=4
        )
        self.assertEqual(mask.numel(), 4 * 10)
        self.assertTrue(torch.all(mask == 1))

    def test_rejects_nonpositive_block(self):
        with self.assertRaises(ValueError):
            build_dflash_draft_block_allow_mask(
                torch.tensor([10], dtype=torch.int32), block_size=0
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
