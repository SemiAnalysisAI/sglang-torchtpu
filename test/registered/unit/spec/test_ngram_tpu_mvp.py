"""Unit tests for the NGram-on-TPU linear-greedy MVP.

These cover the device-agnostic pieces that run on CPU without TPU hardware:
  - the speculative-hook capability gate for the TPU envelope, and
  - the torch-native linear (breadth=1) reconstruct layout plus its equivalence
    with the torch-native greedy verifier that consumes it.

End-to-end greedy correctness on real TPU hardware is covered separately by the
server-level spec tests; those require a TPU device + model and are not part of
this CPU unit module.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_utils import _verify_tree_greedy_torch


class TestNgramTpuCapabilityGate(unittest.TestCase):
    def _make_args(self, **overrides):
        from sglang.srt.server_args import ServerArgs

        args = SimpleNamespace(
            device="tpu",
            enable_torch_compile=False,
            attention_backend=None,
            page_size=1,
            speculative_ngram_max_bfs_breadth=ServerArgs.speculative_ngram_max_bfs_breadth,
            speculative_ngram_min_bfs_breadth=ServerArgs.speculative_ngram_min_bfs_breadth,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _validate(self, args):
        from sglang.srt.arg_groups.speculative_hook import (
            _validate_ngram_device_capability,
        )

        _validate_ngram_device_capability(args)

    def test_cuda_passes_unchanged(self):
        self._validate(self._make_args(device="cuda"))

    def test_tpu_default_breadth_coerced_to_one(self):
        args = self._make_args()
        self._validate(args)
        self.assertEqual(args.speculative_ngram_max_bfs_breadth, 1)
        self.assertEqual(args.speculative_ngram_min_bfs_breadth, 1)

    def test_tpu_breadth_one_accepted(self):
        args = self._make_args(speculative_ngram_max_bfs_breadth=1)
        self._validate(args)
        self.assertEqual(args.speculative_ngram_max_bfs_breadth, 1)

    def test_tpu_explicit_breadth_gt_one_rejected(self):
        args = self._make_args(speculative_ngram_max_bfs_breadth=4)
        with self.assertRaises(ValueError):
            self._validate(args)

    def test_tpu_non_rpa_v3_backend_rejected(self):
        args = self._make_args(attention_backend="pallas_tpu_flash")
        with self.assertRaises(ValueError):
            self._validate(args)

    def test_tpu_rpa_v3_backend_accepted(self):
        self._validate(self._make_args(attention_backend="pallas_tpu_rpa_v3"))

    def test_tpu_torch_compile_rejected(self):
        args = self._make_args(enable_torch_compile=True)
        with self.assertRaises(ValueError):
            self._validate(args)

    def test_tpu_page_size_gt_one_rejected(self):
        args = self._make_args(page_size=16)
        with self.assertRaises(ValueError):
            self._validate(args)

    def test_unsupported_device_rejected(self):
        args = self._make_args(device="cpu")
        with self.assertRaises(ValueError):
            self._validate(args)


class TestNgramLinearReconstruct(unittest.TestCase):
    def _reconstruct(self, bs, n, seq_lens):
        try:
            from sglang.srt.speculative.ngram_worker import NGRAMWorker
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"ngram_worker import unavailable: {exc}")

        worker = NGRAMWorker.__new__(NGRAMWorker)
        worker.draft_token_num = n
        worker.device = "cpu"

        positions = torch.empty((bs * n,), dtype=torch.int64)
        retrieve_index = torch.empty((bs, n), dtype=torch.int64)
        retrieve_next_token = torch.empty((bs, n), dtype=torch.int64)
        retrieve_next_sibling = torch.empty((bs, n), dtype=torch.int64)
        batch = SimpleNamespace(seq_lens=torch.tensor(seq_lens, dtype=torch.int64))

        worker._reconstruct_indices_linear_torch(
            batch,
            bs,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        )
        return positions, retrieve_index, retrieve_next_token, retrieve_next_sibling

    def test_layout_matches_linear_formula(self):
        bs, n = 3, 4
        seq_lens = [5, 7, 2]
        positions, ri, rnt, rns = self._reconstruct(bs, n, seq_lens)

        for i in range(bs):
            for j in range(n):
                # Global flattened retrieve index.
                self.assertEqual(ri[i, j].item(), i * n + j)
                # Local next-token chain, -1 at the tail.
                expected_next = j + 1 if j + 1 < n else -1
                self.assertEqual(rnt[i, j].item(), expected_next)
                # No siblings in a linear chain.
                self.assertEqual(rns[i, j].item(), -1)
                # Absolute positions.
                self.assertEqual(positions[i * n + j].item(), seq_lens[i] + j)

    def test_single_draft_token(self):
        bs, n = 2, 1
        positions, ri, rnt, rns = self._reconstruct(bs, n, [3, 9])
        self.assertEqual(rnt.flatten().tolist(), [-1, -1])
        self.assertEqual(rns.flatten().tolist(), [-1, -1])
        self.assertEqual(ri.flatten().tolist(), [0, 1])
        self.assertEqual(positions.tolist(), [3, 9])


class TestNgramLinearVerifyEquivalence(unittest.TestCase):
    """The linear reconstruct must feed the torch-native greedy verifier so that
    accepted positions select the correct GLOBAL predictions (i*n + j)."""

    def test_partial_and_zero_accept(self):
        bs, n = 2, 4
        # Linear reconstruct: retrieve_index holds global flattened indices.
        retrieve_index = torch.arange(bs * n, dtype=torch.int64).view(bs, n)

        candidates = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int64
        )
        # Greedy linear accept: draft j+1 accepted iff target_predict[:, j] == candidates[:, j+1].
        # req0 accepts drafts 1,2 (then mismatch); req1 accepts none.
        target_predict = torch.tensor(
            [[11, 12, 99, 99], [99, 99, 99, 99]], dtype=torch.int64
        )

        predicts = torch.empty((bs * n,), dtype=torch.int32)
        accept_index = torch.full((bs, n), -1, dtype=torch.int32)
        accept_token_num = torch.empty((bs,), dtype=torch.int32)

        _verify_tree_greedy_torch(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            target_predict,
            topk=1,
        )

        # num_correct_drafts (excludes the bonus token).
        self.assertEqual(accept_token_num.tolist(), [2, 0])
        # Accepted global indices, -1 padded; req1 still keeps its bonus slot (4).
        self.assertEqual(accept_index[0].tolist(), [0, 1, 2, -1])
        self.assertEqual(accept_index[1].tolist(), [4, -1, -1, -1])
        # predicts mirror the flattened target predictions.
        self.assertEqual(predicts[0:4].tolist(), [11, 12, 99, 99])

    def test_full_accept(self):
        bs, n = 1, 3
        retrieve_index = torch.arange(bs * n, dtype=torch.int64).view(bs, n)
        candidates = torch.tensor([[5, 6, 7]], dtype=torch.int64)
        # Every draft matches the target's prediction at the previous position.
        target_predict = torch.tensor([[6, 7, 42]], dtype=torch.int64)

        predicts = torch.empty((bs * n,), dtype=torch.int32)
        accept_index = torch.full((bs, n), -1, dtype=torch.int32)
        accept_token_num = torch.empty((bs,), dtype=torch.int32)

        _verify_tree_greedy_torch(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            target_predict,
            topk=1,
        )

        # All n-1 drafts correct; bonus token is the n-th.
        self.assertEqual(accept_token_num.tolist(), [n - 1])
        self.assertEqual(accept_index[0].tolist(), [0, 1, 2])


class TestNgramTpuAdmissionRejection(unittest.TestCase):
    """Per-request, admission-time rejection of non-greedy NGram-on-TPU requests."""

    def _reject(self, top_k, device="tpu", algo="NGRAM"):
        try:
            from sglang.srt.managers.tokenizer_manager import TokenizerManager
            from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"tokenizer_manager import unavailable: {exc}")

        # Mirror the static flag TokenizerManager.__init__ caches once at startup.
        greedy_only = (
            device == "tpu"
            and algo is not None
            and SpeculativeAlgorithm.from_string(algo).is_ngram()
        )
        fake_self = SimpleNamespace(_tpu_ngram_greedy_only=greedy_only)
        sampling_params = SimpleNamespace(top_k=top_k)
        TokenizerManager._reject_unsupported_tpu_ngram_sampling(
            fake_self, sampling_params
        )

    def test_non_greedy_rejected_on_tpu_ngram(self):
        with self.assertRaises(ValueError):
            self._reject(top_k=50)

    def test_greedy_allowed_on_tpu_ngram(self):
        self._reject(top_k=1)

    def test_non_greedy_allowed_on_cuda(self):
        self._reject(top_k=50, device="cuda")

    def test_non_greedy_allowed_without_ngram(self):
        self._reject(top_k=50, algo="EAGLE")

    def test_no_spec_algo_noop(self):
        self._reject(top_k=50, algo=None)


class TestNgramTpuKernelRouting(unittest.TestCase):
    """AC-2 runtime guard: on the TPU (torch-native) branch, the cache-pool
    dispatch uses the torch-native implementation and never launches the raw
    Triton kernel."""

    def test_assign_req_to_token_pool_func_skips_raw_triton_on_tpu(self):
        from sglang.srt.speculative import spec_utils

        if not spec_utils._use_torch_native:
            self.skipTest("not a torch-native (TPU) build")

        raw_called = {"hit": False}
        original = spec_utils.assign_req_to_token_pool

        def _boom(*args, **kwargs):
            raw_called["hit"] = True
            raise AssertionError("raw Triton assign_req_to_token_pool launched on TPU")

        spec_utils.assign_req_to_token_pool = _boom
        try:
            bs = 2
            req_to_token = torch.zeros((4, 8), dtype=torch.int32)
            req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
            start_offset = torch.tensor([0, 0], dtype=torch.int64)
            end_offset = torch.tensor([2, 3], dtype=torch.int64)
            out_cache_loc = torch.arange(5, dtype=torch.int32)

            spec_utils.assign_req_to_token_pool_func(
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
                bs,
            )
        finally:
            spec_utils.assign_req_to_token_pool = original

        self.assertFalse(raw_called["hit"])
        # The torch-native path must have written the cache locations correctly.
        self.assertEqual(req_to_token[0, 0:2].tolist(), [0, 1])
        self.assertEqual(req_to_token[1, 0:3].tolist(), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
