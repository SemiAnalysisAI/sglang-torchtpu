# SPDX-License-Identifier: Apache-2.0
"""CPU verification of live-row speculative features and preserved PP selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import compute_spec_logprobs_from_tensors
from sglang.srt.sampling.sampling_batch_info import ProcessorEntry
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.experimental.speculative import base_spec_worker
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.experimental.speculative.spec_ops import apply_token_bitmask
from sglang_tpu.experimental.speculative.spec_types import (
    BucketState,
    DraftResult,
    SpecState,
)
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def info(rows=1, physical=2, vocab=5, **kwargs):
    values = dict(
        temperatures=torch.ones(physical, 1),
        top_ps=torch.ones(physical),
        top_ks=torch.ones(physical, dtype=torch.int32),
        min_ps=torch.zeros(physical),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=vocab,
        device="cpu",
        host_pad_bs=lambda n: physical,
        real_batch_size=rows,
    )
    values.update(kwargs)
    return HostPaddedSamplingInfo(**values)


class PackedChainGrammar:
    """A small packed-mask FSM implementing the production grammar contract."""

    def __init__(self):
        self.tokens = []
        self.finished = False

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        return torch.full((batch_size, 1), -1, dtype=torch.int32)

    def fill_vocab_mask(self, mask, idx):
        mask[idx, 0] = 1 << (len(self.tokens) + 1)

    def accept_token(self, token):
        if token != len(self.tokens) + 1:
            raise ValueError(token)
        self.tokens.append(token)

    def rollback(self, n):
        del self.tokens[-n:]

    def is_terminated(self):
        return len(self.tokens) >= 3

    def apply_vocab_mask(self, logits, vocab_mask):
        bits = torch.arange(logits.shape[-1])
        logits.masked_fill_(((vocab_mask.long() >> bits) & 1) == 0, -float("inf"))


class CompiledKernelGrammar(PackedChainGrammar):
    """The backends' own TPU apply is a torch.compile kernel nothing warms."""

    def apply_vocab_mask(self, logits, vocab_mask):
        raise AssertionError("verify must mask through the warmed plugin op")


def _reference_bitmask(logits, bitmask):
    expected = logits.clone()
    for row in range(logits.shape[0]):
        for token in range(min(logits.shape[1], 32 * bitmask.shape[1])):
            word = int(bitmask[row, token // 32]) & 0xFFFFFFFF
            if not (word >> (token % 32)) & 1:
                expected[row, token] = -float("inf")
    return expected


class TestSharedFeatures(CustomTestCase):
    def worker(self, logits):
        worker = object.__new__(TpuBlockSpecWorker)
        worker.num_drafts = 2
        worker.num_verify_rows = 3
        worker.device = "cpu"
        worker._accept_threshold_single = worker._accept_threshold_acc = 1.0
        worker._simulate_acc_len = 0
        worker._tp_sync = SimpleNamespace(sync=lambda site, value: value)
        worker._forward_target_verify = Mock(
            return_value=LogitsProcessorOutput(next_token_logits=logits)
        )
        return worker

    def bucket(self):
        state = SpecState(
            torch.tensor([0, 0]), torch.tensor([10, 0]), torch.tensor([2, 0])
        )
        return BucketState(
            torch.tensor([2]), state, torch.tensor([True, False]), None, None
        )

    def drafts(self):
        return DraftResult(torch.tensor([[1, 2], [0, 0]]), None, torch.tensor([2, 2]))

    def test_custom_penalty_scaling_and_bias_order_at_live_extent(self):
        logits = torch.tensor([[4.0, -2.0, 1.0, 0.0, 0.0]]).repeat(6, 1)
        seen = []

        def processor(values, params):
            seen.append((values.shape, params))
            return values + 1

        sampling = info(
            has_custom_logit_processor=True,
            custom_params=[{"k": 7}],
            custom_logit_processor={
                0: ProcessorEntry(processor, [0], torch.tensor([0]))
            },
            acc_additive_penalties=torch.tensor([[-1.0, -1.0, -1.0, 0.0, 0.0]]),
            acc_scaling_penalties=torch.tensor([[2.0, 2.0, 2.0, 1.0, 1.0]]),
            logit_bias=torch.tensor([[0.0, 0.0, 4.0, 0.0, 0.0]]),
        )
        output, accepted = self.worker(logits).verify(
            SimpleNamespace(has_grammar=False, sampling_info=sampling),
            self.bucket(),
            None,
            self.drafts(),
            None,
        )
        self.assertEqual(seen, [(torch.Size([3, 5]), [{"k": 7}] * 3)])
        torch.testing.assert_close(
            output.next_token_logits[:3],
            torch.tensor([[2.0, -4.0, 4.5, 1.0, 1.0]]).repeat(3, 1),
        )
        torch.testing.assert_close(output.next_token_logits[3:], logits[3:])
        self.assertEqual(accepted.accept_tokens[0, 0].item(), 2)
        self.assertEqual(accepted.accept_lens.tolist(), [1, 0])

    def test_grammar_barrier_precedes_traversal_and_padding_is_neutral(self):
        grammar = PackedChainGrammar()
        sampling = info()
        logits = torch.zeros(6, 5)
        logits[:, 4] = 100
        worker = self.worker(logits)
        order = []
        worker._forward_target_verify.side_effect = lambda **kw: (
            order.append("target") or LogitsProcessorOutput(next_token_logits=logits)
        )

        def barrier():
            order.append("barrier")
            self.assertEqual(grammar.tokens, [])

        output, accepted = worker.verify(
            SimpleNamespace(
                has_grammar=True,
                reqs=[SimpleNamespace(grammar=grammar)],
                sampling_info=sampling,
            ),
            self.bucket(),
            None,
            self.drafts(),
            None,
            grammar_barrier=barrier,
        )
        self.assertEqual(order, ["target", "barrier"])
        self.assertEqual(
            grammar.tokens, []
        )  # DFS rollback; result processing owns commits.
        self.assertEqual(accepted.accept_tokens[0].tolist(), [1, 2, 3])
        self.assertEqual(accepted.accept_lens.tolist(), [3, 0])
        self.assertTrue(torch.isneginf(output.next_token_logits[0, 4]))
        torch.testing.assert_close(output.next_token_logits[3:], logits[3:])

    def test_verify_masks_packed_grammar_bitmasks_through_the_warmed_op(self):
        grammar = CompiledKernelGrammar()
        logits = torch.zeros(6, 5)
        logits[:, 4] = 100
        output, accepted = self.worker(logits).verify(
            SimpleNamespace(
                has_grammar=True,
                reqs=[SimpleNamespace(grammar=grammar)],
                sampling_info=info(),
            ),
            self.bucket(),
            None,
            self.drafts(),
            None,
        )
        self.assertEqual(accepted.accept_tokens[0].tolist(), [1, 2, 3])
        self.assertTrue(torch.isneginf(output.next_token_logits[0, 4]))
        torch.testing.assert_close(output.next_token_logits[3:], logits[3:])

    def test_packed_bitmask_follows_the_xgrammar_bit_rule(self):
        generator = torch.Generator().manual_seed(5)
        # Two words with the sign bit set; logits wider and narrower than 64 columns.
        bitmask = torch.randint(
            -(2**31), 2**31 - 1, (4, 2), dtype=torch.int32, generator=generator
        )
        bitmask[0] = -1
        bitmask[1, 1] = -(2**31)
        for vocab in (70, 64, 50):
            logits = torch.randn(4, vocab, generator=generator)
            actual = apply_token_bitmask(logits, bitmask)
            self.assertEqual(actual.shape, logits.shape)
            torch.testing.assert_close(actual, _reference_bitmask(logits, bitmask))

    def test_bitmask_apply_is_warmed_at_every_verify_bucket(self):
        worker = object.__new__(TpuBlockSpecWorker)
        worker.num_verify_rows = 3
        worker.device = "cpu"
        worker._b_bucketer = lambda: SimpleNamespace(buckets=[2, 4])
        worker._warmup_seeded_accept = lambda pbs, vocab_size: None
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(vocab_size=40))
        )
        warmed = []
        with patch.object(
            base_spec_worker,
            "warmup_compiled_op",
            side_effect=lambda op, *args, **kwargs: warmed.append((op, args)),
        ):
            worker._warmup_verify_logit_modifiers()
        shapes = [
            (tuple(args[0].shape), tuple(args[1].shape), args[1].dtype)
            for op, args in warmed
            if op is apply_token_bitmask
        ]
        self.assertEqual(
            shapes,
            [((6, 40), (6, 2), torch.int32), ((12, 40), (12, 2), torch.int32)],
        )

    def test_logprobs_use_live_temperatures_and_include_bonus(self):
        logits = torch.arange(30, dtype=torch.float32).reshape(6, 5).flip(0)
        tokens = torch.tensor([1, 2, 3, 4, 0, 1])
        output = LogitsProcessorOutput(next_token_logits=logits)
        temperatures = torch.tensor([[0.5], [2.0]])
        compute_spec_logprobs_from_tensors(
            output,
            tokens,
            batch_size=2,
            temperatures=temperatures,
            is_all_greedy=False,
            top_logprobs_nums=[2, 1],
            token_ids_logprobs=[[0, 3], [1]],
            chain_stride=3,
        )
        expected = (logits / temperatures.repeat_interleave(3, dim=0)).log_softmax(-1)
        torch.testing.assert_close(
            output.next_token_logprobs, expected.gather(1, tokens[:, None]).view(2, 3)
        )
        self.assertEqual(
            [len(row) for row in output.next_token_top_logprobs_val], [2] * 3 + [1] * 3
        )
        torch.testing.assert_close(
            output.next_token_token_ids_logprobs_val[2], expected[2, [0, 3]]
        )

    def test_dflash_shared_adjustments_keep_sign_sensitive_repetition(self):
        sampling = info(
            rows=1,
            physical=1,
            vocab=3,
            acc_additive_penalties=torch.tensor([[-1.0, -1.0, 0.0]]),
            acc_scaling_penalties=torch.tensor([[2.0, 2.0, 1.0]]),
            logit_bias=torch.tensor([[0.0, 0.0, 3.0]]),
        )
        logits = torch.tensor([[5.0, -1.0, 0.0]]).repeat(3, 1)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits, sampling_info=sampling, draft_token_num=3
        )
        torch.testing.assert_close(
            logits, torch.tensor([[2.0, -4.0, 3.0]]).repeat(3, 1)
        )

    def test_seeded_target_sampler_uses_live_positions_with_physical_padding(self):
        from sglang.srt.runtime_context import get_context
        from sglang_tpu.sampler import TpuSampler
        from sglang_tpu.sampling.seeded import sample_with_seed

        with get_context().override_server_args(sampling_backend="pytorch"):
            with (
                patch(
                    "sglang.srt.layers.sampler.get_parallel",
                    return_value=SimpleNamespace(
                        tp_group=SimpleNamespace(device_group=None)
                    ),
                ),
                patch(
                    "sglang.srt.layers.sampler.is_dp_attention_enabled",
                    return_value=False,
                ),
            ):
                sampler = TpuSampler()
            sampler._sync_token_ids_across_tp = lambda *args: None
            sampling = info(
                rows=2,
                physical=4,
                vocab=3,
                temperatures=torch.tensor([[0.5], [1.5], [1.0], [1.0]]),
                top_ks=torch.full((4,), 1 << 30),
                is_all_greedy=False,
                is_any_greedy=False,
                need_top_k_sampling=False,
                sampling_seed=torch.tensor([123, 456, 0, 0]),
            )
            positions = torch.tensor([7, 31])
            logits = torch.tensor([[1.0, 2.0, 0.0], [3.0, 1.0, 2.0]])
            padded = torch.nn.functional.pad(logits, (0, 0, 0, 2))
            expected = sample_with_seed(
                (padded / sampling.temperatures).softmax(-1),
                sampling.sampling_seed,
                torch.tensor([7, 31, 0, 0]),
            )
            results = []
            with torch.random.fork_rng():
                for ambient_seed in (1, 789):
                    torch.manual_seed(ambient_seed)
                    before = torch.random.get_rng_state()
                    out = LogitsProcessorOutput(next_token_logits=logits.clone())
                    results.append(
                        sampler(out, sampling, True, [0, 0], [None, None], positions)
                    )
                    self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                    self.assertEqual(out.next_token_logprobs.shape, (4,))
            for result in results:
                torch.testing.assert_close(result, expected.to(torch.int64))

    def test_seeded_bucket_one_params_sample_padded_extend_rows(self):
        from sglang.srt.runtime_context import get_context
        from sglang_tpu.sampler import TpuSampler
        from sglang_tpu.sampling.seeded import sample_with_seed

        with get_context().override_server_args(sampling_backend="pytorch"):
            with (
                patch(
                    "sglang.srt.layers.sampler.get_parallel",
                    return_value=SimpleNamespace(
                        tp_group=SimpleNamespace(device_group=None)
                    ),
                ),
                patch(
                    "sglang.srt.layers.sampler.is_dp_attention_enabled",
                    return_value=False,
                ),
            ):
                sampler = TpuSampler()
            sampler._sync_token_ids_across_tp = lambda *args: None
            # A one-request extend whose target keeps its outputs padded to the
            # extend cap: bucket-1 params, four logits and position rows.
            sampling = info(
                rows=1,
                physical=1,
                vocab=3,
                temperatures=torch.tensor([[0.5]]),
                top_ks=torch.full((1,), 1 << 30),
                is_all_greedy=False,
                is_any_greedy=False,
                need_top_k_sampling=False,
                sampling_seed=torch.tensor([123]),
            )
            positions = torch.tensor([7, 0, 0, 0])
            logits = torch.tensor([[1.0, 2.0, 0.0]]).repeat(4, 1)
            expected = sample_with_seed(
                (logits / 0.5).softmax(-1), torch.tensor([123, 0, 0, 0]), positions
            )
            out = LogitsProcessorOutput(next_token_logits=logits.clone())
            result = sampler(out, sampling, False, None, None, positions)
            self.assertEqual(result.shape, (4,))
            self.assertEqual(int(result[0]), int(expected[0]))

    def test_pipeline_opt_in_keeps_existing_worker_selection(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True):
            for algorithm in (SpeculativeAlgorithm.EAGLE3, SpeculativeAlgorithm.DFLASH):
                self.assertIs(
                    TpuSRTPlatform().get_spec_worker_cls(
                        algorithm, SimpleNamespace(pp_size=2), default=object
                    ),
                    object,
                )


if __name__ == "__main__":
    unittest.main()
