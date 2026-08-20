"""CPU-side coverage for TPU distributed exact top-k dispatch and merging.
Tests enforce conservative fallbacks without importing physical-TPU dependencies."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import sampler as tpu_sampler_module
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.sampler import (
    TpuSampler,
    distributed_top_k_for_batch,
    sample_from_global_top_k_logits,
)
from sglang.srt.hardware_backend.tpu.top_k import (
    TpuVocabShard,
    _prefer_lowest_token_id_for_tied_max,
    distributed_tpu_top_k,
    supports_distributed_tpu_top_k,
    tpu_top_k,
)
from sglang.srt.layers import logits_processor as logits_processor_module
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


def _forward_batch(*, batch_size=1, **info_overrides):
    info = SimpleNamespace(
        uniform_top_k=50,
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=None,
        return_sampling_masks=None,
        sampling_mask_max_top_k=0,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        sampling_seed=None,
        top_ks=torch.tensor([50], dtype=torch.int32),
        top_ps=torch.ones(1),
        min_ps=torch.zeros(1),
        temperatures=torch.ones(1, 1),
    )
    for name, value in info_overrides.items():
        setattr(info, name, value)
    mode = SimpleNamespace(
        is_decode=lambda: True,
        is_extend_without_speculative=lambda: False,
    )
    return SimpleNamespace(
        sampling_info=info,
        batch_size=batch_size,
        return_logprob=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        forward_mode=mode,
    )


class TestDistributedTopKEligibility(CustomTestCase):
    def test_tp16_small_batches_can_lower_cutoff(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        tp16_lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=202_048,
            num_embeddings_per_partition=202_048 // 16,
            shard_indices=object(),
        )

        with patch.object(tpu_sampler_module, "_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE", 1):
            for batch_size in (1, 2, 4, 8, 16):
                with self.subTest(batch_size=batch_size):
                    batch = _forward_batch(batch_size=batch_size)
                    self.assertEqual(
                        LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                            processor, batch, tp16_lm_head
                        ),
                        50,
                    )

    def test_measured_default_uses_distributed_top_k_from_batch_64(self):
        cutoff = envs.SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE
        self.assertEqual(cutoff.default, 64)

        with patch.object(tpu_sampler_module, "_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE", 64):
            self.assertIsNone(
                distributed_top_k_for_batch(
                    _forward_batch(batch_size=63), sampling_backend="tpu"
                )
            )
            self.assertEqual(
                distributed_top_k_for_batch(
                    _forward_batch(batch_size=64), sampling_backend="tpu"
                ),
                50,
            )

    def test_compiled_selector_ignores_logical_size_within_physical_bucket(self):
        batch = _forward_batch(batch_size=64)
        batch._tpu_real_b = 64

        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def select_top_k(value):
            top_k = distributed_top_k_for_batch(batch, sampling_backend="tpu")
            return value + (top_k or 0)

        value = torch.ones(1)
        expected = select_top_k(value)
        batch._tpu_real_b = 1

        with torch.compiler.set_stance("fail_on_recompile"):
            actual = select_top_k(value)

        torch.testing.assert_close(actual, expected)

    def test_below_cutoff_does_not_guard_on_request_metadata(self):
        batch = _forward_batch(batch_size=63)
        batch.spec_algorithm = None

        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def select_top_k(value):
            top_k = distributed_top_k_for_batch(batch, sampling_backend="tpu")
            return value + (top_k or 0)

        value = torch.ones(1)
        expected = select_top_k(value)
        batch.spec_algorithm = SpeculativeAlgorithm.EAGLE

        with torch.compiler.set_stance("fail_on_recompile"):
            actual = select_top_k(value)

        torch.testing.assert_close(actual, expected)

    def test_explicitly_skipped_tensor_parallel_gather_falls_back(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=False,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        sharded_lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=128,
            num_embeddings_per_partition=16,
            shard_indices=object(),
        )

        self.assertIsNone(
            LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                processor, _forward_batch(), sharded_lm_head
            )
        )

    def test_full_vocab_features_fall_back(self):
        cases = {
            "mixed top-k": {"uniform_top_k": None},
            "large top-k": {"uniform_top_k": 65},
            "mixed greedy/top-k": {
                "uniform_top_k": None,
                "is_any_greedy": True,
            },
            "seeded sampling": {"sampling_seed": torch.tensor([17])},
            "top-p": {"need_top_p_sampling": True},
            "min-p": {"need_min_p_sampling": True},
            "custom processor": {"has_custom_logit_processor": True},
            "logit bias": {"logit_bias": torch.zeros(1, 8)},
            "grammar": {"grammars": [object()]},
            "sampling mask": {"sampling_mask_max_top_k": 50},
            "returned sampling mask": {"return_sampling_masks": [True]},
            "additive penalty": {"acc_additive_penalties": torch.zeros(1, 8)},
            "scaling penalty": {"acc_scaling_penalties": torch.ones(1, 8)},
            "live penalizer": {
                "penalizer_orchestrator": SimpleNamespace(is_required=True)
            },
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(
                    distributed_top_k_for_batch(
                        _forward_batch(batch_size=64, **overrides),
                        sampling_backend="tpu",
                    )
                )

    def test_pure_greedy_uses_fixed_width_candidate_reduction(self):
        batch = _forward_batch(
            batch_size=64,
            uniform_top_k=1,
            is_all_greedy=True,
            is_any_greedy=True,
            top_ks=torch.ones(64, dtype=torch.int32),
        )

        self.assertEqual(
            distributed_top_k_for_batch(batch, sampling_backend="tpu"),
            1,
        )

    def test_non_tpu_and_logprob_batches_fall_back(self):
        batch = _forward_batch(batch_size=64)
        self.assertIsNone(
            distributed_top_k_for_batch(batch, sampling_backend="pytorch")
        )
        batch.return_logprob = True
        self.assertIsNone(distributed_top_k_for_batch(batch, sampling_backend="tpu"))

    def test_speculative_batch_falls_back(self):
        batch = _forward_batch(batch_size=64)
        batch.spec_algorithm = SpeculativeAlgorithm.EAGLE

        self.assertIsNone(distributed_top_k_for_batch(batch, sampling_backend="tpu"))

    def test_environment_can_raise_the_pre_gather_cutoff(self):
        cutoff = envs.SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE
        self.assertEqual(cutoff.default, 64)
        try:
            with cutoff.override(128):
                importlib.reload(tpu_sampler_module)

            self.assertEqual(
                tpu_sampler_module._DISTRIBUTED_TOP_K_MIN_BATCH_SIZE,
                128,
            )
            self.assertIsNone(
                tpu_sampler_module.distributed_top_k_for_batch(
                    _forward_batch(batch_size=127), sampling_backend="tpu"
                )
            )
            padded_batch = _forward_batch(batch_size=128)
            padded_batch._tpu_real_b = 127
            self.assertEqual(
                tpu_sampler_module.distributed_top_k_for_batch(
                    padded_batch, sampling_backend="tpu"
                ),
                50,
            )
            self.assertEqual(
                tpu_sampler_module.distributed_top_k_for_batch(
                    _forward_batch(batch_size=128), sampling_backend="tpu"
                ),
                50,
            )
        finally:
            # Restore the process-start configuration for the remaining tests.
            importlib.reload(tpu_sampler_module)

    def test_model_requiring_full_logits_disables_global_top_k(self):
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_fp32_lm_head=False, enable_mis=False),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
            kernel=SimpleNamespace(sampling_backend="tpu"),
        )
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            tp_size=8,
            attn_dp_size=1,
            nnodes=1,
        )
        config = SimpleNamespace(vocab_size=128)

        with (
            patch.object(logits_processor_module, "_is_tpu", True),
            patch.object(logits_processor_module, "get_exec", return_value=execution),
            patch.object(
                logits_processor_module, "get_parallel", return_value=parallel
            ),
            patch.object(
                logits_processor_module.triton_symm_mem_ag,
                "recommended_max_tokens",
                return_value=128,
            ),
            patch.object(
                logits_processor_module.triton_symm_mem_ag,
                "MultimemAllGatherer",
                return_value=object(),
            ),
        ):
            processor = LogitsProcessor(config, requires_full_next_token_logits=True)

        self.assertFalse(processor.use_tpu_distributed_top_k)

    def test_multihost_keeps_distributed_top_k_available(self):
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_fp32_lm_head=False, enable_mis=False),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
            kernel=SimpleNamespace(sampling_backend="tpu"),
        )
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            tp_size=16,
            attn_dp_size=1,
            nnodes=2,
        )
        config = SimpleNamespace(vocab_size=128)

        with (
            patch.object(logits_processor_module, "_is_tpu", True),
            patch.object(logits_processor_module, "get_exec", return_value=execution),
            patch.object(
                logits_processor_module, "get_parallel", return_value=parallel
            ),
            patch.object(
                logits_processor_module.triton_symm_mem_ag,
                "recommended_max_tokens",
                return_value=128,
            ),
            patch.object(
                logits_processor_module.triton_symm_mem_ag,
                "MultimemAllGatherer",
                return_value=object(),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.register_tpu_top_k_ops"
            ) as register_top_k,
        ):
            processor = LogitsProcessor(config)

        self.assertTrue(processor.use_tpu_distributed_top_k)
        register_top_k.assert_called_once_with()

    def test_multihost_compiled_collective_requires_the_deferred_boundary(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=202_048,
            num_embeddings_per_partition=12_672,
            shard_indices=object(),
        )
        batch = _forward_batch(batch_size=64)
        batch.tpu_defer_logits_postprocess = False

        with (
            patch.object(
                logits_processor_module,
                "get_parallel",
                return_value=SimpleNamespace(nnodes=2),
            ),
            patch.object(torch.compiler, "is_compiling", return_value=True),
        ):
            self.assertIsNone(
                LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                    processor, batch, lm_head
                )
            )
            batch.tpu_defer_logits_postprocess = True
            self.assertEqual(
                LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                    processor, batch, lm_head
                ),
                50,
            )

        batch.tpu_defer_logits_postprocess = False
        with patch.object(torch.compiler, "is_compiling", return_value=False):
            self.assertEqual(
                LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                    processor, batch, lm_head
                ),
                50,
            )

    def test_deferred_logits_use_the_largest_model_static_candidate_width(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=202_048,
            num_embeddings_per_partition=12_672,
            shard_indices=object(),
        )

        self.assertEqual(
            LogitsProcessor.get_tpu_deferred_top_k_width(processor, lm_head), 64
        )
        lm_head.num_embeddings_per_partition = 32
        self.assertEqual(
            LogitsProcessor.get_tpu_deferred_top_k_width(processor, lm_head), 32
        )

    def test_deferred_top_k_reduces_to_one_static_candidate_width(self):
        sharded_logits = torch.randn(2, 128)
        global_logits = torch.randn(2, 64)
        global_token_ids = torch.arange(128, dtype=torch.int32).reshape(2, 64)
        reducer = MagicMock(return_value=(global_logits, global_token_ids))
        processor = SimpleNamespace(
            get_tpu_distributed_top_k_for_batch=MagicMock(return_value=7),
            get_tpu_deferred_top_k_width=MagicMock(return_value=64),
            _reduce_tpu_distributed_top_k=reducer,
        )
        lm_head = object()
        forward_batch = _forward_batch(batch_size=64, uniform_top_k=7)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_sharded_logits=sharded_logits,
        )

        result = LogitsProcessor.finalize_tpu_deferred_logits(
            processor, output, forward_batch, lm_head, 64
        )

        self.assertIsNone(result.next_token_sharded_logits)
        self.assertIsNone(result.next_token_logits)
        self.assertIs(result.next_token_global_top_k_logits, global_logits)
        self.assertIs(result.next_token_global_top_k_token_ids, global_token_ids)
        reducer.assert_called_once_with(sharded_logits, lm_head, 64)

    def test_deferred_default_sampling_uses_full_vocab_gather(self):
        sharded_logits = torch.randn(2, 128)
        full_logits = torch.randn(2, 256)
        finish_full_logits = MagicMock(return_value=full_logits)
        processor = SimpleNamespace(
            get_tpu_distributed_top_k_for_batch=MagicMock(return_value=None),
            _finalize_full_logits=finish_full_logits,
        )
        forward_batch = _forward_batch(batch_size=64, uniform_top_k=1 << 30)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_sharded_logits=sharded_logits,
        )
        metadata = object()

        with patch.object(
            logits_processor_module.LogitsMetadata,
            "from_forward_batch",
            return_value=metadata,
        ):
            result = LogitsProcessor.finalize_tpu_deferred_logits(
                processor, output, forward_batch, object(), 64
            )

        self.assertIsNone(result.next_token_sharded_logits)
        self.assertIs(result.next_token_logits, full_logits)
        self.assertFalse(result.has_next_token_global_top_k)
        finish_full_logits.assert_called_once_with(
            sharded_logits,
            None,
            metadata,
        )

    def test_deferred_full_vocab_matches_existing_logits_path(self):
        processor = object.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.do_tensor_parallel_all_gather = True
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor.use_attn_tp_group = False
        processor.use_fp32_lm_head = False
        processor.rl_on_policy_target = None
        processor.logit_scale = 1.25
        processor.final_logit_softcapping = 1.75
        # Two four-token shards model a padded global vocabulary of width eight.
        processor.vocab_size = 5

        def gather_shards(local_logits):
            return torch.cat((local_logits, local_logits + 0.5), dim=-1)

        processor._logits_gatherer = MagicMock(side_effect=gather_shards)
        lm_head = torch.nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.4],
                        [-0.3, 0.5, 0.1],
                        [0.7, 0.2, -0.2],
                        [0.1, -0.6, 0.3],
                    ]
                )
            )
        hidden_states = torch.tensor(
            [[0.5, -1.0, 0.25], [1.5, 0.1, -0.75]], dtype=torch.float32
        )
        metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)

        with patch.object(logits_processor_module, "_is_cpu", True):
            existing_full_logits = processor._get_logits(
                hidden_states,
                lm_head,
                metadata,
                use_logits_buffer=False,
            )
            sharded_logits, _ = processor._get_local_logits(
                hidden_states,
                lm_head,
                metadata,
            )
            output = LogitsProcessorOutput(
                next_token_logits=None,
                next_token_sharded_logits=sharded_logits,
            )

            with (
                patch.object(
                    processor,
                    "get_tpu_distributed_top_k_for_batch",
                    return_value=None,
                ),
                patch.object(
                    logits_processor_module.LogitsMetadata,
                    "from_forward_batch",
                    return_value=metadata,
                ),
            ):
                deferred_full_logits = processor.finalize_tpu_deferred_logits(
                    output,
                    _forward_batch(batch_size=2, uniform_top_k=1 << 30),
                    lm_head,
                    candidate_width=4,
                ).next_token_logits

        scaled_local_logits = (hidden_states @ lm_head.weight.T) * 1.25
        expected = torch.cat((scaled_local_logits, scaled_local_logits + 0.5), dim=-1)[
            :, :5
        ]
        expected = 1.75 * torch.tanh(expected / 1.75)

        self.assertEqual(existing_full_logits.shape, (2, 5))
        self.assertIsNone(output.next_token_sharded_logits)
        torch.testing.assert_close(existing_full_logits, expected)
        torch.testing.assert_close(deferred_full_logits, existing_full_logits)
        self.assertEqual(processor._logits_gatherer.call_count, 2)

    def test_deferred_compiled_logits_do_not_guard_on_sampling_mode(self):
        processor = object.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor.final_logit_softcapping = None
        processor.logit_scale = None
        processor.rl_on_policy_target = None
        processor.use_fp32_lm_head = False

        lm_head = torch.nn.Linear(4, 8, bias=False)
        policy = DecoderTpuCompilePolicy()
        policy._tpu_logits_postprocessor = (processor, lm_head, 8)
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=4,
            input_ids=torch.arange(4),
            req_pool_indices=torch.arange(4),
            seq_lens=torch.ones(4, dtype=torch.int32),
            out_cache_loc=torch.arange(4),
            seq_lens_sum=4,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            sampling_info=_forward_batch(batch_size=4).sampling_info,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        hidden_states = torch.randn(4, 4)

        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def compiled_logits(states, forward_batch):
            return processor(None, states, lm_head, forward_batch)

        def run():
            with policy.compiled_forward_context(batch):
                return compiled_logits(hidden_states, batch)

        expected = run().next_token_sharded_logits
        self.assertEqual(expected.shape, (4, 8))

        sampling_info = batch.sampling_info
        sampling_info.uniform_top_k = 1
        sampling_info.is_any_greedy = True
        sampling_info.is_all_greedy = True
        sampling_info.top_ks.fill_(1)
        with torch.compiler.set_stance("fail_on_recompile"):
            greedy = run().next_token_sharded_logits

        sampling_info.uniform_top_k = 1 << 30
        sampling_info.is_any_greedy = False
        sampling_info.is_all_greedy = False
        sampling_info.need_top_k_sampling = False
        sampling_info.top_ks.fill_(1 << 30)
        with torch.compiler.set_stance("fail_on_recompile"):
            full_vocab = run().next_token_sharded_logits

        torch.testing.assert_close(greedy, expected)
        torch.testing.assert_close(full_vocab, expected)


class TestTpuSamplerGlobalTopK(CustomTestCase):
    def test_global_top_k_outputs_are_paired_and_exclusive(self):
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            LogitsProcessorOutput(
                next_token_logits=None,
                next_token_global_top_k_logits=torch.ones(1, 2),
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LogitsProcessorOutput(
                next_token_logits=torch.ones(1, 8),
                next_token_global_top_k_logits=torch.ones(1, 2),
                next_token_global_top_k_token_ids=torch.ones(1, 2, dtype=torch.int32),
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LogitsProcessorOutput(
                next_token_logits=torch.ones(1, 8),
                next_token_sharded_logits=torch.ones(1, 4),
            )

    def test_global_top_k_position_maps_to_global_token_id(self):
        sampler = SimpleNamespace(_sync_token_ids_across_tp=MagicMock())
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_global_top_k_logits=torch.tensor([[3.0, 2.0, 1.0]]),
            next_token_global_top_k_token_ids=torch.tensor(
                [[91, 17, 8]], dtype=torch.int32
            ),
        )
        info = _forward_batch().sampling_info

        with patch.object(
            tpu_sampler_module,
            "sample_from_global_top_k_logits",
            return_value=torch.tensor([1], dtype=torch.int32),
        ):
            actual = TpuSampler.forward(
                sampler,
                output,
                info,
                False,
                [0],
                [[]],
                torch.tensor([7]),
            )

        torch.testing.assert_close(actual, torch.tensor([17], dtype=torch.int64))
        sampler._sync_token_ids_across_tp.assert_called_once()

    def test_greedy_selects_the_first_global_candidate_without_sampling(self):
        sampler = SimpleNamespace(_sync_token_ids_across_tp=MagicMock())
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_global_top_k_logits=torch.tensor(
                [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]
            ),
            next_token_global_top_k_token_ids=torch.tensor(
                [[91, 17, 8], [12, 42, 3]], dtype=torch.int32
            ),
        )
        info = _forward_batch(
            batch_size=2,
            uniform_top_k=1,
            is_all_greedy=True,
            is_any_greedy=True,
            top_ks=torch.ones(2, dtype=torch.int32),
        ).sampling_info

        with patch.object(
            tpu_sampler_module,
            "sample_from_global_top_k_logits",
        ) as sample:
            actual = TpuSampler.forward(
                sampler,
                output,
                info,
                False,
                [0, 0],
                [[], []],
                torch.tensor([7, 8]),
            )

        torch.testing.assert_close(actual, torch.tensor([91, 12], dtype=torch.int64))
        sample.assert_not_called()
        sampler._sync_token_ids_across_tp.assert_called_once()

    def test_sampler_rejects_an_unresolved_compile_boundary_output(self):
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_sharded_logits=torch.ones(1, 8),
        )

        with self.assertRaisesRegex(RuntimeError, "must finalize"):
            TpuSampler.forward(
                SimpleNamespace(),
                output,
                _forward_batch().sampling_info,
                False,
                [0],
                [[]],
                torch.tensor([7]),
            )

    def test_global_top_k_sampling_normalizes_only_top_k(self):
        global_top_k_logits = torch.tensor([[4.0, 2.0]])
        with patch.object(
            tpu_sampler_module,
            "sampling_from_probs_torch",
            return_value=torch.tensor([1], dtype=torch.int32),
        ) as sample:
            actual = sample_from_global_top_k_logits(
                global_top_k_logits,
                torch.tensor([[2.0]]),
            )

        torch.testing.assert_close(actual, torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(
            sample.call_args.args[0], torch.softmax(torch.tensor([[2.0, 1.0]]), -1)
        )

    def test_fixed_candidate_width_is_masked_to_each_requested_top_k(self):
        global_top_k_logits = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [8.0, 6.0, 4.0, 2.0],
            ]
        )
        with patch.object(
            tpu_sampler_module,
            "sampling_from_probs_torch",
            return_value=torch.tensor([0, 1], dtype=torch.int32),
        ) as sample:
            sample_from_global_top_k_logits(
                global_top_k_logits,
                torch.ones(2, 1),
                torch.tensor([2, 3], dtype=torch.int32),
            )

        expected = torch.tensor(
            [
                [0.7310586, 0.2689414, 0.0, 0.0],
                [0.8668133, 0.1173104, 0.0158762, 0.0],
            ]
        )
        torch.testing.assert_close(sample.call_args.args[0], expected)

    def test_fallback_uses_generic_pytorch_sampler(self):
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch(uniform_top_k=None).sampling_info
        expected = torch.tensor([17], dtype=torch.int32)

        with patch.object(
            tpu_sampler_module,
            "top_k_top_p_min_p_sampling_from_probs_torch",
            return_value=expected,
        ) as generic_sampler:
            actual = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        self.assertIs(actual, expected)
        generic_sampler.assert_called_once_with(
            probs,
            info.top_ks,
            info.top_ps,
            info.min_ps,
            False,
            None,
            positions,
        )

    def test_uniform_top_k_uses_post_gather_native_top_k(self):
        sampler = SimpleNamespace()
        probs = torch.rand(1, 128)
        positions = torch.tensor([7])
        info = _forward_batch().sampling_info
        global_top_k_probs = torch.tensor([[0.7, 0.3]])
        global_top_k_token_ids = torch.tensor([[91, 17]], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.tpu_top_k",
                return_value=(global_top_k_probs, global_top_k_token_ids),
            ) as top_k,
            patch.object(
                tpu_sampler_module,
                "sampling_from_probs_torch",
                return_value=torch.tensor([1], dtype=torch.int32),
            ) as sample,
        ):
            actual = TpuSampler._sample_from_probs(
                sampler, probs, info, positions, simple_sampling_case=False
            )

        torch.testing.assert_close(actual, torch.tensor([17], dtype=torch.int32))
        top_k.assert_called_once_with(probs, 50)
        sample.assert_called_once_with(global_top_k_probs)


class TestTpuTopKLayout(CustomTestCase):
    def test_tied_max_swaps_the_lowest_selected_token_id_into_slot_zero(self):
        candidate_logits = torch.tensor([[9.0, 9.0, 8.0]])
        candidate_token_ids = torch.tensor([[17, 3, 8]], dtype=torch.int32)
        unstable_top_k_ids = torch.tensor([[17, 3]], dtype=torch.int32)

        corrected = _prefer_lowest_token_id_for_tied_max(
            candidate_logits,
            candidate_token_ids,
            unstable_top_k_ids,
        )

        torch.testing.assert_close(
            corrected,
            torch.tensor([[3, 17]], dtype=torch.int32),
        )
        self.assertEqual(corrected.unique().numel(), corrected.numel())

    def test_tied_max_inserts_lowest_token_id_when_global_top_k_omits_it(self):
        candidate_logits = torch.tensor([[9.0, 9.0, 8.0]])
        candidate_token_ids = torch.tensor([[17, 3, 8]], dtype=torch.int32)
        unstable_top_k_ids = torch.tensor([[17]], dtype=torch.int32)

        corrected = _prefer_lowest_token_id_for_tied_max(
            candidate_logits,
            candidate_token_ids,
            unstable_top_k_ids,
        )

        torch.testing.assert_close(corrected, torch.tensor([[3]], dtype=torch.int32))

    def test_distributed_path_requires_a_sharded_lm_head(self):
        layout = SimpleNamespace(
            enable_tp=True,
            num_embeddings=128,
            num_embeddings_per_partition=16,
            shard_indices=object(),
        )
        self.assertTrue(supports_distributed_tpu_top_k(layout, 16))

        layout.enable_tp = False
        self.assertFalse(supports_distributed_tpu_top_k(layout, 16))
        with self.assertRaisesRegex(ValueError, "vocab-sharded LM head"):
            TpuVocabShard.from_lm_head(layout)

    def test_padding_and_added_vocab_are_mapped(self):
        shard = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=8,
            padded_org_vocab_end_index=16,
            padded_added_vocab_start_index=20,
            padded_added_vocab_end_index=24,
            org_vocab_start_index=8,
            org_vocab_end_index=14,
            added_vocab_start_index=20,
            added_vocab_end_index=22,
        )
        lm_head = SimpleNamespace(
            enable_tp=True,
            shard_indices=shard,
            num_embeddings=22,
            num_embeddings_per_partition=12,
        )

        self.assertEqual(
            TpuVocabShard.from_lm_head(lm_head),
            TpuVocabShard(
                width=12,
                global_vocab_size=22,
                org_vocab_start=8,
                org_vocab_count=6,
                org_vocab_padded_count=8,
                added_vocab_start=20,
                added_vocab_count=2,
            ),
        )

    def test_primitive_rejects_cpu_tensor_before_tpu_dependencies_load(self):
        with self.assertRaisesRegex(ValueError, "requires a TPU tensor"):
            tpu_top_k(torch.ones(1, 8), 4)

    def test_distributed_merge_matches_full_top_k(self):
        local_logits = torch.tensor([[9.0, 3.0, -2.0, 7.0]])
        shard = TpuVocabShard(
            width=4,
            global_vocab_size=8,
            org_vocab_start=0,
            org_vocab_count=2,
            org_vocab_padded_count=3,
            added_vocab_start=3,
            added_vocab_count=1,
        )
        peer_logits = torch.tensor([[8.0, 6.0, 1.0, 0.0]])
        peer_ids = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
        k = 3

        local_ids = torch.tensor([0, 1, -1, 3], dtype=torch.int32)
        masked_local = local_logits.masked_fill(
            local_ids.lt(0).unsqueeze(0), -torch.inf
        )
        local_sharded_top_k_logits, local_sharded_top_k_positions = torch.topk(
            masked_local, k, dim=-1
        )
        local_sharded_top_k_token_ids = local_ids[local_sharded_top_k_positions]
        peer_sharded_top_k_logits, peer_sharded_top_k_positions = torch.topk(
            peer_logits, k, dim=-1
        )
        peer_sharded_top_k_token_ids = peer_ids[peer_sharded_top_k_positions]
        group = SimpleNamespace(
            world_size=2,
            all_gather=MagicMock(
                return_value=torch.cat(
                    (
                        torch.stack(
                            (
                                local_sharded_top_k_logits.float(),
                                local_sharded_top_k_token_ids.float(),
                            ),
                            dim=-1,
                        ),
                        torch.stack(
                            (
                                peer_sharded_top_k_logits.float(),
                                peer_sharded_top_k_token_ids.float(),
                            ),
                            dim=-1,
                        ),
                    ),
                    dim=1,
                )
            ),
        )

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.tpu_sharded_top_k",
                return_value=(
                    local_sharded_top_k_logits,
                    local_sharded_top_k_token_ids,
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.tpu_top_k",
                side_effect=lambda values, count: torch.topk(values, count, dim=-1),
            ),
        ):
            global_top_k_logits, global_top_k_token_ids = distributed_tpu_top_k(
                local_logits.clone(),
                shard,
                k,
                group=group,
            )

        torch.testing.assert_close(global_top_k_logits, torch.tensor([[9.0, 8.0, 7.0]]))
        torch.testing.assert_close(
            global_top_k_token_ids, torch.tensor([[0, 4, 3]], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
