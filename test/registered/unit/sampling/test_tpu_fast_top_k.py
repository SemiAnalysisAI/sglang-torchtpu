"""CPU-side coverage for TPU distributed exact top-k dispatch and merging.
Tests enforce conservative fallbacks without importing physical-TPU dependencies."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import sampler as tpu_sampler_module
from sglang.srt.hardware_backend.tpu.sampler import (
    TpuSampler,
    distributed_top_k_for_batch,
    sample_from_global_top_k_logits,
)
from sglang.srt.hardware_backend.tpu.top_k import (
    TpuVocabShard,
    distributed_tpu_top_k,
    merge_tpu_top_k_candidates,
    supports_distributed_tpu_top_k,
    tpu_top_k,
)
from sglang.srt.layers import logits_processor as logits_processor_module
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


def _forward_batch(*, batch_size=64, **info_overrides):
    info = SimpleNamespace(
        uniform_top_k=50,
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
    def test_candidate_width_is_static_across_request_and_batch_sizes(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        processor.get_tpu_distributed_top_k_width = lambda lm_head: (
            LogitsProcessor.get_tpu_distributed_top_k_width(processor, lm_head)
        )
        tp16_lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=202_048,
            num_embeddings_per_partition=202_048 // 16,
            shard_indices=object(),
        )

        with patch.object(tpu_sampler_module, "_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE", 1):
            for batch_size in (1, 2, 4, 8, 16):
                for top_k in (2, 7, 50, 64):
                    with self.subTest(batch_size=batch_size, top_k=top_k):
                        batch = _forward_batch(
                            batch_size=batch_size,
                            uniform_top_k=top_k,
                            top_ks=torch.tensor([top_k], dtype=torch.int32),
                        )
                        self.assertEqual(
                            LogitsProcessor.get_tpu_distributed_top_k_for_batch(
                                processor, batch, tp16_lm_head
                            ),
                            64,
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

    def test_explicitly_skipped_tensor_parallel_gather_falls_back(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=False,
            do_tensor_parallel_all_gather_dp_attn=False,
        )
        processor.get_tpu_distributed_top_k_width = lambda lm_head: (
            LogitsProcessor.get_tpu_distributed_top_k_width(processor, lm_head)
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
            "greedy row": {"uniform_top_k": 1, "is_any_greedy": True},
            "large top-k": {"uniform_top_k": 65},
            "seeded sampling": {"sampling_seed": torch.tensor([17])},
            "top-p": {"need_top_p_sampling": True},
            "min-p": {"need_min_p_sampling": True},
            "custom processor": {"has_custom_logit_processor": True},
            "logit bias": {"logit_bias": torch.zeros(1, 8)},
            "grammar": {"grammars": [object()]},
            "sampling mask": {"sampling_mask_max_top_k": 50},
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
                        _forward_batch(**overrides), sampling_backend="tpu"
                    )
                )

    def test_non_tpu_and_logprob_batches_fall_back(self):
        batch = _forward_batch()
        self.assertIsNone(
            distributed_top_k_for_batch(batch, sampling_backend="pytorch")
        )
        batch.return_logprob = True
        self.assertIsNone(distributed_top_k_for_batch(batch, sampling_backend="tpu"))

    def test_speculative_batch_falls_back(self):
        batch = _forward_batch()
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


class TestTpuSamplerGlobalTopK(CustomTestCase):
    def test_sharded_logits_and_metadata_are_paired_and_exclusive(self):
        shard = TpuVocabShard(8, 16, 0, 8, 8, 16, 0)
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            LogitsProcessorOutput(
                next_token_logits=None,
                next_token_sharded_logits=torch.ones(1, 8),
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LogitsProcessorOutput(
                next_token_logits=torch.ones(1, 8),
                next_token_sharded_logits=torch.ones(1, 8),
                tpu_top_k_vocab_shard=shard,
            )

    def test_fullgraph_output_preserves_static_vocab_shard_metadata(self):
        shard = TpuVocabShard(8, 16, 0, 8, 8, 16, 0)
        lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=16,
            num_embeddings_per_partition=8,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                num_org_elements=8,
                num_org_elements_padded=8,
                added_vocab_start_index=16,
                num_added_elements=0,
            ),
        )

        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def compiled_candidate_output(values):
            # The production model returns this same tensor-plus-host-metadata
            # contract. Fullgraph coverage prevents a silent graph break from
            # hiding an unsupported output type until the TPU qualification.
            return LogitsProcessorOutput(
                next_token_logits=None,
                next_token_sharded_logits=values + 1,
                tpu_top_k_vocab_shard=TpuVocabShard.from_lm_head(lm_head),
            )

        output = compiled_candidate_output(torch.zeros(2, 8))

        torch.testing.assert_close(output.next_token_sharded_logits, torch.ones(2, 8))
        self.assertEqual(output.tpu_top_k_vocab_shard, shard)

    def test_equal_row_trim_preserves_prewarmed_sharded_tensor_layout(self):
        shard = TpuVocabShard(8, 16, 0, 8, 8, 16, 0)
        sharded_logits = torch.randn(64, 8)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_sharded_logits=sharded_logits,
            tpu_top_k_vocab_shard=shard,
        )

        output.trim_next_token_outputs(64)

        # A no-op slice would create a view between the compiled output and the
        # eager TopK boundary even though Richard's B64 batch needs no unpadding.
        self.assertIs(output.next_token_sharded_logits, sharded_logits)

    def test_eager_global_merge_maps_sampled_slot_to_token_id(self):
        sampler = SimpleNamespace(_sync_token_ids_across_tp=MagicMock())
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_sharded_logits=torch.tensor([[3.0, 2.0, 1.0]]),
            tpu_top_k_vocab_shard=TpuVocabShard(3, 3, 0, 3, 3, 3, 0),
        )
        info = _forward_batch().sampling_info
        global_logits = torch.tensor([[8.0, 7.0, 6.0]])
        global_token_ids = torch.tensor([[503, 17, 8]], dtype=torch.int32)

        with (
            patch.object(
                tpu_sampler_module,
                "_distributed_top_k_from_sharded_logits",
                return_value=(global_logits, global_token_ids),
            ) as distributed_top_k,
            patch.object(
                tpu_sampler_module,
                "sample_from_global_top_k_logits",
                return_value=torch.tensor([1], dtype=torch.int32),
            ),
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
        distributed_top_k.assert_called_once_with(output)
        sampler._sync_token_ids_across_tp.assert_called_once()

    def test_complete_distributed_top_k_runs_at_eager_sampler_boundary(self):
        local_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        shard = TpuVocabShard(4, 8, 0, 4, 4, 8, 0)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_sharded_logits=local_logits,
            tpu_top_k_vocab_shard=shard,
        )
        group = SimpleNamespace()
        global_logits = torch.tensor([[8.0, 7.0]])
        global_ids = torch.tensor([[17, 5]], dtype=torch.int32)

        with (
            patch.object(
                tpu_sampler_module,
                "get_parallel",
                return_value=SimpleNamespace(enable_dp_lm_head=False),
            ),
            patch.object(tpu_sampler_module, "get_tp_group", return_value=group),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.distributed_tpu_top_k",
                return_value=(global_logits, global_ids),
            ) as distributed_top_k,
        ):
            actual = tpu_sampler_module._distributed_top_k_from_sharded_logits(output)

        torch.testing.assert_close(actual[0], global_logits)
        torch.testing.assert_close(actual[1], global_ids)
        distributed_top_k.assert_called_once_with(
            local_logits,
            shard,
            4,
            group=group,
        )

    def test_eager_top_k_is_bounded_by_input_and_output_materialization(self):
        # Tensor-like mocks let this CPU test exercise the TPU-only sync branch
        # without importing a physical torch_tpu runtime.
        local_logits = MagicMock(device=SimpleNamespace(type="tpu"), shape=(2, 8))
        shard = TpuVocabShard(8, 16, 0, 8, 8, 16, 0)
        output = SimpleNamespace(
            next_token_sharded_logits=local_logits,
            tpu_top_k_vocab_shard=shard,
        )
        group = SimpleNamespace()
        merged = (MagicMock(), MagicMock())
        events = []

        def run_distributed_top_k(*_args, **_kwargs):
            events.append("distributed_top_k")
            return merged

        torch_tpu_module = ModuleType("torch_tpu")
        internal_module = ModuleType("torch_tpu._internal")
        sync_package = ModuleType("torch_tpu._internal.sync")
        sync_module = ModuleType("torch_tpu._internal.sync.sync")
        sync_module.synchronize = MagicMock()
        torch_tpu_module._internal = internal_module
        internal_module.sync = sync_package
        sync_package.sync = sync_module
        with (
            patch.dict(
                sys.modules,
                {
                    "torch_tpu": torch_tpu_module,
                    "torch_tpu._internal": internal_module,
                    "torch_tpu._internal.sync": sync_package,
                    "torch_tpu._internal.sync.sync": sync_module,
                },
            ),
            patch.object(
                tpu_sampler_module,
                "get_parallel",
                return_value=SimpleNamespace(enable_dp_lm_head=False),
            ),
            patch.object(tpu_sampler_module, "get_tp_group", return_value=group),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.distributed_tpu_top_k",
                side_effect=run_distributed_top_k,
            ) as distributed_top_k,
        ):
            sync_module.synchronize.side_effect = lambda *_args, **_kwargs: (
                events.append("sync")
            )
            actual = tpu_sampler_module._distributed_top_k_from_sharded_logits(output)

        self.assertIs(actual, merged)
        self.assertEqual(events, ["sync", "distributed_top_k", "sync"])
        self.assertEqual(
            sync_module.synchronize.call_args_list,
            [
                call([local_logits], wait=True),
                call(list(merged), wait=True),
            ],
        )
        distributed_top_k.assert_called_once_with(
            local_logits,
            shard,
            8,
            group=group,
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

    def test_global_top_k_sampling_masks_requested_tail(self):
        logits = torch.tensor([[4.0, 3.0, 2.0], [6.0, 5.0, 4.0]])
        top_ks = torch.tensor([2, 2], dtype=torch.int32)
        with patch.object(
            tpu_sampler_module,
            "sampling_from_probs_torch",
            return_value=torch.tensor([0, 1], dtype=torch.int32),
        ) as sample:
            sample_from_global_top_k_logits(logits, torch.ones(2, 1), top_ks)

        probs = sample.call_args.args[0]
        torch.testing.assert_close(probs[:, 2], torch.zeros(2))
        torch.testing.assert_close(probs[:, :2].sum(dim=1), torch.ones(2))

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
    def test_compiled_logits_path_stops_at_fp32_shard_local_logits(self):
        # A transposed BF16 tensor models the unsafe terminal LM-head result:
        # candidate mode must establish an explicit dtype/layout boundary while
        # keeping all TopK operations outside the compiled model.
        local_logits = torch.randn(16, 2, dtype=torch.bfloat16).T
        processor = SimpleNamespace(
            _compute_lm_head=MagicMock(return_value=local_logits),
            logit_scale=None,
            final_logit_softcapping=None,
        )
        lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=32,
            num_embeddings_per_partition=16,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                num_org_elements=16,
                num_org_elements_padded=16,
                added_vocab_start_index=32,
                num_added_elements=0,
            ),
        )

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.tpu_sharded_top_k"
            ) as local_top_k,
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.distributed_tpu_top_k"
            ) as distributed_top_k,
        ):
            actual_logits, actual_shard = LogitsProcessor._get_tpu_sharded_logits(
                processor,
                torch.randn(2, 4),
                lm_head,
            )

        # No TopK operation is allowed in the complete model. Static shard
        # metadata plus materializable FP32 logits are enough for the eager
        # boundary to run it.
        torch.testing.assert_close(actual_logits, local_logits.float())
        self.assertEqual(actual_logits.dtype, torch.float32)
        self.assertTrue(actual_logits.is_contiguous())
        self.assertEqual(actual_shard, TpuVocabShard(16, 32, 0, 16, 16, 32, 0))
        local_top_k.assert_not_called()
        distributed_top_k.assert_not_called()

    def test_shard_local_softcap_is_applied_before_eager_top_k(self):
        local_logits = torch.tensor([[4.0, -2.0]], dtype=torch.bfloat16)
        processor = SimpleNamespace(
            _compute_lm_head=MagicMock(return_value=local_logits),
            logit_scale=None,
            final_logit_softcapping=2.0,
        )
        lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=4,
            num_embeddings_per_partition=2,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                num_org_elements=2,
                num_org_elements_padded=2,
                added_vocab_start_index=4,
                num_added_elements=0,
            ),
        )

        actual, _ = LogitsProcessor._get_tpu_sharded_logits(
            processor, torch.randn(1, 2), lm_head
        )

        expected = 2.0 * torch.tanh(local_logits.float() / 2.0)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, torch.float32)

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
        gathered_logits = torch.cat(
            (local_sharded_top_k_logits, peer_sharded_top_k_logits), dim=1
        ).float()
        gathered_ids = torch.cat(
            (local_sharded_top_k_token_ids, peer_sharded_top_k_token_ids), dim=1
        )
        group = SimpleNamespace(
            world_size=2,
            all_gather=MagicMock(side_effect=(gathered_logits, gathered_ids)),
        )
        expected_global_logits = torch.tensor([[9.0, 8.0, 7.0]])
        expected_global_positions = torch.tensor([[0, 3, 1]], dtype=torch.int32)

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
                return_value=(expected_global_logits, expected_global_positions),
            ) as native_global_top_k,
        ):
            global_top_k_logits, global_top_k_token_ids = distributed_tpu_top_k(
                local_logits.clone(), shard, k, group=group
            )

        torch.testing.assert_close(global_top_k_logits, expected_global_logits)
        torch.testing.assert_close(
            global_top_k_token_ids, torch.tensor([[0, 4, 3]], dtype=torch.int32)
        )
        native_global_top_k.assert_called_once_with(
            gathered_logits,
            k,
        )
        self.assertEqual(group.all_gather.call_count, 2)
        logits_gather, ids_gather = group.all_gather.call_args_list
        torch.testing.assert_close(
            logits_gather.args[0], local_sharded_top_k_logits.float()
        )
        self.assertEqual(logits_gather.kwargs, {"dim": -1})
        torch.testing.assert_close(ids_gather.args[0], local_sharded_top_k_token_ids)
        self.assertEqual(ids_gather.kwargs, {"dim": -1})

    def test_single_device_returns_local_candidates_without_global_selector(self):
        local_logits = torch.tensor([[9.0, 7.0]])
        local_ids = torch.tensor([[4, 3]], dtype=torch.int32)
        shard = TpuVocabShard(
            width=4,
            global_vocab_size=8,
            org_vocab_start=0,
            org_vocab_count=4,
            org_vocab_padded_count=4,
            added_vocab_start=8,
            added_vocab_count=0,
        )
        group = SimpleNamespace(world_size=1, all_gather=MagicMock())

        with (
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.tpu_sharded_top_k",
                return_value=(local_logits, local_ids),
            ),
            patch("sglang.srt.hardware_backend.tpu.top_k.tpu_top_k") as global_selector,
        ):
            actual_logits, actual_ids = distributed_tpu_top_k(
                torch.ones(1, 4),
                shard,
                2,
                group=group,
            )

        torch.testing.assert_close(actual_logits, local_logits)
        torch.testing.assert_close(actual_ids, local_ids)
        group.all_gather.assert_not_called()
        global_selector.assert_not_called()

    def test_eager_merge_requires_aligned_candidate_pairs(self):
        group = SimpleNamespace(world_size=2)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            merge_tpu_top_k_candidates(
                torch.ones(2, 3),
                torch.ones(2, 2, dtype=torch.int32),
                2,
                group=group,
            )
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            merge_tpu_top_k_candidates(
                torch.ones(2, 1, 3),
                torch.ones(2, 1, 3, dtype=torch.int32),
                2,
                group=group,
            )
        with self.assertRaisesRegex(ValueError, "must remain int32"):
            merge_tpu_top_k_candidates(
                torch.ones(2, 3),
                torch.ones(2, 3, dtype=torch.int64),
                2,
                group=group,
            )


if __name__ == "__main__":
    unittest.main()
