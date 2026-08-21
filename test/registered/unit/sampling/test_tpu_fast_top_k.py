"""CPU-side coverage for TPU distributed exact top-k dispatch and merging.
Tests enforce conservative fallbacks without importing physical-TPU dependencies."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import sampler as tpu_sampler_module
from sglang.srt.hardware_backend.tpu import top_k as tpu_top_k_module
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
    supports_pallas_distributed_tpu_top_k,
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
        max_top_k=50,
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
    @staticmethod
    def _canonical_shard(rank=0):
        return TpuVocabShard(
            width=128,
            global_vocab_size=1024,
            org_vocab_size=1024,
            org_vocab_start=rank * 128,
            org_vocab_count=128,
            org_vocab_padded_count=128,
            added_vocab_start=1024,
            added_vocab_count=0,
        )

    def test_multihost_pallas_eligibility_is_full_world_and_fail_closed(self):
        shard = self._canonical_shard(rank=3)
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.override(False),
        ):
            self.assertTrue(
                supports_pallas_distributed_tpu_top_k(
                    shard, 50, group_world_size=8, model_rank=3
                )
            )
            self.assertFalse(
                supports_pallas_distributed_tpu_top_k(
                    shard, 65, group_world_size=8, model_rank=3
                )
            )
            self.assertFalse(
                supports_pallas_distributed_tpu_top_k(
                    shard, 50, group_world_size=4, model_rank=3
                )
            )

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.override(True),
        ):
            self.assertFalse(
                supports_pallas_distributed_tpu_top_k(
                    shard, 50, group_world_size=8, model_rank=3
                )
            )

    def test_required_pallas_never_reaches_collective_fallback(self):
        group = SimpleNamespace(
            world_size=16,
            rank_in_group=0,
            all_gather=MagicMock(),
        )
        with (
            patch.object(tpu_top_k_module, "_validate_tpu_sharded_top_k"),
            patch.object(
                tpu_top_k_module,
                "supports_pallas_distributed_tpu_top_k",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "lost Pallas eligibility"):
                distributed_tpu_top_k(
                    torch.ones(64, 128),
                    self._canonical_shard(),
                    50,
                    group=group,
                    require_pallas=True,
                )
        group.all_gather.assert_not_called()

    def test_tp16_small_batches_can_lower_cutoff(self):
        processor = SimpleNamespace(
            use_tpu_distributed_top_k=True,
            do_tensor_parallel_all_gather=True,
            do_tensor_parallel_all_gather_dp_attn=False,
            require_pallas_distributed_top_k=False,
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
                with self.subTest(batch_size=batch_size):
                    batch = _forward_batch(batch_size=batch_size)
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
            require_pallas_distributed_top_k=False,
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
            "large top-k": {"uniform_top_k": 65, "max_top_k": 65},
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
                        _forward_batch(**overrides), sampling_backend="tpu"
                    )
                )

    def test_mixed_top_k_and_greedy_rows_use_one_candidate_mode(self):
        batch = _forward_batch(
            uniform_top_k=None,
            max_top_k=64,
            is_any_greedy=True,
            top_ks=torch.tensor([1, 7, 64], dtype=torch.int32),
            temperatures=torch.ones(3, 1),
        )

        self.assertEqual(distributed_top_k_for_batch(batch, sampling_backend="tpu"), 64)

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

    def test_multihost_uses_only_static_pallas_eligible_shards(self):
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_fp32_lm_head=False, enable_mis=False),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
            kernel=SimpleNamespace(sampling_backend="tpu"),
        )
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            tp_size=16,
            tp_rank=0,
            attn_dp_size=1,
            nnodes=2,
        )
        shard_indices = SimpleNamespace(
            org_vocab_start_index=0,
            num_org_elements=128,
            num_org_elements_padded=128,
            added_vocab_start_index=1024,
            num_added_elements=0,
        )
        lm_head = SimpleNamespace(
            enable_tp=True,
            num_embeddings=1024,
            org_vocab_size=1024,
            num_embeddings_per_partition=128,
            shard_indices=shard_indices,
        )

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16"}),
            envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.override(False),
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
            patch.object(tpu_top_k_module, "register_tpu_top_k_ops") as register,
        ):
            processor = LogitsProcessor(SimpleNamespace(vocab_size=1024))
            self.assertEqual(
                processor.get_tpu_distributed_top_k_for_batch(
                    _forward_batch(batch_size=64), lm_head
                ),
                64,
            )
            shard_indices.org_vocab_start_index = 1
            self.assertIsNone(
                processor.get_tpu_distributed_top_k_for_batch(
                    _forward_batch(batch_size=64), lm_head
                )
            )

        self.assertTrue(processor.require_pallas_distributed_top_k)
        register.assert_called_once_with()

    def test_pallas_kill_switch_falls_back_before_local_logits(self):
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_fp32_lm_head=False, enable_mis=False),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
            kernel=SimpleNamespace(sampling_backend="tpu"),
        )
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            tp_size=16,
            tp_rank=0,
            attn_dp_size=1,
            nnodes=2,
        )

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16"}),
            envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.override(True),
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
            patch.object(tpu_top_k_module, "register_tpu_top_k_ops") as register,
        ):
            processor = LogitsProcessor(SimpleNamespace(vocab_size=1024))

        self.assertFalse(processor.use_tpu_distributed_top_k)
        register.assert_not_called()


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

    def test_global_top_k_sampling_masks_each_rows_requested_tail(self):
        logits = torch.tensor([[4.0, 3.0, 2.0], [6.0, 5.0, 4.0]])
        top_ks = torch.tensor([1, 2], dtype=torch.int32)
        with patch.object(
            tpu_sampler_module,
            "sampling_from_probs_torch",
            return_value=torch.tensor([0, 1], dtype=torch.int32),
        ) as sample:
            sample_from_global_top_k_logits(logits, torch.ones(2, 1), top_ks)

        probs = sample.call_args.args[0]
        torch.testing.assert_close(probs[0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(probs[1, 2], torch.tensor(0.0))
        torch.testing.assert_close(probs[1, :2].sum(), torch.tensor(1.0))

    def test_all_greedy_uses_first_candidate_without_sampling(self):
        sampler = SimpleNamespace(_sync_token_ids_across_tp=MagicMock())
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_global_top_k_logits=torch.tensor([[3.0, 2.0]]),
            next_token_global_top_k_token_ids=torch.tensor([[91, 17]]),
        )
        info = _forward_batch(
            is_all_greedy=True,
            is_any_greedy=True,
            uniform_top_k=1,
            max_top_k=1,
            top_ks=torch.ones(1, dtype=torch.int32),
            temperatures=torch.ones(1, 1),
        ).sampling_info

        with patch.object(
            tpu_sampler_module,
            "sample_from_global_top_k_logits",
        ) as sample:
            actual = TpuSampler.forward(
                sampler, output, info, False, [0], [[]], torch.tensor([7])
            )

        torch.testing.assert_close(actual, torch.tensor([91], dtype=torch.int64))
        sample.assert_not_called()

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
    def test_tied_max_prefers_the_lowest_global_token_id(self):
        candidate_logits = torch.tensor([[9.0, 9.0, 8.0]])
        candidate_token_ids = torch.tensor([[17, 3, 8]], dtype=torch.int32)

        corrected = _prefer_lowest_token_id_for_tied_max(
            candidate_logits,
            candidate_token_ids,
            torch.tensor([[17, 3]], dtype=torch.int32),
        )

        torch.testing.assert_close(
            corrected,
            torch.tensor([[3, 17]], dtype=torch.int32),
        )

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
            org_vocab_size=20,
            num_embeddings_per_partition=12,
        )

        self.assertEqual(
            TpuVocabShard.from_lm_head(lm_head),
            TpuVocabShard(
                width=12,
                global_vocab_size=22,
                org_vocab_size=20,
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

    def test_pallas_dispatch_supports_tp8_tp16_bf16_fp32(self):
        expected = (
            torch.randn(2, 2, dtype=torch.float32),
            torch.randint(0, 1_000, (2, 2), dtype=torch.int32),
        )
        for world_size in (8, 16):
            for dtype in (torch.bfloat16, torch.float32):
                rank = 3
                org_vocab_size = world_size * 128 - 24
                shard = TpuVocabShard(
                    width=128,
                    global_vocab_size=org_vocab_size,
                    org_vocab_size=org_vocab_size,
                    org_vocab_start=rank * 128,
                    org_vocab_count=128,
                    org_vocab_padded_count=128,
                    added_vocab_start=org_vocab_size,
                    added_vocab_count=0,
                )
                local_logits = torch.randn(2, 128, dtype=dtype)
                group = SimpleNamespace(world_size=world_size, rank_in_group=rank)

                with (
                    self.subTest(world_size=world_size, dtype=dtype),
                    patch.dict(
                        tpu_top_k_module.os.environ,
                        {"WORLD_SIZE": str(world_size)},
                    ),
                    envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.override(False),
                    patch.object(tpu_top_k_module, "_validate_tpu_sharded_top_k"),
                    patch(
                        "sglang.srt.hardware_backend.tpu.distributed_topk_pallas."
                        "pallas_distributed_top_k",
                        return_value=expected,
                    ) as pallas,
                ):
                    actual = distributed_tpu_top_k(
                        local_logits,
                        shard,
                        2,
                        group=group,
                    )

                self.assertIs(actual, expected)
                pallas.assert_called_once_with(
                    local_logits,
                    org_vocab_size=org_vocab_size,
                    global_vocab_size=org_vocab_size,
                    org_vocab_padded_count=128,
                    k=2,
                    world_size=world_size,
                    model_rank=rank,
                )

    def test_pallas_dispatch_preserves_conservative_fallbacks(self):
        shard = TpuVocabShard(
            width=128,
            global_vocab_size=1_000,
            org_vocab_size=1_000,
            org_vocab_start=384,
            org_vocab_count=128,
            org_vocab_padded_count=128,
            added_vocab_start=1_000,
            added_vocab_count=0,
        )
        default_logits = torch.randn(2, 128)
        default_group = SimpleNamespace(world_size=8, rank_in_group=3)
        cases = (
            ("kill switch", default_logits, shard, 2, default_group, "8", True),
            (
                "unsupported dtype",
                default_logits.half(),
                shard,
                2,
                default_group,
                "8",
                False,
            ),
            (
                "unsupported topology",
                default_logits,
                shard,
                2,
                SimpleNamespace(world_size=4, rank_in_group=3),
                "4",
                False,
            ),
            (
                "group mismatch",
                default_logits,
                shard,
                2,
                default_group,
                "16",
                False,
            ),
            (
                "missing group rank",
                default_logits,
                shard,
                2,
                SimpleNamespace(world_size=8),
                "8",
                False,
            ),
            (
                "noncanonical shard",
                default_logits,
                shard._replace(org_vocab_start=383),
                2,
                default_group,
                "8",
                False,
            ),
            ("k exceeds Top64", default_logits, shard, 65, default_group, "8", False),
        )
        expected = (torch.randn(2, 2), torch.ones(2, 2, dtype=torch.int32))

        for name, logits, case_shard, k, group, world_size, disabled in cases:
            with (
                self.subTest(name=name),
                patch.dict(
                    tpu_top_k_module.os.environ,
                    {"WORLD_SIZE": world_size},
                ),
                envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.override(disabled),
                patch.object(tpu_top_k_module, "_validate_tpu_sharded_top_k"),
                patch.object(
                    tpu_top_k_module,
                    "distributed_tpu_top_k_collective",
                    return_value=expected,
                ) as fallback,
                patch(
                    "sglang.srt.hardware_backend.tpu.distributed_topk_pallas."
                    "pallas_distributed_top_k",
                ) as pallas,
            ):
                actual = distributed_tpu_top_k(
                    logits,
                    case_shard,
                    k,
                    group=group,
                )

            self.assertIs(actual, expected)
            fallback.assert_called_once_with(logits, case_shard, k, group=group)
            pallas.assert_not_called()

    def test_distributed_validation_runs_before_pallas_dispatch(self):
        shard = TpuVocabShard(
            width=128,
            global_vocab_size=1_000,
            org_vocab_size=1_000,
            org_vocab_start=384,
            org_vocab_count=128,
            org_vocab_padded_count=128,
            added_vocab_start=1_000,
            added_vocab_count=0,
        )
        with (
            patch.dict(tpu_top_k_module.os.environ, {"WORLD_SIZE": "8"}),
            envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.override(False),
            patch(
                "sglang.srt.hardware_backend.tpu.distributed_topk_pallas."
                "pallas_distributed_top_k",
            ) as pallas,
            self.assertRaisesRegex(ValueError, "requires a TPU tensor"),
        ):
            distributed_tpu_top_k(
                torch.randn(2, 128),
                shard,
                2,
                group=SimpleNamespace(world_size=8, rank_in_group=3),
            )
        pallas.assert_not_called()

    def test_distributed_merge_matches_full_top_k(self):
        local_logits = torch.tensor([[9.0, 3.0, -2.0, 7.0]])
        shard = TpuVocabShard(
            width=4,
            global_vocab_size=8,
            org_vocab_size=3,
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
            patch.object(tpu_top_k_module, "_validate_tpu_sharded_top_k"),
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
