"""CPU-side coverage for DFlash TPU distributed sampling dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.models import dflash as dflash_module
from sglang.srt.models.dflash import DFlashDraftModel


class TestDFlashTpuSampling(CustomTestCase):
    def _model(self, *, mapping, tp_size=2):
        return SimpleNamespace(
            block_size=2,
            lm_head=SimpleNamespace(
                weight=torch.eye(2, dtype=torch.bfloat16),
                enable_tp=tp_size > 1,
                tp_size=tp_size,
            ),
            _tpu_full_vocab_indices=mapping,
            _tpu_full_vocab_group=None,
            _tpu_distributed_top_k_initialized=True,
        )

    def test_target_lm_head_prepares_full_vocab_contract(self):
        model = torch.nn.Module()
        weight = torch.empty(3, 2, dtype=torch.bfloat16)
        lm_head = SimpleNamespace(
            weight=weight,
            num_embeddings=4,
            enable_tp=True,
            tp_size=2,
            get_sharded_to_full_mapping=lambda: [0, 1, 3, 4, 2, 5],
        )
        group = SimpleNamespace(world_size=2)

        with (
            patch.object(dflash_module, "_is_tpu", True),
            patch.object(
                dflash_module, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(
                dflash_module, "get_tp_group", return_value=group
            ) as get_tp_group,
        ):
            DFlashDraftModel.set_lm_head_from_target(model, lm_head)

        self.assertIs(model.lm_head, lm_head)
        self.assertIs(model._tpu_full_vocab_group, group)
        torch.testing.assert_close(
            model._tpu_full_vocab_indices,
            torch.tensor([0, 1, 3, 4], dtype=torch.long),
        )
        get_tp_group.assert_called_once()

    def test_ineligible_tp_uses_exact_full_vocabulary_argmax(self):
        mapping = torch.tensor([0, 1, 3, 4], dtype=torch.long)
        model = self._model(mapping=mapping)
        hidden_states = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        gathered_logits = torch.tensor(
            [[1.0, 2.0, -100.0, 3.0, 9.0, -100.0]],
            dtype=torch.bfloat16,
        )
        group = SimpleNamespace(
            world_size=2,
            all_gather=MagicMock(return_value=gathered_logits),
        )
        model._tpu_full_vocab_group = group

        actual = DFlashDraftModel._sample_draft_tokens_tpu(model, hidden_states)

        torch.testing.assert_close(actual, torch.tensor([[3]], dtype=torch.long))
        group.all_gather.assert_called_once()
        self.assertEqual(group.all_gather.call_args.kwargs, {"dim": -1})

    def test_eligible_tp_uses_the_pallas_distributed_kernel(self):
        config = SimpleNamespace(output_dtype=torch.bfloat16)
        model = self._model(mapping=torch.tensor([0, 1, 3, 4]))
        model._tpu_distributed_top_k_config = config
        hidden_states = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        group = SimpleNamespace(world_size=2, all_gather=MagicMock())
        token_ids = torch.tensor([[17]], dtype=torch.int32)

        with (
            patch.object(dflash_module, "get_tp_group") as get_tp_group,
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.distributed_tpu_top_k",
                return_value=(torch.ones(1, 1), token_ids),
            ) as distributed_top_k,
        ):
            actual = DFlashDraftModel._sample_draft_tokens_tpu(model, hidden_states)

        torch.testing.assert_close(actual, torch.tensor([[17]], dtype=torch.long))
        distributed_top_k.assert_called_once()
        self.assertIs(distributed_top_k.call_args.args[1], config)
        self.assertEqual(distributed_top_k.call_args.kwargs, {"k": 1})
        get_tp_group.assert_not_called()
        group.all_gather.assert_not_called()

    def test_target_pallas_contract_is_reused_before_draft_compile(self):
        model = torch.nn.Module()
        model._tpu_distributed_top_k_config = None
        model._tpu_distributed_top_k_init_key = None
        model._tpu_distributed_top_k_initialized = False
        target_config = SimpleNamespace(world_size=8, output_dtype=torch.bfloat16)
        lm_head = SimpleNamespace(
            weight=torch.empty(128, 2, dtype=torch.bfloat16),
            num_embeddings=128,
            enable_tp=True,
            tp_size=8,
            get_sharded_to_full_mapping=lambda: list(range(128)),
        )
        group = SimpleNamespace(
            world_size=8,
            rank_in_group=0,
            ranks=list(range(8)),
        )

        with (
            patch.object(dflash_module, "_is_tpu", True),
            patch.object(
                dflash_module, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(dflash_module, "get_tp_group", return_value=group),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k."
                "pallas_distributed_top_k_config_matches",
                return_value=True,
            ) as matches,
            patch(
                "sglang.srt.hardware_backend.tpu.top_k."
                "negotiate_pallas_distributed_top_k_config"
            ) as negotiate,
            patch(
                "sglang.srt.hardware_backend.tpu.top_k.register_tpu_top_k_ops"
            ) as register,
        ):
            DFlashDraftModel.set_lm_head_from_target(
                model,
                lm_head,
                target_config=target_config,
            )

        self.assertIs(model._tpu_distributed_top_k_config, target_config)
        self.assertTrue(model._tpu_distributed_top_k_initialized)
        matches.assert_called_once_with(
            target_config,
            lm_head,
            group=group,
            expected_dtype=torch.bfloat16,
        )
        negotiate.assert_not_called()
        register.assert_called_once_with(8)

    def test_target_pallas_contract_mismatch_is_fatal_without_a_collective(self):
        model = torch.nn.Module()
        model._tpu_distributed_top_k_initialized = False
        target_config = SimpleNamespace(world_size=8, output_dtype=torch.bfloat16)
        lm_head = SimpleNamespace(
            weight=torch.empty(128, 2, dtype=torch.bfloat16),
            num_embeddings=128,
            enable_tp=True,
            tp_size=8,
            get_sharded_to_full_mapping=lambda: list(range(128)),
        )
        group = SimpleNamespace(
            world_size=8,
            rank_in_group=0,
            ranks=list(range(8)),
        )

        with (
            patch.object(dflash_module, "_is_tpu", True),
            patch.object(
                dflash_module, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(dflash_module, "get_tp_group", return_value=group),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k."
                "pallas_distributed_top_k_config_matches",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.top_k."
                "negotiate_pallas_distributed_top_k_config"
            ) as negotiate,
            self.assertRaisesRegex(RuntimeError, "cannot reuse"),
        ):
            DFlashDraftModel.set_lm_head_from_target(
                model,
                lm_head,
                target_config=target_config,
            )

        negotiate.assert_not_called()
        self.assertFalse(model._tpu_distributed_top_k_initialized)

    def test_full_vocab_fallback_compiles_as_one_graph(self):
        class FullVocabGroup:
            @staticmethod
            def all_gather(values, dim=-1):
                return torch.cat((values, values), dim=dim)

        model = self._model(mapping=torch.arange(4), tp_size=2)
        model._tpu_full_vocab_group = FullVocabGroup()
        hidden_states = torch.tensor(
            [[0.0, 0.0], [1.0, 2.0]],
            dtype=torch.bfloat16,
        )

        compiled = torch.compile(
            lambda states: DFlashDraftModel._sample_draft_tokens_tpu(model, states),
            backend="eager",
            fullgraph=True,
        )

        torch.testing.assert_close(
            compiled(hidden_states),
            torch.tensor([[1]], dtype=torch.long),
        )

    def test_pallas_dispatch_compiles_as_one_graph(self):
        config = SimpleNamespace(output_dtype=torch.float32)
        model = self._model(mapping=torch.arange(4), tp_size=2)
        model._tpu_distributed_top_k_config = config
        hidden_states = torch.tensor(
            [[0.0, 0.0], [1.0, 2.0]],
            dtype=torch.bfloat16,
        )

        def fake_distributed_top_k(local_logits, actual_config, k):
            del actual_config
            values, token_ids = torch.topk(local_logits.float(), k, dim=-1)
            return values, token_ids.to(torch.int32)

        with patch(
            "sglang.srt.hardware_backend.tpu.top_k.distributed_tpu_top_k",
            new=fake_distributed_top_k,
        ):
            compiled = torch.compile(
                lambda states: DFlashDraftModel._sample_draft_tokens_tpu(model, states),
                backend="eager",
                fullgraph=True,
            )
            actual = compiled(hidden_states)

        torch.testing.assert_close(actual, torch.tensor([[1]], dtype=torch.long))

    def test_sharded_argmax_preserves_lowest_token_id_on_ties(self):
        model = self._model(mapping=torch.tensor([0, 1, 3, 4], dtype=torch.long))
        hidden_states = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        gathered_logits = torch.ones(1, 6, dtype=torch.bfloat16)
        group = SimpleNamespace(
            world_size=2,
            all_gather=MagicMock(return_value=gathered_logits),
        )
        model._tpu_full_vocab_group = group

        actual = DFlashDraftModel._sample_draft_tokens_tpu(model, hidden_states)

        torch.testing.assert_close(actual, torch.tensor([[0]], dtype=torch.long))

    def test_replicated_lm_head_ignores_ambient_tp_group(self):
        model = torch.nn.Module()
        lm_head = SimpleNamespace(
            weight=torch.empty(4, 2, dtype=torch.bfloat16),
            num_embeddings=3,
            enable_tp=False,
            tp_size=1,
            shard_indices=SimpleNamespace(
                num_org_elements=2,
                num_org_elements_padded=3,
                num_added_elements=1,
            ),
            get_sharded_to_full_mapping=lambda: None,
        )
        ambient_group = SimpleNamespace(world_size=8)

        with (
            patch.object(dflash_module, "_is_tpu", True),
            patch.object(
                dflash_module, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(
                dflash_module, "get_tp_group", return_value=ambient_group
            ) as get_tp_group,
        ):
            DFlashDraftModel.set_lm_head_from_target(model, lm_head)

        get_tp_group.assert_not_called()
        torch.testing.assert_close(
            model._tpu_full_vocab_indices,
            torch.tensor([0, 1, 3], dtype=torch.long),
        )

        model.block_size = 2
        lm_head.weight = torch.eye(2, dtype=torch.bfloat16).repeat(2, 1)
        hidden_states = torch.tensor(
            [[0.0, 0.0], [1.0, 2.0]],
            dtype=torch.bfloat16,
        )
        with (
            patch.object(
                dflash_module, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(dflash_module, "get_tp_group") as get_tp_group,
        ):
            actual = DFlashDraftModel._sample_draft_tokens_tpu(model, hidden_states)

        get_tp_group.assert_not_called()
        torch.testing.assert_close(actual, torch.tensor([[1]], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
