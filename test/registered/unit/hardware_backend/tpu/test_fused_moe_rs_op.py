# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS Torch wrapper."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.pallas import fused_moe_rs_op
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestFusedMoeRsOp(CustomTestCase):
    def test_padding_satisfies_ep_and_routed_row_alignment(self):
        cases = (
            # num_tokens, topk, ep_size, padded_tokens
            (8, 1, 8, 16),
            (4, 2, 4, 8),
            (16, 8, 16, 16),
            (17, 8, 16, 32),
        )
        for num_tokens, topk, ep_size, expected in cases:
            with self.subTest(num_tokens=num_tokens, topk=topk, ep_size=ep_size):
                actual = fused_moe_rs_op.fused_moe_rs_padded_tokens(
                    num_tokens, topk, ep_size
                )
                self.assertEqual(actual, expected)
                self.assertEqual(actual % ep_size, 0)
                self.assertEqual((actual * topk) % 16, 0)

    def test_prepare_uses_runtime_topology_for_tp16(self):
        layer = torch.nn.Module()
        layer.num_local_experts = 8
        layer.w13_weight = torch.empty(1)
        routes = (8, 9, 12, 13, 14, 15, 10, 11, 0, 1, 4, 5, 6, 7, 2, 3)
        cpu_group = object()
        tp_group = SimpleNamespace(ranks=list(range(16)), cpu_group=cpu_group)

        with (
            patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
            patch(
                "sglang.srt.hardware_backend.tpu.pallas_export."
                "get_tpu_group_rank_to_device_id",
                return_value=routes,
            ) as get_routes,
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(moe_ep_rank=11),
            ),
            patch.object(fused_moe_rs_op, "_build_fused_moe_rs_op") as build,
        ):
            fused_moe_rs_op.prepare_fused_moe_rs(layer)

        get_routes.assert_called_once_with(tuple(range(16)), cpu_group)
        torch.testing.assert_close(
            layer._tpu_fused_rs_tp_rank_to_device_id,
            torch.tensor(routes, dtype=torch.int32),
        )
        torch.testing.assert_close(
            layer._tpu_fused_rs_group_offset,
            torch.tensor([88, 0], dtype=torch.int32),
        )
        self.assertIn("_tpu_fused_rs_tp_rank_to_device_id", layer._buffers)
        self.assertIn("_tpu_fused_rs_group_offset", layer._buffers)
        self.assertIn(
            "_tpu_fused_rs_tp_rank_to_device_id",
            layer._non_persistent_buffers_set,
        )
        self.assertIn(
            "_tpu_fused_rs_group_offset", layer._non_persistent_buffers_set
        )
        build.assert_called_once_with(routes)

    def test_gather_preserves_tp_group_rank_order(self):
        rank_order_shards = torch.arange(8).reshape(8, 1)
        rank_to_device_id = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])
        tp_group = Mock(
            world_size=8,
            all_gather=Mock(return_value=rank_order_shards),
        )
        token_shard = torch.zeros(1, 1)

        with (
            patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
            patch.object(
                fused_moe_rs_op,
                "fused_moe_rs_call",
                return_value=token_shard,
            ) as fused_call,
        ):
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=torch.zeros(8, 1),
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=torch.empty(1),
                topk_ids=torch.empty(1),
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=1,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=True,
                tp_rank_to_device_id=rank_to_device_id,
            )

        torch.testing.assert_close(output, rank_order_shards)
        self.assertIs(
            fused_call.call_args.kwargs["tp_rank_to_device_id"],
            rank_to_device_id,
        )
        tp_group.all_gather.assert_called_once_with(token_shard, dim=0)


if __name__ == "__main__":
    unittest.main()
