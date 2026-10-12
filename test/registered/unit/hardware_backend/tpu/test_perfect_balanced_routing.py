"""Registered coverage for TPU perfect-balanced expert simulation."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.hardware_backend.tpu.fused_rs_sequence_parallel import (
    FusedMoeSequenceParallelPlan,
    use_fused_moe_sequence_parallel_plan,
)
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _simulate_tpu_perfect_balanced_routing,
    _tpu_perfect_balanced_routing_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _select_routing(
    num_tokens,
    valid_tokens,
    *,
    tp_rank,
    sharded,
    dp_rank=0,
    dp_size=1,
    layer_id=17,
    policy="perfect_balanced",
    device="cpu",
):
    def router(hidden_states, gating_output, topk, renormalize):
        return (
            torch.full((num_tokens, topk), 0.25, device=device),
            torch.full((num_tokens, topk), 7, dtype=torch.int32, device=device),
        )

    parallel = SimpleNamespace(
        attn_tp_group=SimpleNamespace(rank_in_group=tp_rank, world_size=16),
        attn_dp_rank=dp_rank,
        attn_dp_size=dp_size,
        moe_ep_size=16,
    )
    config = TopKConfig(
        top_k=8,
        custom_routing_function=router,
        torch_native=True,
        allow_routed_experts_capture=False,
    )
    plan = FusedMoeSequenceParallelPlan(sharded, sharded, True)
    with (
        use_fused_moe_sequence_parallel_plan(plan),
        patch.object(topk_module, "_is_tpu", True),
        patch.object(topk_module, "_is_cuda", False),
        patch.object(topk_module, "_is_hip", False),
        patch.object(topk_module, "get_parallel", return_value=parallel),
        patch.object(
            topk_module,
            "is_moe_input_scattered_across_dp_ranks",
            return_value=dp_size > 1,
        ),
        patch.object(
            topk_module.envs.SGLANG_TPU_SIMULATE_PERFECT_BALANCED_EXPERTS,
            "get",
            return_value=policy == "perfect_balanced",
        ),
        patch.object(
            topk_module.envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS,
            "get",
            return_value=policy == "round_robin",
        ),
        patch.object(
            topk_module.envs.SGLANG_SIMULATE_UNIFORM_EXPERTS,
            "get",
            return_value=False,
        ),
    ):
        return topk_module.select_experts(
            torch.zeros((num_tokens, 4), device=device),
            torch.zeros((num_tokens, 256), device=device),
            config,
            layer_id=layer_id,
            num_token_non_padded=torch.tensor(
                valid_tokens, device=device, dtype=torch.int32
            ),
        )


def test_tpu_flag_is_ignored_on_other_platforms():
    with (
        patch("sglang.srt.layers.moe.topk._is_tpu", False),
        patch(
            "sglang.srt.layers.moe.topk.envs."
            "SGLANG_TPU_SIMULATE_PERFECT_BALANCED_EXPERTS.get",
            return_value=True,
        ),
    ):
        assert not _tpu_perfect_balanced_routing_enabled()


def test_tpu_flag_is_enabled_on_tpu():
    with (
        patch("sglang.srt.layers.moe.topk._is_tpu", True),
        patch(
            "sglang.srt.layers.moe.topk.envs."
            "SGLANG_TPU_SIMULATE_PERFECT_BALANCED_EXPERTS.get",
            return_value=True,
        ),
    ):
        assert _tpu_perfect_balanced_routing_enabled()


def test_tpu_perfect_balanced_routing_spreads_assignments_across_ranks():
    num_experts, topk, ep_size = 256, 8, 16
    for num_tokens in (1, 2, 4, 300):
        ids = torch.full((num_tokens, topk), -1, dtype=torch.int32)
        weights = torch.zeros((num_tokens, topk), dtype=torch.float32)

        _simulate_tpu_perfect_balanced_routing(
            ids,
            weights,
            num_experts,
            num_ranks=ep_size,
            layer_id=0,
        )

        counts = torch.bincount(
            ids.flatten().long() // (num_experts // ep_size), minlength=ep_size
        )
        assert int(counts.max() - counts.min()) <= 1
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / topk))


def test_tpu_perfect_balanced_routing_uses_routed_topk_as_assignment_stride():
    num_experts, routed_topk, total_topk, ep_size = 256, 8, 9, 16
    ids = torch.full((2, total_topk), -1, dtype=torch.int32)
    weights = torch.zeros((2, total_topk), dtype=torch.float32)

    _simulate_tpu_perfect_balanced_routing(
        ids,
        weights,
        num_experts,
        num_ranks=ep_size,
        num_routed_topk=routed_topk,
        layer_id=0,
    )

    counts = torch.bincount(
        ids[:, :routed_topk].flatten().long() // (num_experts // ep_size),
        minlength=ep_size,
    )
    assert torch.equal(counts, torch.ones_like(counts))


@pytest.mark.parametrize("num_tokens", (64, 128))
@pytest.mark.parametrize("dp_size", (1, 3))
@pytest.mark.parametrize("valid_fraction", (0, 0.5, 1))
def test_native_routing_concatenated_tp_shards_match_gathered(
    num_tokens, dp_size, valid_fraction
):
    local_rows = num_tokens // 16
    valid_tokens = (
        num_tokens // 2 + 1
        if valid_fraction == 0.5
        else int(num_tokens * valid_fraction)
    )
    for dp_rank in range(dp_size):
        # An active SP plan can still present globally gathered routing rows.
        expected = _select_routing(
            num_tokens,
            valid_tokens,
            tp_rank=8,
            sharded=False,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        shards = [
            _select_routing(
                local_rows,
                min(max(valid_tokens - rank * local_rows, 0), local_rows),
                tp_rank=rank,
                sharded=True,
                dp_rank=dp_rank,
                dp_size=dp_size,
            )
            for rank in range(16)
        ]
        actual_ids = torch.cat([shard.topk_ids for shard in shards])
        actual_weights = torch.cat([shard.topk_weights for shard in shards])
        assert torch.equal(actual_ids, expected.topk_ids)
        assert torch.equal(actual_weights, expected.topk_weights)
        assert torch.equal(
            actual_ids[valid_tokens:], torch.full_like(actual_ids[valid_tokens:], -1)
        )
        assert torch.equal(
            actual_weights[valid_tokens:],
            torch.zeros_like(actual_weights[valid_tokens:]),
        )
        if dp_size == 1 and valid_tokens == num_tokens:
            counts = torch.bincount(actual_ids.flatten().long(), minlength=256)
            assert torch.equal(counts, torch.full_like(counts, num_tokens * 8 // 256))


def test_native_round_robin_receives_tp_offset_before_dp_interleaving():
    with patch.object(topk_module, "_simulate_balanced_routing") as simulate:
        _select_routing(
            8, 1, tp_rank=8, sharded=True, dp_rank=2, dp_size=3, policy="round_robin"
        )
    assert simulate.call_count == 1
    assert simulate.call_args.kwargs["token_offset"] == 64
    assert simulate.call_args.kwargs["token_shard_rank"] == 2
    assert simulate.call_args.kwargs["num_token_shards"] == 3


def test_unsimulated_native_routing_is_unchanged_on_tp_shards():
    with (
        patch.object(topk_module, "_simulate_balanced_routing") as balanced,
        patch.object(topk_module, "_simulate_tpu_perfect_balanced_routing") as perfect,
    ):
        output = _select_routing(8, 3, tp_rank=8, sharded=True, policy="native")
    balanced.assert_not_called()
    perfect.assert_not_called()
    assert torch.equal(output.topk_ids[:3], torch.full((3, 8), 7, dtype=torch.int32))
    assert torch.equal(output.topk_weights[:3], torch.full((3, 8), 0.25))
    assert torch.equal(output.topk_ids[3:], torch.full((5, 8), -1, dtype=torch.int32))
    assert torch.equal(output.topk_weights[3:], torch.zeros(5, 8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
