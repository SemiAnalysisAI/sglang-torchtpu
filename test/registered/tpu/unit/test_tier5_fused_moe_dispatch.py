# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=600, suite="stage-a-test-tpu-unit")

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_dispatch_sp import (
    fused_sp_moe_dispatch_metadata,
    should_use_fused_sp_moe_dispatch_metadata,
)


class TestSupportsSpFusedRsRouting:
    @pytest.mark.parametrize(
        ("local_assignments", "expected"),
        [(64, False), (128, True), (2048, True), (4096, True), (8192, True)],
    )
    def test_uses_measured_global_assignment_crossover(
        self, local_assignments, expected
    ):
        topk_indices = jnp.zeros((local_assignments,), dtype=jnp.uint8)

        assert (
            should_use_fused_sp_moe_dispatch_metadata(
                topk_indices.size,
                num_experts=256,
                ep_size=16,
            )
            is expected
        )


class TestFusedSpMoeDispatchMetadata:
    def test_rejects_non_power_of_two_local_assignments(self):
        with pytest.raises(ValueError, match="power of two"):
            fused_sp_moe_dispatch_metadata(
                jnp.zeros((17, 8), dtype=jnp.int32),
                num_experts=256,
                topk=8,
                ep_size=16,
                ep_axis_name="model",
            )

    @pytest.mark.parametrize("global_tokens", [256, 8192])
    def test_matches_global_stable_routing(self, global_tokens):
        devices = jax.devices()
        if len(devices) != 16 or "TPU7" not in devices[0].device_kind:
            pytest.skip("requires a TP16 TPU7x slice")

        mesh = jax.sharding.Mesh(np.asarray(devices), ("model",))
        topk = 8
        token_sharding = NamedSharding(mesh, P("model", None))

        @jax.jit
        def make_indices():
            token = jnp.arange(global_tokens, dtype=jnp.int32)[:, None]
            slot = jnp.arange(topk, dtype=jnp.int32)[None, :]
            return (token * 37 + slot * 53 + token // 11) % 256

        indices = jax.device_put(make_indices(), token_sharding)

        def run(local_indices):
            local_indices = local_indices.reshape(-1)
            actual = fused_sp_moe_dispatch_metadata(
                local_indices,
                num_experts=256,
                topk=topk,
                ep_size=16,
                ep_axis_name="model",
            )
            gathered = jax.lax.all_gather(local_indices, "model", axis=0, tiled=True)
            gathered_flat = gathered.reshape(-1)
            order = jnp.argsort(gathered_flat)
            expert_ids = jnp.arange(256, dtype=jnp.int32)
            group_sizes = jnp.sum(
                (gathered_flat[:, None] == expert_ids[None, :]).astype(jnp.int32),
                axis=0,
            )
            lhs_indices = order // topk
            output_indices = lhs_indices
            topk_slot_indices = order % topk
            expected_routing = (
                lhs_indices,
                group_sizes,
                output_indices,
                topk_slot_indices,
            )
            rank = jax.lax.axis_index("model")
            expert_ids = jnp.arange(256, dtype=jnp.int32)
            first_local_expert = rank * 16
            group_start = jnp.sum(
                jnp.where(expert_ids < first_local_expert, group_sizes, 0)
            )
            group_end = group_start + jnp.sum(
                jnp.where(
                    (expert_ids >= first_local_expert)
                    & (expert_ids < first_local_expert + 16),
                    group_sizes,
                    0,
                )
            )
            assignment_rows = jnp.arange(output_indices.size, dtype=jnp.int32)
            destination_ranks = output_indices // (global_tokens // 16)
            total_recv_count = jnp.sum(
                (
                    (destination_ranks == rank)
                    & ((assignment_rows < group_start) | (assignment_rows >= group_end))
                ).astype(jnp.int32)
            )
            expected = (
                *expected_routing,
                jnp.full((128,), group_start, dtype=jnp.int32),
                jnp.full((128,), total_recv_count, dtype=jnp.int32),
            )
            return actual, expected

        actual, expected = jax.jit(
            jax.shard_map(
                run,
                mesh=mesh,
                in_specs=P("model", None),
                out_specs=(
                    (P(), P(), P(), P(), P("model"), P("model")),
                    (P(), P(), P(), P(), P("model"), P("model")),
                ),
                check_vma=False,
            )
        )(indices)

        for actual_value, expected_value in zip(actual, expected, strict=True):
            actual_shards = actual_value.addressable_shards
            expected_shards = expected_value.addressable_shards
            assert [shard.index for shard in actual_shards] == [
                shard.index for shard in expected_shards
            ]
            for actual_shard, expected_shard in zip(
                actual_shards, expected_shards, strict=True
            ):
                np.testing.assert_array_equal(actual_shard.data, expected_shard.data)
