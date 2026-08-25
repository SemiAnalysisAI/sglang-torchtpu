# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=600, suite="stage-a-test-tpu-unit")

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_router import (
    build_fused_moe_router_fn,
    fused_moe_router,
    ref_fused_moe_router,
)


class TestFusedMoeRouter:
    @pytest.mark.parametrize(
        "tokens,hidden,experts,topk,normalization",
        [
            (7, 256, 8, 2, "sigmoid"),
            (16, 512, 128, 4, "softmax"),
            (64, 1024, 256, 8, "sigmoid"),
        ],
    )
    def test_reference_shapes(self, tokens, hidden, experts, topk, normalization):
        x = jnp.ones((tokens, hidden), dtype=jnp.float32)
        weight = jnp.ones((experts, hidden), dtype=jnp.float32)
        bias = jnp.zeros((experts,), dtype=jnp.float32)

        weights, indices = ref_fused_moe_router(
            x,
            weight,
            bias,
            topk=topk,
            score_normalization=normalization,
        )

        assert weights.shape == (tokens, topk)
        assert weights.dtype == jnp.float32
        assert indices.shape == (tokens, topk)
        assert indices.dtype == jnp.int32

    @pytest.mark.parametrize("normalization", ["sigmoid", "softmax"])
    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas fused router requires a TPU",
    )
    def test_kernel_matches_reference(self, normalization):
        x = jax.random.normal(jax.random.key(0), (16, 256), dtype=jnp.float32)
        weight = jax.random.normal(jax.random.key(1), (128, 256), dtype=jnp.float32)
        bias = jax.random.normal(jax.random.key(2), (128,), dtype=jnp.float32) * 0.01
        kwargs = dict(
            topk=8,
            router_output_multiplier=0.25,
            route_scale=1.25,
            division_eps=1e-6,
            score_normalization=normalization,
        )

        expected_weights, expected_indices = ref_fused_moe_router(
            x, weight, bias, **kwargs
        )
        weights, indices = fused_moe_router(x, weight, bias, **kwargs)

        np.testing.assert_array_equal(np.asarray(indices), np.asarray(expected_indices))
        np.testing.assert_allclose(
            np.asarray(weights), np.asarray(expected_weights), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize(
        ("tokens", "hidden", "input_kind", "seed"),
        [
            (1, 512, "zero", 20),
            (16, 512, "random", 20),
            (256, 512, "random", 20),
            (256, 512, "outlier", 20),
            (256, 8192, "random", 20),
            (256, 8192, "random", 21),
            (256, 8192, "random", 22),
            (256, 8192, "random", 23),
        ],
    )
    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas fused router requires a TPU",
    )
    def test_bf16_kernel_matches_reference_across_inputs(
        self, tokens, hidden, input_kind, seed
    ):
        experts, topk = 256, 8
        shape = (tokens, hidden)
        if input_kind == "zero":
            x = jnp.zeros(shape, dtype=jnp.bfloat16)
        else:
            x = jax.random.normal(jax.random.key(seed), shape, dtype=jnp.bfloat16)
            if input_kind == "outlier":
                x = x.at[::31, ::17].set(jnp.asarray(32, jnp.bfloat16))
        weight = (
            jax.random.normal(
                jax.random.key(seed + 100),
                (experts, hidden),
                dtype=jnp.bfloat16,
            )
            * jnp.asarray(0.01, jnp.bfloat16)
        ).astype(jnp.bfloat16)
        # A monotonic bias makes the zero-input case deterministic instead of a
        # tie-order test; exact ties have separate validity coverage below.
        if input_kind == "zero":
            bias = jnp.linspace(-0.05, 0.05, experts, dtype=jnp.float32)
        else:
            bias = (
                jax.random.normal(
                    jax.random.key(seed + 200), (experts,), dtype=jnp.float32
                )
                * 0.01
            )
        kwargs = dict(
            topk=topk,
            router_output_multiplier=0.125,
            route_scale=1.0,
            division_eps=1e-15,
            score_normalization="sigmoid",
        )

        expected_weights, expected_indices = ref_fused_moe_router(
            x, weight, bias, **kwargs
        )
        weights, indices = fused_moe_router(x, weight, bias, **kwargs)

        np.testing.assert_array_equal(np.asarray(indices), np.asarray(expected_indices))
        np.testing.assert_allclose(
            np.asarray(weights), np.asarray(expected_weights), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas fused router requires a TPU",
    )
    def test_ties_return_distinct_valid_expert_ids(self):
        x = jnp.zeros((8, 256), dtype=jnp.float32)
        weight = jnp.zeros((8, 256), dtype=jnp.float32)
        bias = jnp.zeros((8,), dtype=jnp.float32)

        _, indices = fused_moe_router(x, weight, bias, topk=2)

        got = np.asarray(indices)
        assert np.all((0 <= got) & (got < 8))
        assert np.all(got[:, 0] != got[:, 1])

    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas fused router requires a TPU",
    )
    def test_t5_router_shape_matches_reference(self):
        x = jax.random.normal(jax.random.key(3), (8, 8192), dtype=jnp.float32)
        weight = (
            jax.random.normal(jax.random.key(4), (256, 8192), dtype=jnp.float32) * 0.01
        )
        bias = jax.random.normal(jax.random.key(5), (256,), dtype=jnp.float32) * 0.01
        kwargs = dict(
            topk=8,
            router_output_multiplier=0.125,
            route_scale=1.0,
            division_eps=1e-15,
            score_normalization="sigmoid",
        )

        expected_weights, expected_indices = ref_fused_moe_router(
            x, weight, bias, **kwargs
        )
        weights, indices = fused_moe_router(x, weight, bias, **kwargs)

        np.testing.assert_array_equal(np.asarray(indices), np.asarray(expected_indices))
        np.testing.assert_allclose(
            np.asarray(weights), np.asarray(expected_weights), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu" or len(jax.devices()) < 2,
        reason="sharded fused router requires multiple TPU devices",
    )
    def test_wrapper_preserves_token_sharding(self):
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        devices = np.asarray(jax.devices())
        mesh = Mesh(devices, ("model",))
        tokens = devices.size * 8
        x = jax.device_put(
            jax.random.normal(jax.random.key(6), (tokens, 256), dtype=jnp.float32),
            NamedSharding(mesh, P("model", None)),
        )
        weight = jax.device_put(
            jax.random.normal(jax.random.key(7), (128, 256), dtype=jnp.float32),
            NamedSharding(mesh, P()),
        )
        bias = jax.device_put(
            jax.random.normal(jax.random.key(8), (128,), dtype=jnp.float32),
            NamedSharding(mesh, P()),
        )
        kwargs = dict(topk=4, score_normalization="sigmoid")

        route = build_fused_moe_router_fn(mesh, P("model", None))
        weights, indices = route(x, weight, bias, **kwargs)
        expected_weights, expected_indices = ref_fused_moe_router(
            x, weight, bias, **kwargs
        )

        assert weights.sharding.spec == P("model", None)
        assert indices.sharding.spec == P("model", None)
        indices_global = multihost_utils.process_allgather(indices, tiled=True)
        expected_indices_global = multihost_utils.process_allgather(
            expected_indices, tiled=True
        )
        weights_global = multihost_utils.process_allgather(weights, tiled=True)
        expected_weights_global = multihost_utils.process_allgather(
            expected_weights, tiled=True
        )
        np.testing.assert_array_equal(indices_global, expected_indices_global)
        np.testing.assert_allclose(
            weights_global, expected_weights_global, atol=1e-5, rtol=1e-5
        )
