# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=600, suite="stage-a-test-tpu-unit")

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_attention_moe_prelude import (
    build_fused_attention_moe_prelude_fn,
    build_fused_attention_moe_prelude_router_fn,
    fused_attention_moe_prelude,
    fused_attention_moe_prelude_router,
    ref_fused_attention_moe_prelude,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_router import (
    ref_fused_moe_router,
)


class TestFusedAttentionMoePrelude:
    @pytest.fixture
    def inputs(self):
        attn = jax.random.normal(jax.random.PRNGKey(0), (8, 128), dtype=jnp.bfloat16)
        stream = jax.random.normal(jax.random.PRNGKey(1), (8, 128), dtype=jnp.bfloat16)
        post_weight = 1.0 + 0.01 * jax.random.normal(
            jax.random.PRNGKey(2), (128,), dtype=jnp.float32
        )
        gate = 0.01 * jax.random.normal(
            jax.random.PRNGKey(3), (128,), dtype=jnp.float32
        )
        return attn, stream, post_weight, gate

    def test_reference_shapes(self, inputs):
        h, scale = ref_fused_attention_moe_prelude(
            *inputs,
            post_eps=1e-8,
            pre_eps=1e-5,
        )

        assert h.shape == (8, 128)
        assert h.dtype == jnp.bfloat16
        assert scale.shape == (8,)
        assert scale.dtype == jnp.float32

    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas TPU prelude kernel requires a TPU",
    )
    def test_kernel_matches_reference(self, inputs):
        expected = ref_fused_attention_moe_prelude(
            *inputs,
            post_eps=1e-8,
            pre_eps=1e-5,
        )
        actual = fused_attention_moe_prelude(
            *inputs,
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
        )

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-6, rtol=1e-6)

    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas TPU prelude kernel requires a TPU",
    )
    def test_kernel_emits_prepared_ffn_input(self, inputs):
        pre_weight = jax.random.uniform(
            jax.random.key(4),
            (128,),
            dtype=jnp.bfloat16,
            minval=0.5,
            maxval=1.5,
        )
        expected_h, expected_scale = ref_fused_attention_moe_prelude(
            *inputs,
            post_eps=1e-8,
            pre_eps=1e-5,
        )
        expected_prepared = (
            expected_h.astype(jnp.float32) * expected_scale[:, None]
        ).astype(jnp.bfloat16)
        expected_prepared = (expected_prepared * pre_weight).astype(jnp.bfloat16)

        actual_h, actual_prepared = fused_attention_moe_prelude(
            *inputs,
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
            pre_weight=pre_weight,
            prepare_ffn=True,
        )

        np.testing.assert_array_equal(actual_h, expected_h)
        np.testing.assert_array_equal(actual_prepared, expected_prepared)

    @pytest.mark.parametrize("tokens", [128, 256, 512])
    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas TPU prelude kernel requires a TPU",
    )
    def test_prepared_input_matches_across_tile_boundary(self, tokens):
        hidden = 128
        shape = (tokens, hidden)
        attn = jax.random.normal(jax.random.key(40), shape, dtype=jnp.bfloat16)
        stream = jax.random.normal(jax.random.key(41), shape, dtype=jnp.bfloat16)
        post_weight = jax.random.uniform(
            jax.random.key(42),
            (hidden,),
            dtype=jnp.float32,
            minval=0.5,
            maxval=1.5,
        )
        gate = jax.random.normal(jax.random.key(43), (hidden,), dtype=jnp.float32)
        pre_weight = jax.random.uniform(
            jax.random.key(44),
            (hidden,),
            dtype=jnp.bfloat16,
            minval=0.5,
            maxval=1.5,
        )
        expected_h, expected_scale = ref_fused_attention_moe_prelude(
            attn,
            stream,
            post_weight,
            gate,
            post_eps=1e-8,
            pre_eps=1e-5,
        )
        expected_prepared = (
            expected_h.astype(jnp.float32) * expected_scale[:, None]
        ).astype(jnp.bfloat16)
        expected_prepared = (expected_prepared * pre_weight).astype(jnp.bfloat16)

        actual_h, actual_prepared = fused_attention_moe_prelude(
            attn,
            stream,
            post_weight,
            gate,
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
            pre_weight=pre_weight,
            prepare_ffn=True,
        )

        np.testing.assert_array_equal(actual_h, expected_h)
        np.testing.assert_allclose(
            actual_prepared,
            expected_prepared,
            rtol=0.01,
            atol=0.002,
        )

    @pytest.mark.parametrize("tokens", [16, 32, 64, 128])
    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas TPU prelude-router kernel requires a TPU",
    )
    def test_kernel_fuses_prepared_input_and_router(self, tokens):
        hidden, experts = 256, 128
        attn = jax.random.normal(
            jax.random.key(40), (tokens, hidden), dtype=jnp.bfloat16
        )
        stream = jax.random.normal(
            jax.random.key(41), (tokens, hidden), dtype=jnp.bfloat16
        )
        post_weight = jax.random.uniform(
            jax.random.key(42), (hidden,), minval=0.5, maxval=1.5
        )
        gate = jax.random.normal(jax.random.key(43), (hidden,)) * 0.1
        pre_weight = jax.random.uniform(
            jax.random.key(44),
            (hidden,),
            dtype=jnp.bfloat16,
            minval=0.5,
            maxval=1.5,
        )
        router_weight = jax.random.normal(
            jax.random.key(45), (experts, hidden), dtype=jnp.float32
        )
        expert_bias = jax.random.normal(jax.random.key(46), (experts,)) * 0.01

        expected_h, expected_scale = ref_fused_attention_moe_prelude(
            attn,
            stream,
            post_weight,
            gate,
            post_eps=1e-8,
            pre_eps=1e-5,
        )
        expected_prepared = (
            expected_h.astype(jnp.float32) * expected_scale[:, None]
        ).astype(jnp.bfloat16)
        expected_prepared = (expected_prepared * pre_weight).astype(jnp.bfloat16)
        expected_weights, expected_indices = ref_fused_moe_router(
            expected_prepared,
            router_weight,
            expert_bias,
            topk=8,
            route_scale=2.5,
            division_eps=1e-8,
        )

        actual = fused_attention_moe_prelude_router(
            attn,
            stream,
            post_weight,
            gate,
            pre_weight,
            router_weight,
            expert_bias,
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
            topk=8,
            route_scale=2.5,
            division_eps=1e-8,
        )

        np.testing.assert_array_equal(actual[0], expected_h)
        np.testing.assert_array_equal(actual[1], expected_prepared)
        assert actual[3].dtype == jnp.uint8
        np.testing.assert_array_equal(
            actual[3].reshape(tokens, 8).astype(jnp.int32), expected_indices
        )
        np.testing.assert_allclose(actual[2], expected_weights, atol=1e-6, rtol=1e-6)

    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas TPU prelude kernel requires a TPU",
    )
    def test_prepared_input_matches_at_production_hidden_size(self):
        shape = (64, 8192)
        attn = jax.random.normal(jax.random.key(30), shape, dtype=jnp.bfloat16)
        stream = jax.random.normal(jax.random.key(31), shape, dtype=jnp.bfloat16)
        post_weight = jax.random.uniform(
            jax.random.key(32),
            (shape[1],),
            dtype=jnp.float32,
            minval=0.5,
            maxval=1.5,
        )
        gate = jax.random.normal(jax.random.key(33), (shape[1],), dtype=jnp.float32)
        pre_weight = jax.random.uniform(
            jax.random.key(34),
            (shape[1],),
            dtype=jnp.bfloat16,
            minval=0.5,
            maxval=1.5,
        )
        expected_h, expected_scale = ref_fused_attention_moe_prelude(
            attn,
            stream,
            post_weight,
            gate,
            post_eps=1e-8,
            pre_eps=1e-5,
        )
        expected_prepared = (
            expected_h.astype(jnp.float32) * expected_scale[:, None]
        ).astype(jnp.bfloat16)
        expected_prepared = (expected_prepared * pre_weight).astype(jnp.bfloat16)

        actual_h, actual_prepared = fused_attention_moe_prelude(
            attn,
            stream,
            post_weight,
            gate,
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
            pre_weight=pre_weight,
            prepare_ffn=True,
        )

        np.testing.assert_array_equal(actual_h, expected_h)
        np.testing.assert_array_equal(actual_prepared, expected_prepared)

    @pytest.mark.parametrize(
        ("tokens", "input_kind"),
        [(1, "zero"), (256, "random"), (256, "outlier")],
    )
    @pytest.mark.skipif(
        jax.devices()[0].platform != "tpu",
        reason="Pallas TPU prelude kernel requires a TPU",
    )
    def test_kernel_matches_reference_across_inputs(self, tokens, input_kind):
        shape = (tokens, 128)
        if input_kind == "zero":
            attn = jnp.zeros(shape, dtype=jnp.bfloat16)
            stream = jnp.zeros(shape, dtype=jnp.bfloat16)
        else:
            attn = jax.random.normal(jax.random.key(10), shape, dtype=jnp.bfloat16)
            stream = jax.random.normal(jax.random.key(11), shape, dtype=jnp.bfloat16)
            if input_kind == "outlier":
                attn = attn.at[::31, ::17].set(jnp.asarray(32, jnp.bfloat16))
                stream = stream.at[::29, ::13].set(jnp.asarray(-24, jnp.bfloat16))
        post_weight = jax.random.uniform(
            jax.random.key(12), (128,), jnp.float32, minval=0.5, maxval=1.5
        )
        gate = jax.random.normal(jax.random.key(13), (128,), jnp.float32) * 0.1

        expected = ref_fused_attention_moe_prelude(
            attn,
            stream,
            post_weight,
            gate,
            post_eps=1e-8,
            pre_eps=1e-5,
        )
        actual = fused_attention_moe_prelude(
            attn,
            stream,
            post_weight,
            gate,
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
        )

        np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
        expected_scale_shape = (tokens,) if tokens <= 64 else (tokens, 1)
        assert actual[1].shape == expected_scale_shape
        np.testing.assert_allclose(
            np.asarray(actual[1]).reshape(tokens),
            np.asarray(expected[1]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_rejects_non_rank2_inputs(self, inputs):
        attn, stream, post_weight, gate = inputs

        with pytest.raises(ValueError, match="rank-2"):
            fused_attention_moe_prelude(
                attn[None, ...],
                stream[None, ...],
                post_weight,
                gate,
                has_post_weight=True,
                post_eps=1e-8,
                pre_eps=1e-5,
            )

    @pytest.mark.parametrize(
        ("tokens", "input_kind", "has_post_weight"),
        [
            (16, "zero", False),
            (256, "random", False),
            (256, "random", True),
            (4096, "outlier", True),
        ],
    )
    def test_sharded_wrapper_preserves_unfused_rounding(
        self, tokens, input_kind, has_post_weight
    ):
        devices = jax.devices()
        if len(devices) < 2 or "TPU7" not in devices[0].device_kind:
            pytest.skip("requires multiple TPU7x devices")
        if tokens % len(devices):
            pytest.skip("token count must divide the model mesh")
        mesh = jax.sharding.Mesh(np.asarray(devices), ("model",))
        token_sharding = NamedSharding(mesh, P("model", None))
        replicated = NamedSharding(mesh, P())
        hidden = 128

        @jax.jit
        def make_inputs():
            if input_kind == "zero":
                attn = jnp.zeros((tokens, hidden), dtype=jnp.bfloat16)
                stream = jnp.zeros((tokens, hidden), dtype=jnp.bfloat16)
            else:
                attn = jax.random.normal(
                    jax.random.key(20), (tokens, hidden), dtype=jnp.bfloat16
                )
                stream = jax.random.normal(
                    jax.random.key(21), (tokens, hidden), dtype=jnp.bfloat16
                )
                if input_kind == "outlier":
                    attn = attn.at[::31, ::17].set(jnp.asarray(32, dtype=jnp.bfloat16))
                    stream = stream.at[::29, ::13].set(
                        jnp.asarray(-24, dtype=jnp.bfloat16)
                    )
            weight = jax.random.uniform(
                jax.random.key(22),
                (hidden,),
                dtype=jnp.float32,
                minval=0.5,
                maxval=1.5,
            )
            gate = jax.random.normal(jax.random.key(23), (hidden,), dtype=jnp.float32)
            return attn, stream, weight, gate

        attn, stream, weight, gate = make_inputs()
        attn = jax.device_put(attn, token_sharding)
        stream = jax.device_put(stream, token_sharding)
        weight = jax.device_put(weight, replicated)
        gate = jax.device_put(gate, replicated)
        kwargs = dict(
            has_post_weight=has_post_weight,
            post_eps=1e-8,
            pre_eps=1e-5,
            temperature=0.3,
            residual_eps=1e-3,
        )
        actual_h, actual_scale = build_fused_attention_moe_prelude_fn(
            mesh, token_spec=P("model", None)
        )(attn, stream, weight, gate, **kwargs)
        expected_h, expected_scale = ref_fused_attention_moe_prelude(
            attn,
            stream,
            weight if has_post_weight else None,
            gate,
            post_eps=kwargs["post_eps"],
            pre_eps=kwargs["pre_eps"],
            temperature=kwargs["temperature"],
            residual_eps=kwargs["residual_eps"],
        )
        expected_scale = expected_scale[:, None]

        for actual, expected in (
            (actual_h, expected_h),
            (actual_scale, expected_scale),
        ):
            actual_global = multihost_utils.process_allgather(actual, tiled=True)
            expected_global = multihost_utils.process_allgather(
                jax.device_put(expected, actual.sharding), tiled=True
            )
            np.testing.assert_array_equal(actual_global, expected_global)
