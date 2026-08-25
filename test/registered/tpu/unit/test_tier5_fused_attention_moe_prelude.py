# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=900, suite="stage-a-test-tpu-unit")

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_attention_moe_prelude import (
    build_fused_attention_moe_prelude_fn,
    build_fused_attention_moe_prelude_router_fn,
    fused_attention_moe_prelude,
    fused_attention_moe_prelude_router,
    ref_fused_attention_moe_prelude,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_attention_moe_prelude_replicated import (
    fused_attention_moe_prelude_router_pooler_replicated,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    hypercube_device_order,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.fused_moe_ops.fused_moe_router import (
    ref_fused_moe_router,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_intra_tp_all_gather._matmul_tp_intra_topo import (
    fused_attention_moe_prelude_router_pooler_topo,
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


class TestFusedAttentionMoePreludeRouterPooler:
    @pytest.fixture
    def tp16_mesh(self):
        devices = jax.devices()
        if len(devices) != 16 or "TPU7" not in devices[0].device_kind:
            pytest.skip("requires one TPU7x 2x2x2 TP16 slice")
        ordered = np.asarray(hypercube_device_order(devices), dtype=object).reshape(
            8, 2
        )
        return jax.sharding.Mesh(ordered, ("chip", "core"))

    @pytest.mark.parametrize(
        ("tokens", "hidden", "pooled_hidden", "bn", "stream_dtype"),
        (
            (256, 256, 256, 128, jnp.bfloat16),
            (512, 256, 256, 128, jnp.bfloat16),
            (512, 1024, 1024, 512, jnp.bfloat16),
            (1024, 256, 256, 128, jnp.bfloat16),
            (1024, 256, 256, 128, jnp.float32),
            (2048, 256, 256, 128, jnp.bfloat16),
            (4096, 256, 256, 128, jnp.float32),
        ),
        ids=(
            "t256",
            "t512",
            "t512-bn512",
            "t1024",
            "t1024-fp32-stream",
            "t2048",
            "t4096-fp32-stream",
        ),
    )
    def test_matches_separate_prelude_router_pooler_and_norm(
        self, tp16_mesh, tokens, hidden, pooled_hidden, bn, stream_dtype
    ):
        experts, topk = 128, 8
        token_sharding = NamedSharding(tp16_mesh, P(("chip", "core"), None))
        replicated = NamedSharding(tp16_mesh, P())
        pooler_sharding = NamedSharding(tp16_mesh, P("core", None))

        @functools.partial(
            jax.jit,
            out_shardings=(
                token_sharding,
                token_sharding,
                replicated,
                replicated,
                replicated,
                replicated,
                replicated,
                pooler_sharding,
                replicated,
            ),
        )
        def make_inputs():
            attn = jax.random.normal(
                jax.random.key(60), (tokens, hidden), dtype=jnp.bfloat16
            )
            stream = jax.random.normal(
                jax.random.key(61), (tokens, hidden), dtype=stream_dtype
            )
            post_weight = jax.random.uniform(
                jax.random.key(62),
                (hidden,),
                dtype=jnp.float32,
                minval=0.5,
                maxval=1.5,
            )
            gate = jax.random.normal(jax.random.key(63), (hidden,), dtype=jnp.float32)
            pre_weight = jax.random.uniform(
                jax.random.key(64),
                (hidden,),
                dtype=jnp.bfloat16,
                minval=0.5,
                maxval=1.5,
            )
            router_weight = (
                jax.random.normal(
                    jax.random.key(65),
                    (experts, hidden),
                    dtype=jnp.float32,
                )
                * 0.02
            )
            expert_bias = jax.random.normal(
                jax.random.key(66), (experts,), dtype=jnp.float32
            )
            pooler_weight = (
                jax.random.normal(
                    jax.random.key(67),
                    (pooled_hidden, hidden),
                    dtype=jnp.bfloat16,
                )
                * jnp.asarray(0.02, jnp.bfloat16)
            ).astype(jnp.bfloat16)
            pre_expert_weight = jax.random.uniform(
                jax.random.key(68),
                (pooled_hidden,),
                dtype=jnp.bfloat16,
                minval=0.5,
                maxval=1.5,
            )
            return (
                attn,
                stream,
                post_weight,
                gate,
                pre_weight,
                router_weight,
                expert_bias,
                pooler_weight,
                pre_expert_weight,
            )

        inputs = make_inputs()
        kwargs = dict(
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
            topk=topk,
            route_scale=2.5,
            division_eps=1e-8,
        )
        (
            actual_h,
            actual_pooled,
            actual_prepared_expert,
            actual_weights,
            actual_indices,
        ) = fused_attention_moe_prelude_router_pooler_topo(
            *inputs,
            tp16_mesh,
            "chip",
            "core",
            bn=bn,
            **kwargs,
        )

        (
            expected_h,
            expected_prepared,
            expected_weights,
            expected_indices,
        ) = build_fused_attention_moe_prelude_router_fn(
            tp16_mesh, token_spec=P(("chip", "core"), None)
        )(
            *inputs[:7],
            **kwargs,
        )
        expected_pooled = jnp.dot(
            expected_prepared,
            inputs[7].T,
            preferred_element_type=jnp.float32,
        ).astype(inputs[0].dtype)
        pooled_f32 = actual_pooled.astype(jnp.float32)
        expected_prepared_expert = (
            pooled_f32
            * jax.lax.rsqrt(
                jnp.mean(pooled_f32 * pooled_f32, axis=-1, keepdims=True) + 1e-5
            )
        ).astype(inputs[0].dtype)
        expected_prepared_expert = (expected_prepared_expert * inputs[8]).astype(
            inputs[0].dtype
        )
        expected_prepared_expert = expected_prepared_expert.reshape(
            tokens, pooled_hidden // 128, 128
        )

        def gather(value):
            return np.asarray(multihost_utils.process_allgather(value, tiled=True))

        assert actual_h.dtype == stream_dtype
        assert actual_pooled.dtype == inputs[0].dtype
        assert actual_prepared_expert.dtype == inputs[0].dtype
        np.testing.assert_array_equal(gather(actual_h), gather(expected_h))
        np.testing.assert_allclose(
            gather(actual_pooled).astype(np.float32),
            gather(expected_pooled).astype(np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            gather(actual_weights), gather(expected_weights), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            gather(actual_prepared_expert).astype(np.float32),
            gather(expected_prepared_expert).astype(np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_array_equal(
            gather(actual_indices).reshape(-1),
            gather(expected_indices).reshape(-1),
        )

    def test_replicated_path_matches_separate_ops(self, tp16_mesh):
        tokens, hidden, pooled_hidden = 512, 256, 256
        experts, topk = 128, 8
        replicated = NamedSharding(tp16_mesh, P())
        pooler_sharding = NamedSharding(tp16_mesh, P("core", None))

        @functools.partial(
            jax.jit,
            out_shardings=(
                replicated,
                replicated,
                replicated,
                replicated,
                replicated,
                replicated,
                replicated,
                pooler_sharding,
                replicated,
            ),
        )
        def make_inputs():
            attn = jax.random.normal(
                jax.random.key(70), (tokens, hidden), dtype=jnp.bfloat16
            )
            stream = jax.random.normal(
                jax.random.key(71), (tokens, hidden), dtype=jnp.bfloat16
            )
            post_weight = jax.random.uniform(
                jax.random.key(72),
                (hidden,),
                dtype=jnp.float32,
                minval=0.5,
                maxval=1.5,
            )
            gate = jax.random.normal(jax.random.key(73), (hidden,), dtype=jnp.float32)
            pre_weight = jax.random.uniform(
                jax.random.key(74),
                (hidden,),
                dtype=jnp.bfloat16,
                minval=0.5,
                maxval=1.5,
            )
            router_weight = (
                jax.random.normal(
                    jax.random.key(75),
                    (experts, hidden),
                    dtype=jnp.float32,
                )
                * 0.02
            )
            expert_bias = jax.random.normal(
                jax.random.key(76), (experts,), dtype=jnp.float32
            )
            pooler_weight = (
                jax.random.normal(
                    jax.random.key(77),
                    (pooled_hidden, hidden),
                    dtype=jnp.bfloat16,
                )
                * jnp.asarray(0.02, jnp.bfloat16)
            ).astype(jnp.bfloat16)
            pre_expert_weight = jax.random.uniform(
                jax.random.key(78),
                (pooled_hidden,),
                dtype=jnp.bfloat16,
                minval=0.5,
                maxval=1.5,
            )
            return (
                attn,
                stream,
                post_weight,
                gate,
                pre_weight,
                router_weight,
                expert_bias,
                pooler_weight,
                pre_expert_weight,
            )

        inputs = make_inputs()
        kwargs = dict(
            has_post_weight=True,
            post_eps=1e-8,
            pre_eps=1e-5,
            topk=topk,
            route_scale=2.5,
            division_eps=1e-8,
        )
        actual = fused_attention_moe_prelude_router_pooler_replicated(
            *inputs,
            tp16_mesh,
            "chip",
            "core",
            bn=128,
            **kwargs,
        )
        (
            expected_h,
            expected_prepared,
            expected_weights,
            expected_indices,
        ) = build_fused_attention_moe_prelude_router_fn(
            tp16_mesh, token_spec=P(None, None)
        )(
            *inputs[:7],
            **kwargs,
        )
        expected_pooled = jnp.dot(
            expected_prepared,
            inputs[7].T,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        pooled_f32 = expected_pooled.astype(jnp.float32)
        expected_prepared_expert = (
            pooled_f32
            * jax.lax.rsqrt(
                jnp.mean(pooled_f32 * pooled_f32, axis=-1, keepdims=True) + 1e-5
            )
            * inputs[8].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        expected_prepared_expert = expected_prepared_expert.reshape(
            tokens, pooled_hidden // 128, 128
        )

        def gather(value):
            return np.asarray(multihost_utils.process_allgather(value, tiled=True))

        actual_h, actual_pooled, actual_expert, actual_weights, actual_indices = actual
        np.testing.assert_array_equal(gather(actual_h), gather(expected_h))
        np.testing.assert_allclose(
            gather(actual_pooled).astype(np.float32),
            gather(expected_pooled).astype(np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            gather(actual_expert).astype(np.float32),
            gather(expected_prepared_expert).astype(np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            gather(actual_weights), gather(expected_weights), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            gather(actual_indices).reshape(-1),
            gather(expected_indices).reshape(-1),
        )
