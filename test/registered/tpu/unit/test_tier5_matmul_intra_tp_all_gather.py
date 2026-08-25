# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=600, suite="stage-a-test-tpu-unit")
from jax.sharding import NamedSharding, PartitionSpec as P

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_intra_tp_all_gather import (
    matmul_intra_tp_all_gather,
    matmul_intra_tp_all_gather_post_moe_epilogue,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    hypercube_device_order,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_intra_tp_all_gather._routing import (
    _topo_bn,
)


class TestTopoBn:
    def test_t5_fp8_full_gather_pooler_uses_automatic_bn512(self):
        assert _topo_bn(2048, 8192, 8192) == 512


class TestMatmulIntraTpAllGather:
    @pytest.fixture
    def tp16_mesh(self):
        devices = jax.devices()
        if len(devices) != 16 or "TPU7" not in devices[0].device_kind:
            pytest.skip("requires one TPU7x 2x2x2 TP16 slice")
        ordered = np.asarray(hypercube_device_order(devices), dtype=object).reshape(
            8, 2
        )
        return jax.sharding.Mesh(ordered, ("chip", "core"))

    def test_m4096_bn2048_compiles_with_v7x_vmem_limit(self, tp16_mesh):
        """Reproduce the T5 unpooler shape that requested 128 MiB on TPU7x."""
        m, k, n = 4096, 4096, 8192
        x_sharding = NamedSharding(tp16_mesh, P(("chip", "core"), None))
        w_sharding = NamedSharding(tp16_mesh, P("core", None))

        @functools.partial(jax.jit, out_shardings=(x_sharding, w_sharding))
        def make_inputs():
            return (
                jnp.ones((m, k), dtype=jnp.bfloat16),
                jnp.ones((n, k), dtype=jnp.bfloat16),
            )

        x, w = make_inputs()
        output = matmul_intra_tp_all_gather(
            x,
            w,
            tp16_mesh,
            "chip",
            "core",
            w_transposed=True,
            gather_kind="topo",
            bn=2048,
        )
        output.block_until_ready()

        assert output.shape == (m, n)
        assert float(output.addressable_data(0)[0, 0]) == k

    @pytest.mark.parametrize("residual_dtype", [jnp.bfloat16, jnp.float32])
    def test_post_moe_epilogue_matches_separate_ops(self, tp16_mesh, residual_dtype):
        m, k, n = 256, 256, 512
        token_sharding = NamedSharding(tp16_mesh, P(("chip", "core"), None))
        weight_sharding = NamedSharding(tp16_mesh, P("core", None))
        replicated = NamedSharding(tp16_mesh, P())

        @functools.partial(
            jax.jit,
            out_shardings=(
                token_sharding,
                weight_sharding,
                token_sharding,
                replicated,
                replicated,
                replicated,
            ),
        )
        def make_inputs():
            keys = jax.random.split(jax.random.key(7), 6)
            return (
                jax.random.normal(keys[0], (m, k), dtype=jnp.float32) * 0.05,
                jax.random.normal(keys[1], (n, k), dtype=jnp.bfloat16) * 0.05,
                jax.random.normal(keys[2], (m, n), dtype=residual_dtype) * 0.05,
                jax.random.normal(keys[3], (n,), dtype=jnp.bfloat16) * 0.01,
                jax.random.normal(keys[4], (n,), dtype=jnp.bfloat16) * 0.01,
                jax.random.normal(keys[5], (n,), dtype=jnp.bfloat16) * 0.01,
            )

        x, w, residual, norm_weight, gate, bias = make_inputs()
        actual = matmul_intra_tp_all_gather_post_moe_epilogue(
            x,
            w,
            residual,
            norm_weight,
            gate,
            bias,
            tp16_mesh,
            "chip",
            "core",
            norm_eps=1e-8,
            has_norm_weight=True,
            norm_weight_before=True,
            norm_zero_centered_gamma=True,
            norm_gain_center=1.0,
            has_unpooler_bias=True,
            use_residual_gate=True,
            residual_temperature=0.3,
            residual_eps=1e-3,
            bn=256,
        )

        branch = (x.astype(jnp.float32) @ w.astype(jnp.float32).T).astype(jnp.bfloat16)
        branch_f32 = branch.astype(jnp.float32) + bias.astype(jnp.float32)
        branch_f32 = branch_f32 * (norm_weight.astype(jnp.float32) + 1.0)
        variance = jnp.mean(branch_f32 * branch_f32, axis=1, keepdims=True)
        normalized = (branch_f32 * jax.lax.rsqrt(variance + 1e-8)).astype(jnp.bfloat16)
        gate_f32 = gate.astype(jnp.float32) / 0.3
        beta = jax.nn.sigmoid(gate_f32)
        alpha = jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-gate_f32) * (1.0 + beta), 1e-3))
        expected = (
            alpha * residual.astype(jnp.float32) + beta * normalized.astype(jnp.float32)
        ).astype(residual_dtype)
        expected = jax.device_put(expected, replicated)

        np.testing.assert_allclose(
            np.asarray(actual.addressable_data(0), dtype=np.float32),
            np.asarray(expected.addressable_data(0), dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
