# SPDX-License-Identifier: Apache-2.0

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=600, suite="stage-a-test-tpu-unit")
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_intra_tp_all_gather import (
    api,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.matmul_intra_tp_all_gather._matmul_tp_intra_local import (
    _pick_local_bn,
)
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    hypercube_device_order,
)


class TestLocalBn:
    @pytest.mark.parametrize("half_n", [4096 // 2, 8192 // 2])
    def test_small_local_shards_use_bn512(self, half_n):
        assert _pick_local_bn(half_n, m_local=128) == 512

    @pytest.mark.parametrize("half_n", [4096 // 2, 8192 // 2])
    def test_full_backbone_shards_retain_bn256(self, half_n):
        assert _pick_local_bn(half_n, m_local=256) == 256


class TestMatmulIntraTpOutputGatherDispatch:
    @pytest.fixture
    def operands(self):
        x = types.SimpleNamespace(shape=(4096, 8192), dtype="bf16")
        w = types.SimpleNamespace(shape=(2048, 8192), dtype="bf16")
        mesh = types.SimpleNamespace(shape={"chip": 8, "core": 2})
        return x, w, mesh

    def test_auto_uses_pallas_for_large_tp16_shape(self, monkeypatch, operands):
        x, w, mesh = operands
        captured = {}

        def local_kernel(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return "pallas"

        monkeypatch.setattr(api, "_is_2x2x2_slice", lambda *_: True)
        monkeypatch.setattr(api, "_matmul_intra_tp_local_output", local_kernel)

        result = api.matmul_intra_tp_output_gather(
            x,
            w,
            mesh,
            "chip",
            "core",
            out_dtype="bf16",
            w_transposed=True,
        )

        assert result == "pallas"
        assert captured["args"] == (x, w, mesh, "chip", "core")
        assert captured["kwargs"] == {
            "bn": None,
            "out_dtype": "bf16",
            "w_transposed": True,
        }

    def test_auto_uses_pallas_from_measured_small_shape(self, monkeypatch, operands):
        _, w, mesh = operands
        x = types.SimpleNamespace(shape=(128, 8192), dtype="bf16")

        monkeypatch.setattr(api, "_is_2x2x2_slice", lambda *_: True)
        monkeypatch.setattr(
            api, "_matmul_intra_tp_local_output", lambda *_args, **_kwargs: "pallas"
        )

        result = api.matmul_intra_tp_output_gather(
            x, w, mesh, "chip", "core", w_transposed=True
        )

        assert result == "pallas"

    def test_auto_uses_xla_below_pallas_tile(self, monkeypatch, operands):
        _, w, mesh = operands
        x = types.SimpleNamespace(shape=(64, 8192), dtype="bf16")

        def unexpected_pallas(*_args, **_kwargs):
            pytest.fail("unsupported token tile selected Pallas")

        monkeypatch.setattr(api, "_matmul_intra_tp_local_output", unexpected_pallas)
        monkeypatch.setattr(
            api.jax,
            "shard_map",
            lambda *_args, **_kwargs: lambda *_operands: "xla",
        )
        monkeypatch.setattr(api.jax, "jit", lambda fn: fn)

        result = api.matmul_intra_tp_output_gather(
            x, w, mesh, "chip", "core", w_transposed=True
        )

        assert result == "xla"

    def test_forced_pallas_rejects_activation_scale(self, operands):
        x, w, mesh = operands
        scale = types.SimpleNamespace(shape=(4096,))

        with pytest.raises(ValueError, match="does not support x_scale"):
            api.matmul_intra_tp_output_gather(
                x,
                w,
                mesh,
                "chip",
                "core",
                x_scale=scale,
                gather_kind="pallas",
            )

    def test_rejects_unknown_gather_kind(self, operands):
        x, w, mesh = operands

        with pytest.raises(ValueError, match="gather_kind must be"):
            api.matmul_intra_tp_output_gather(
                x, w, mesh, "chip", "core", gather_kind="unknown"
            )


class TestMatmulIntraTpOutputGatherNumerics:
    @pytest.fixture
    def tp16_mesh(self):
        devices = jax.devices()
        if len(devices) != 16 or "TPU7" not in devices[0].device_kind:
            pytest.skip("requires one TPU7x 2x2x2 TP16 slice")
        ordered = np.asarray(hypercube_device_order(devices), dtype=object).reshape(
            8, 2
        )
        return jax.sharding.Mesh(ordered, ("chip", "core"))

    @pytest.mark.parametrize("m", [128, 4096])
    @pytest.mark.parametrize("input_kind", ["zero", "random", "outlier"])
    def test_pallas_matches_xla_and_dense_reference(self, tp16_mesh, input_kind, m):
        """The SP pooler half-W kernel preserves a dense replicated-weight matmul."""
        k, n = 512, 1024
        x_sharding = NamedSharding(tp16_mesh, P(("chip", "core"), None))
        w_sharding = NamedSharding(tp16_mesh, P("core", None))
        out_sharding = NamedSharding(tp16_mesh, P(("chip", "core"), None))

        @functools.partial(jax.jit, out_shardings=(x_sharding, w_sharding))
        def make_inputs():
            if input_kind == "zero":
                x = jnp.zeros((m, k), dtype=jnp.bfloat16)
            else:
                x = jax.random.normal(jax.random.key(30), (m, k), dtype=jnp.bfloat16)
                if input_kind == "outlier":
                    x = x.at[::257, ::127].set(jnp.asarray(32, jnp.bfloat16))
            w = (
                jax.random.normal(jax.random.key(31), (n, k), dtype=jnp.bfloat16)
                * jnp.asarray(0.02, jnp.bfloat16)
            ).astype(jnp.bfloat16)
            return x, w

        x, w = make_inputs()

        @functools.partial(jax.jit, out_shardings=out_sharding)
        def dense_reference(xx, ww):
            return jnp.dot(xx, ww.T, preferred_element_type=jnp.float32).astype(
                jnp.bfloat16
            )

        expected = dense_reference(x, w)
        xla = api.matmul_intra_tp_output_gather(
            x,
            w,
            tp16_mesh,
            "chip",
            "core",
            out_dtype=jnp.bfloat16,
            w_transposed=True,
            gather_kind="xla",
        )
        pallas = api.matmul_intra_tp_output_gather(
            x,
            w,
            tp16_mesh,
            "chip",
            "core",
            out_dtype=jnp.bfloat16,
            w_transposed=True,
            gather_kind="pallas",
        )

        expected_g, xla_g, pallas_g = (
            np.asarray(multihost_utils.process_allgather(value, tiled=True), np.float32)
            for value in (expected, xla, pallas)
        )
        np.testing.assert_allclose(xla_g, expected_g, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(pallas_g, expected_g, rtol=2e-2, atol=2e-2)
