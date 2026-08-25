# SPDX-License-Identifier: Apache-2.0
"""Tests for the TP16 reduce-scatter kernel."""

import importlib
from types import SimpleNamespace

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp import reduce_scatter
from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.common import (
    make_hypercube_tp_mesh,
    make_ring_tp_mesh,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=900, suite="stage-a-test-tpu-unit")


def _make_input(mesh, *, tokens: int, hidden: int) -> jax.Array:
    global_shape = (tokens * mesh.size, hidden)
    sharding = NamedSharding(mesh, P(mesh.axis_names[0], None))
    with jax.set_mesh(mesh):
        values = jax.random.normal(
            jax.random.key(0),
            global_shape,
            dtype=jnp.bfloat16,
            out_sharding=sharding,
        )
    return values * (mesh.size**-0.5)


def _candidate_functions(mesh):
    axis_name = mesh.axis_names[0]
    partition = P(axis_name, None)
    pallas_fn = jax.jit(
        jax.shard_map(
            lambda local_x: reduce_scatter(local_x, mesh, collective_id=0),
            mesh=mesh,
            in_specs=partition,
            out_specs=partition,
            check_vma=False,
        )
    )
    xla_fn = jax.jit(
        jax.shard_map(
            lambda local_x: lax.psum_scatter(
                local_x,
                axis_name,
                scatter_dimension=0,
                tiled=True,
            ),
            mesh=mesh,
            in_specs=partition,
            out_specs=partition,
            check_vma=False,
        )
    )
    return pallas_fn, xla_fn


def _assert_addressable_shards_close(actual: jax.Array, expected: jax.Array) -> None:
    def index_key(shard):
        return tuple(
            (part.start or 0, part.stop or 0, part.step or 1) for part in shard.index
        )

    actual_shards = sorted(actual.addressable_shards, key=index_key)
    expected_shards = sorted(expected.addressable_shards, key=index_key)
    assert [shard.index for shard in actual_shards] == [
        shard.index for shard in expected_shards
    ]
    for actual_shard, expected_shard in zip(actual_shards, expected_shards):
        np.testing.assert_allclose(
            np.asarray(actual_shard.data),
            np.asarray(expected_shard.data),
            rtol=0.025,
            atol=0.0625,
        )


class TestReduceScatter:
    @pytest.fixture(params=(("ring", "x"), ("hypercube", "tp")))
    def mesh(self, request):
        if jax.device_count() != 16:
            pytest.skip("the kernel requires a TP16 2x2x2 topology")
        order, axis_name = request.param
        if order == "ring":
            return make_ring_tp_mesh(axis_name)
        return make_hypercube_tp_mesh(axis_name)

    @pytest.mark.parametrize(
        ("tokens", "hidden"),
        (
            pytest.param(16, 512, id="single-token-padding"),
            pytest.param(16, 8192, id="cube-single-token-padding"),
            pytest.param(32, 8192, id="decode-wide-hidden"),
            pytest.param(32, 16384, id="multi-hidden-block"),
            pytest.param(48, 8192, id="cube-odd-output-token-padding"),
            pytest.param(64, 8192, id="decode-four-tokens"),
            pytest.param(128, 1024, id="small"),
            pytest.param(1024, 4096, id="medium"),
            pytest.param(2048, 8192, id="cube-boundary"),
            pytest.param(3072, 8192, id="cube-non-power-of-two"),
            pytest.param(4096, 8192, id="large-single-chunk"),
            pytest.param(8192, 8192, id="prefill"),
            pytest.param(16384, 8192, id="multi-chunk"),
            pytest.param(
                16384,
                16384,
                id="multi-token-and-hidden-blocks",
            ),
        ),
    )
    def test_matches_xla(self, mesh, tokens, hidden):
        input_arr = _make_input(mesh, tokens=tokens, hidden=hidden)
        pallas_fn, xla_fn = _candidate_functions(mesh)

        actual = pallas_fn(input_arr)
        expected = xla_fn(input_arr)

        assert actual.shape == expected.shape
        _assert_addressable_shards_close(actual, expected)

    @pytest.mark.parametrize(
        ("tokens", "repetitions"),
        (
            pytest.param(4096, 8, id="single-block"),
            pytest.param(8192, 4, id="two-blocks"),
            pytest.param(16384, 4, id="four-blocks"),
        ),
    )
    def test_large_cube_is_repeatable(self, mesh, tokens, repetitions):
        input_arr = _make_input(mesh, tokens=tokens, hidden=8192)
        pallas_fn, xla_fn = _candidate_functions(mesh)
        expected = xla_fn(input_arr)

        for _ in range(repetitions):
            _assert_addressable_shards_close(pallas_fn(input_arr), expected)


class TestReduceScatterInvalidInputs:
    @pytest.fixture
    def valid_mesh(self):
        devices = [
            SimpleNamespace(coords=(x, y, z), core_on_chip=core)
            for x in range(2)
            for y in range(2)
            for z in range(2)
            for core in range(2)
        ]
        return SimpleNamespace(
            size=16,
            devices=np.asarray(devices, dtype=object),
            axis_names=("tp",),
        )

    def test_rejects_non_matrix_input(self):
        with pytest.raises(ValueError, match="rank-local"):
            reduce_scatter(
                np.zeros((1, 2, 3)), SimpleNamespace(size=16), collective_id=0
            )

    def test_rejects_unsupported_mesh_size(self):
        mesh = SimpleNamespace(size=8, devices=np.empty((8,), dtype=object))
        with pytest.raises(ValueError, match="not supported with mesh size of 8"):
            reduce_scatter(np.zeros((16, 512)), mesh, collective_id=0)

    def test_rejects_multidimensional_mesh(self):
        mesh = SimpleNamespace(size=16, devices=np.empty((4, 4), dtype=object))
        with pytest.raises(ValueError, match="requires one non-singleton mesh axis"):
            reduce_scatter(np.zeros((16, 512)), mesh, collective_id=0)

    def test_rejects_non_2x2x2_topology(self):
        devices = [
            SimpleNamespace(coords=(x, y, 0), core_on_chip=core)
            for x in range(4)
            for y in range(2)
            for core in range(2)
        ]
        mesh = SimpleNamespace(
            size=16,
            devices=np.asarray(devices, dtype=object),
            axis_names=("tp",),
        )

        with pytest.raises(ValueError, match="requires a 2x2x2 TPU topology"):
            reduce_scatter(np.zeros((16, 512)), mesh, collective_id=0)

    def test_rejects_non_bf16_input(self, valid_mesh):
        input_arr = jnp.zeros((16, 512), dtype=jnp.float32)

        with pytest.raises(ValueError, match="requires BF16 input"):
            reduce_scatter(input_arr, valid_mesh, collective_id=0)

    def test_rejects_unaligned_hidden_size(self, valid_mesh):
        input_arr = jnp.zeros((16, 513), dtype=jnp.bfloat16)

        with pytest.raises(ValueError, match="hidden divisible by 512"):
            reduce_scatter(input_arr, valid_mesh, collective_id=0)

    def test_rejects_shape_exceeding_vmem(self, valid_mesh, monkeypatch):
        module = importlib.import_module(
            "sglang.srt.hardware_backend.tpu.kernels.tier5_sp.reduce_scatter"
        )
        monkeypatch.setattr(module, "_get_usable_vmem_bytes", lambda: 1)
        input_arr = jax.ShapeDtypeStruct((16, 2097152), jnp.bfloat16)

        with pytest.raises(ValueError, match="block and scratch exceed VMEM"):
            reduce_scatter(input_arr, valid_mesh, collective_id=0)
