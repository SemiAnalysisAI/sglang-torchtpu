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
