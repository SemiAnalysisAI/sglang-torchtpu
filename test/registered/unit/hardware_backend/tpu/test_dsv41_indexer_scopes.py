"""Keep Indexer scheduling groups distinct across exported model layers."""

import re

import jax
import jax.numpy as jnp
import pytest
from jax import export

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.dsv41.custom_ops import select
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("tokens", [256, 1024])
def test_layer_exports_use_disjoint_scheduling_groups(monkeypatch, tokens):
    monkeypatch.setattr(
        "sglang_tpu.pallas_export.get_pallas_tpu_device_kind", lambda: "TPU7x"
    )

    def shape(dims, dtype=jnp.int32):
        return jax.ShapeDtypeStruct(dims, dtype)

    args = (
        shape((tokens, 32, 128), jnp.bfloat16),
        shape((tokens, 32), jnp.float32),
        shape((8448, 128)),
        shape((1, 37)),
        shape((tokens,)),
        shape((tokens,)),
        shape((tokens, 2048)),
        256,
        512,
        8,
        False,
        False,
    )
    exported = export.export(
        jax.jit(select, static_argnums=(7, 8, 9, 10, 11, 12)), platforms=("tpu",)
    )
    groups = []
    with tpu_info_for_cpu_export():
        for layer_id in (0, 1, 0):
            mlir = exported(*args, layer_id).mlir_module()
            groups.append(set(re.findall(r'_scheduling_group_id = "(\d+)"', mlir)))
    assert len(groups[0]) == len(groups[1]) == tokens // 128 - 1
    assert groups[0].isdisjoint(groups[1])
    assert groups[0] == groups[2]
