# SPDX-License-Identifier: Apache-2.0
"""Gather selected logical KV pages with a Pallas copy kernel."""

from functools import cache

import torch


def _kernel(block_ids_ref, kv_ref, pages_ref):
    del block_ids_ref
    pages_ref[...] = kv_ref[...]


def read_pages(block_ids, kv, *, interpret=False):
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    block = (None, *kv.shape[1:])
    zeros = (0,) * (kv.ndim - 1)
    grid = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(block_ids.shape[0],),
        in_specs=[pl.BlockSpec(block, lambda i, ids: (ids[i], *zeros))],
        out_specs=pl.BlockSpec(block, lambda i, ids: (i, *zeros)),
    )
    return pl.pallas_call(
        _kernel,
        grid_spec=grid,
        out_shape=jax.ShapeDtypeStruct((block_ids.shape[0], *kv.shape[1:]), kv.dtype),
        interpret=interpret,
    )(block_ids, kv)


@cache
def _read_pages_op():
    import jax
    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def gather(block_ids: jax.Array, kv: jax.Array) -> jax.Array:
        return read_pages(block_ids, kv)

    return cpu_export_jax_op("sglang_tpu::kv_read_pages", gather)


def read_pages_torch(block_ids, kv):
    if block_ids.numel() == 0:
        return torch.empty((0, *kv.shape[1:]), dtype=kv.dtype, device=kv.device)
    if kv.device.type == "cpu":
        return kv.index_select(0, block_ids.long())
    return _read_pages_op()(block_ids.to(torch.int32), kv)
