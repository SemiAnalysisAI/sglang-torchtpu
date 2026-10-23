# SPDX-License-Identifier: Apache-2.0
"""Publish native V4.1 caches through the existing TorchTPU JAX boundary."""

from functools import cache
from typing import NamedTuple

import torch


class CacheOps(NamedTuple):
    compressed: object
    index: object
    swa: object


@cache
def register_cache_ops():
    import jax

    from sglang_tpu.dsv41.kernels.cache_rows import scatter_index_keys
    from sglang_tpu.dsv41.kernels.csa2_cache import publish_csa2_kv
    from sglang_tpu.dsv41.kernels.paged_swa import write_swa_word_pool
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def compressed(
        words: jax.Array,
        scales: jax.Array,
        records: jax.Array,
        values: jax.Array,
        after: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        words, scales, _ = jax.lax.optimization_barrier((words, scales, after))
        return publish_csa2_kv(words, scales, records, values)

    def index(
        words: jax.Array,
        records: jax.Array,
        values: jax.Array,
        after: jax.Array,
    ) -> jax.Array:
        words, _ = jax.lax.optimization_barrier((words, after))
        return scatter_index_keys(words, records, values)

    def swa(
        words: jax.Array,
        records: jax.Array,
        values: jax.Array,
        page_size: int,
    ) -> jax.Array:
        return write_swa_word_pool(words, records, values, page_size=page_size)

    compressed_op = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_publish_compressed",
        compressed,
        donate_argnums=(0, 1),
    )
    compressed_op.register_fake(
        lambda words, scales, *_: (torch.empty_like(words), torch.empty_like(scales))
    )
    index_op = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_publish_index", index, donate_argnums=(0,)
    )
    index_op.register_fake(lambda words, *_: torch.empty_like(words))
    swa_op = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_publish_swa", swa, donate_argnums=(0,)
    )
    swa_op.register_fake(lambda words, *_: torch.empty_like(words))
    return CacheOps(compressed_op, index_op, swa_op)


def publish_source(ops, pool, layer_id, records, latent, key, after):
    """Preserve pool tensor identity and serialize reuse after its last gather."""
    words = pool.low_ratio_cache(layer_id)
    scales = pool.low_ratio_scales(layer_id)
    new_words, new_scales = ops.compressed(words, scales, records, latent, after)
    words.copy_(new_words)
    scales.copy_(new_scales)
    index = pool.low_ratio_cache(layer_id, index=True)
    index.copy_(ops.index(index, records, key, after))
