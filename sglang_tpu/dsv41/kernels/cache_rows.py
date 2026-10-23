# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/models/vllm/deepseek_v41/v2/cache.py
# Revision: bdb38700dd25959c029c7d77f00cee91ccd3a0bc
# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import jax

from sglang_tpu.dsv41.kernels.target import is_tpu_kernel_target

ROW_WORDS = 128


def _permute(x: jax.Array, source: jax.Array) -> jax.Array:
    """``x[..., source]`` as an exact 0/1 matmul (finite ``x``; subnormals flush).

    Lane-stride slices and lane gathers are slow on TPU; the MXU is not.
    """
    import jax.numpy as jnp

    permutation = (jnp.arange(x.shape[-1])[:, None] == source[None]).astype(x.dtype)
    return jnp.einsum(
        "...d,de->...e",
        x,
        permutation,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def piece_order(elements: int) -> jax.Array:
    """Element order of `gather_cache_pieces`: per word row, even then odd."""
    import jax.numpy as jnp

    width = min(elements // 2, ROW_WORDS)
    return jnp.concatenate(
        [
            start + 2 * jnp.arange(width) + parity
            for start in range(0, elements, 2 * width)
            for parity in (0, 1)
        ]
    )


def split_halves(x: jax.Array) -> jax.Array:
    """Reorder the last axis as `piece_order` (``split`` reads)."""
    return _permute(x, piece_order(x.shape[-1]))


def merge_sub_rows(
    cache: jax.Array, records: jax.Array, values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Turn unique sub-row records into full word-row writes.

    Each valid record yields its row: the current row with every segment
    written in this batch replaced. Records sharing a row yield identical
    rows, so writing them in any order is equivalent to the lane scatter.
    Returns ``(row records, int32 [n, ROW_WORDS])``; invalid records stay -1.
    """
    import jax.numpy as jnp

    width = values.shape[1]
    per_row = ROW_WORDS // width
    valid = (records >= 0) & (records < cache.shape[0] * per_row)
    rows = jnp.where(valid, records // per_row, -1)
    segments = records % per_row
    merged = cache[jnp.maximum(rows, 0)]
    # [n, n]: which batch records land in each record's row. Batches are a
    # step's tokens (<= max_num_batched_tokens), so this stays small.
    same_row = (rows[:, None] == rows[None, :]) & valid[:, None] & valid[None, :]
    pieces = []
    for segment in range(per_row):
        writers = same_row & (segments[None, :] == segment)
        source = jnp.take(values, jnp.argmax(writers, axis=1), axis=0)
        pieces.append(
            jnp.where(
                writers.any(axis=1)[:, None],
                source,
                merged[:, segment * width : (segment + 1) * width],
            )
        )
    return rows, jnp.concatenate(pieces, axis=1)


def _pack_bf16_words(values: jax.Array) -> jax.Array:
    """Pack BF16 ``[n, 2 * width]`` into int32 ``[n, width]`` via `split_halves`.

    A direct ``bitcast_convert_type(values.reshape(n, -1, 2), int32)`` lowers on
    TPU to stride-2 lane slices ``[0:511:2]`` and ``[1:512:2]`` (~6.0 ms/chunk
    across 21 SWA/compressor writes). Reordering via `split_halves` (MXU 0/1
    matmul) places even and odd elements into contiguous 128-lane halves first.
    """
    import jax.numpy as jnp

    half_width = min(values.shape[1] // 2, ROW_WORDS)
    split = split_halves(values).reshape(values.shape[0], -1, 2, half_width)
    even = jax.lax.bitcast_convert_type(
        split[:, :, 0, :].astype(jnp.float32), jnp.int32
    )
    odd = jax.lax.bitcast_convert_type(split[:, :, 1, :].astype(jnp.float32), jnp.int32)
    return (jax.lax.shift_right_logical(even, 16) | (odd & jnp.int32(-65536))).reshape(
        values.shape[0], values.shape[1] // 2
    )


def scatter_cache_values(
    cache: jax.Array,
    records: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Write unique physical records, preserving the shared owner and padding.

    Full word rows use DSV4-style aliased DMA. Sub-row writes use independent
    (row, lane) coordinates, so two records sharing a row cannot overwrite
    each other through concurrent read/modify/write of the entire word row.
    """
    import jax.numpy as jnp

    assert values.dtype in (jnp.bfloat16, jnp.float32, jnp.int32)
    if isinstance(cache, tuple):
        cache_lo, cache_hi = cache
        record_words = (
            values.shape[1] // 2 if values.dtype == jnp.bfloat16 else values.shape[1]
        )
        bank_records = cache_lo.shape[0] * ROW_WORDS // record_words
        records_lo = jnp.where(records < bank_records, records, -1)
        records_hi = jnp.where(records >= bank_records, records - bank_records, -1)
        return cast(
            Any,
            (
                scatter_cache_values(cache_lo, records_lo, values),
                scatter_cache_values(cache_hi, records_hi, values),
            ),
        )
    assert cache.dtype == jnp.int32 and cache.ndim == 2 and cache.shape[1] == ROW_WORDS
    if not values.shape[0]:
        return cache
    words = (
        values
        if values.dtype == jnp.int32
        else (
            _pack_bf16_words(values)
            if (
                values.dtype == jnp.bfloat16
                and is_tpu_kernel_target()
                and values.shape[1] >= 2 * ROW_WORDS
                and values.shape[1] % (2 * ROW_WORDS) == 0
            )
            else jax.lax.bitcast_convert_type(
                values.reshape(values.shape[0], -1, 2)
                if values.dtype == jnp.bfloat16
                else values,
                jnp.int32,
            )
        )
    )
    width = words.shape[1]
    if width < ROW_WORDS and is_tpu_kernel_target():
        # An elementwise (row, lane) scatter is slow on TPU. Rebuild whole word
        # rows instead and reuse the full-row DMA below.
        records, words = merge_sub_rows(cache, records, words)
        width = ROW_WORDS
    if width < ROW_WORDS or not is_tpu_kernel_target():
        columns = jnp.arange(width)[None]
        if width < ROW_WORDS:
            per_row = ROW_WORDS // width
            rows = records[:, None] // per_row
            lanes = (records[:, None] % per_row) * width + columns
        else:
            rows = records[:, None] * (width // ROW_WORDS) + columns // ROW_WORDS
            lanes = jnp.broadcast_to(columns % ROW_WORDS, rows.shape)
        rows = jnp.where(records[:, None] >= 0, rows, cache.shape[0])
        return cache.at[rows, lanes].set(words, mode="drop")

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    chunks = width // ROW_WORDS
    # Records per grid step whose row DMAs are in flight together. A serial
    # start/wait per 512 B row is DMA-latency bound.
    block = 64
    count = records.shape[0]
    records = jnp.pad(records, (0, -count % block), constant_values=-1)

    def kernel(record_ref, values_ref, _input_ref, output_ref, sem):
        def copies(index):
            token = pl.program_id(0) * block + index
            record = record_ref[token]
            return (record >= 0) & (record < cache.shape[0] // chunks), [
                pltpu.make_async_copy(
                    values_ref.at[
                        pl.ds((token * chunks + chunk) * ROW_WORDS, ROW_WORDS)
                    ],
                    output_ref.at[record * chunks + chunk, :],
                    sem,
                )
                for chunk in range(chunks)
            ]

        def run(method):
            def body(index, carry):
                valid, row_copies = copies(index)

                @pl.when(valid)
                def _():
                    for copy in row_copies:
                        getattr(copy, method)()

                return carry

            jax.lax.fori_loop(0, block, body, 0)

        run("start")
        run("wait")

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(records.shape[0] // block,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={2: 0},
        # Flatten only the small update batch. A partial DMA from a tiled 2D
        # source can straddle its padded minor dimension; the cache stays 2D.
    )(records, words.reshape(-1), cache)


def scatter_index_keys(
    cache: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Index-only half-plane RMW, retaining the existing aliased pool writer."""
    import jax.numpy as jnp

    from sglang_tpu.kernels.index_selection.key_layout import merge_record_rows

    if isinstance(cache, tuple):
        boundary = cache[0].shape[0] * 2
        return (
            scatter_index_keys(
                cache[0], jnp.where(records < boundary, records, -1), values
            ),
            scatter_index_keys(
                cache[1], jnp.where(records >= boundary, records - boundary, -1), values
            ),
        )
    rows, words = merge_record_rows(cache, records, values)
    return scatter_cache_values(cache, rows, words)
