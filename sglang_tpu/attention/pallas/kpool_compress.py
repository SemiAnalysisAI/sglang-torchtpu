# SPDX-License-Identifier: Apache-2.0
"""K4/D128 pooling directly from raw pages into the native pooled cache.

Short queries batch independent aligned raw tiles across requests. Long queries
stream larger contiguous tiles through two buffers. Dispatch uses a static
per-request query bound, regardless of whether the caller is decoding or prefilling.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _hadamard128(value):
    """Sylvester-order butterfly in f32; rounding is checked at the accuracy gate."""
    lane = jnp.arange(128, dtype=jnp.int32)[None, :]
    for step in (1, 2, 4, 8, 16, 32, 64):
        low = (lane & step) == 0
        partner = jnp.where(
            low,
            pltpu.roll(value, 128 - step, axis=1),
            pltpu.roll(value, step, axis=1),
        )
        value = partner + jnp.where(low, value, -value)
    return value


def _pool_rows(raw, ape):
    """Pool a small aligned raw tile without expanding the live set."""
    pools = raw.shape[0] // 4
    raw = raw.astype(jnp.float32)
    keys = raw[:, :128].reshape((pools, 4, 128))
    scores = raw[:, 128:].reshape((pools, 4, 128)) + ape[None, :, :]
    scores -= jnp.max(scores, axis=1, keepdims=True)
    weight = jnp.exp(scores)
    weight /= jnp.sum(weight, axis=1, keepdims=True)
    return jnp.sum(weight * keys, axis=1).astype(jnp.bfloat16)


def _descriptors(kv_lens, page_indices, cu_q_lens, distribution, num_tokens, tile_rows):
    """One descriptor per aligned raw tile containing a newly closed pool."""
    batch = kv_lens.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    req = jnp.minimum(jnp.searchsorted(cu_q_lens[1:], tok, side="right"), batch - 1)
    start = kv_lens[req] - (cu_q_lens[req + 1] - cu_q_lens[req])
    pos = start + tok - cu_q_lens[req]
    closing = (tok < cu_q_lens[distribution[2]]) & (pos % 4 == 3)
    first = closing & ((pos % tile_rows == 3) | (tok - cu_q_lens[req] < 4))
    # Each request contributes at most two boundary tiles plus q_len // tile_rows.
    capacity = min(num_tokens, num_tokens // tile_rows + 2 * batch)
    chosen = jnp.nonzero(first, size=capacity, fill_value=0)[0]
    r, page = req[chosen], pos[chosen] // 128
    tile_start = pos[chosen] // tile_rows * tile_rows
    raw_offset = tile_start % 128
    pages_per_seq = page_indices.shape[0] // batch
    desc = jnp.stack(
        (
            page_indices[r * pages_per_seq + page],
            page_indices[r * pages_per_seq + page // 4],
            (page % 4) * 8 + raw_offset // 16,
            start[chosen] - tile_start,
            kv_lens[r] - tile_start,
            raw_offset // tile_rows,
        )
    ).astype(jnp.int32)
    return jnp.sum(first, dtype=jnp.int32)[None], desc


def _request_descriptors(
    kv_lens, page_indices, cu_q_lens, distribution, tile_rows, max_q
):
    """Fixed tile slots cover every newly completed pool, including verify Q>1."""
    batch = kv_lens.shape[0]
    # Closed groups end at 3 mod 4. A request's first end can be at the last
    # group in a tile; account for that alignment when bounding tile slots.
    slots = (max_q + tile_rows - 5) // tile_rows + 1
    req = jnp.arange(batch, dtype=jnp.int32)[None, :]
    q_len = cu_q_lens[1:] - cu_q_lens[:-1]
    start = kv_lens - q_len
    first_end = (start // 4) * 4 + 3
    tile_start = (first_end // tile_rows) * tile_rows
    tile_start = tile_start[None, :] + jnp.arange(slots)[:, None] * tile_rows
    first_in_tile = jnp.maximum(first_end[None, :], tile_start + 3)
    live = (req < distribution[2]) & (q_len > 0) & (first_in_tile < kv_lens)
    # Inactive slots still need safe metadata gathers, but issue no DMA.
    safe_start = jnp.where(live, tile_start, 0)
    pages_per_seq = page_indices.shape[0] // batch
    # Tile-major order keeps the first tiles of different requests adjacent.
    # Request-major order would insert unused slots between Q=1 requests.
    return (
        jnp.stack(
            (
                page_indices[req * pages_per_seq + safe_start // 128],
                (safe_start % 128) // tile_rows,
                page_indices[req * pages_per_seq + safe_start // 512],
                (safe_start % 512) // 16,
                start[None, :] - tile_start,
                kv_lens[None, :] - tile_start,
                live.astype(jnp.int32),
            )
        )
        .reshape(7, slots * batch)
        .astype(jnp.int32)
    )


def _pack_pools(pooled):
    """BF16 pooled keys -> F32 Hadamard -> FP8 keys and scale bytes."""
    rotated = _hadamard128(pooled.astype(jnp.float32)) * (128**-0.5)
    amax = jnp.maximum(jnp.max(jnp.abs(rotated), axis=1, keepdims=True), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0))
    q = jnp.clip(rotated / jnp.exp2(exponent), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    scale_bytes = jnp.where(
        jnp.arange(128)[None, :] == 0,
        (exponent + 127).astype(jnp.uint8),
        jnp.uint8(0),
    )
    # Each four adjacent pool rows belong to one physical destination quad.
    return jnp.concatenate(
        (pltpu.bitcast(q, jnp.uint32), pltpu.bitcast(scale_bytes, jnp.uint32)),
        axis=1,
    )


def _short_kernel(
    desc_ref,
    raw_ref,
    old_ref,
    ape_ref,
    mask_ref,
    out_ref,
    raw_buf,
    out_buf,
    read_sem,
    old_sem,
    write_sem,
    *,
    group_size,
):
    """Bounded groups of independent tiles share one vector computation.

    All stores drain before the next group reads the aliased pooled cache.
    Within a group, shared destination quads use ordered read/modify/write.
    """
    del old_ref
    words = out_ref.bitcast(jnp.uint32)

    def destination(i):
        return words.at[desc_ref[2, i], pl.ds(desc_ref[3, i], 1), 0, :]

    def raw_read(i, slot):
        return pltpu.make_async_copy(
            raw_ref.at[desc_ref[0, i], pl.ds(desc_ref[1, i] * 16, 16), :],
            raw_buf.at[slot],
            read_sem.at[slot],
        )

    def old_read(i, slot):
        return pltpu.make_async_copy(destination(i), out_buf.at[slot], old_sem.at[slot])

    def write(i, slot):
        return pltpu.make_async_copy(
            out_buf.at[slot], destination(i), write_sem.at[slot]
        )

    @pl.loop(0, desc_ref.shape[1] // group_size)
    def _group(group):
        begin = group * group_size

        @pl.when(desc_ref[7, begin] != 0)
        def _shared_quad():
            @pl.loop(0, group_size)
            def _ordered(slot):
                i = begin + slot

                @pl.when(desc_ref[6, i] != 0)
                def _live():
                    raw_read(i, slot).start()
                    old_read(i, slot).start()
                    raw_read(i, slot).wait()
                    packed = _pack_pools(_pool_rows(raw_buf[slot], ape_ref[...]))
                    mask = mask_ref[pl.ds(i, 1), :]
                    old_read(i, slot).wait()
                    out_buf[slot] = (out_buf[slot] & ~mask) | (packed & mask)
                    write(i, slot).start()
                    write(i, slot).wait()

        @pl.when(desc_ref[7, begin] == 0)
        def _independent():
            @pl.loop(0, group_size)
            def _start(slot):
                i = begin + slot

                @pl.when(desc_ref[6, i] != 0)
                def _live():
                    raw_read(i, slot).start()
                    old_read(i, slot).start()

            @pl.loop(0, group_size)
            def _wait_raw(slot):
                i = begin + slot

                @pl.when(desc_ref[6, i] != 0)
                def _live():
                    raw_read(i, slot).wait()

            # All four pools of each aligned tile stay adjacent through packing;
            # no Q=1 extraction or packing across unrelated requests is needed.
            raw = raw_buf[...].reshape((group_size * 16, 256))
            packed = _pack_pools(_pool_rows(raw, ape_ref[...]))
            mask = mask_ref[pl.ds(begin, group_size), :]

            @pl.loop(0, group_size)
            def _wait_old(slot):
                i = begin + slot

                @pl.when(desc_ref[6, i] != 0)
                def _live():
                    old_read(i, slot).wait()

            old = out_buf[...].reshape((group_size, 256))
            out_buf[...] = ((old & ~mask) | (packed & mask)).reshape(
                (group_size, 1, 256)
            )

            @pl.loop(0, group_size)
            def _start_write(slot):
                i = begin + slot

                @pl.when(desc_ref[6, i] != 0)
                def _live():
                    write(i, slot).start()

            @pl.loop(0, group_size)
            def _wait_write(slot):
                i = begin + slot

                @pl.when(desc_ref[6, i] != 0)
                def _live():
                    write(i, slot).wait()


def _long_kernel(
    count_ref,
    desc_ref,
    raw_ref,
    old_ref,
    ape_ref,
    out_ref,
    raw_buf,
    out_buf,
    pooled_buf,
    read_sem,
    write_sem,
    old_sem,
    *,
    tile_rows,
):
    del old_ref  # out_ref aliases this input, including untouched bytes.
    words = out_ref.bitcast(jnp.uint32)
    count = count_ref[0]
    word_rows = tile_rows // 16

    def read(i):
        return pltpu.make_async_copy(
            # Keep alignment visible after scalar-prefetched descriptors cross
            # the call boundary: row 5 holds a tile index, not a byte/row offset.
            raw_ref.at[desc_ref[0, i], pl.ds(desc_ref[5, i] * tile_rows, tile_rows), :],
            raw_buf.at[i % 2],
            read_sem.at[i % 2],
        )

    def write(i):
        return pltpu.make_async_copy(
            out_buf.at[i % 2],
            words.at[desc_ref[1, i], pl.ds(desc_ref[2, i], word_rows), 0, :],
            write_sem.at[i % 2],
        )

    for initial in range(2):

        @pl.when(count > initial)
        def _prime(initial=initial):
            read(initial).start()

    def _page(i, slot):
        read(i).wait()

        if tile_rows == 32:
            pooled = _pool_rows(raw_buf[slot], ape_ref[...])
        else:
            # Retain the larger DMA while bounding pooling's register live
            # set. The VMEM staging also bounds values carried across loops.
            @pl.loop(0, tile_rows // 32)
            def _pool_chunk(chunk):
                value = _pool_rows(
                    raw_buf[slot, pl.ds(chunk * 32, 32), :], ape_ref[...]
                )
                # BF16 staging tiles require 16-row alignment in this runtime;
                # FP32 permits an 8-row store. Preserve the BF16 rounding above.
                pooled_buf[pl.ds(chunk * 8, 8), :] = value.astype(jnp.float32)

            pooled = pooled_buf[...]

        # Every raw Ref load precedes this refill. Subsequent stages only use
        # SSA values; no consumer rereads the overwritten VMEM generation.
        @pl.when(i + 2 < count)
        def _refill():
            read(i + 2).start()

        packed = _pack_pools(pooled)
        mask = jnp.zeros((word_rows, 256), jnp.uint32)
        for lane in range(4):
            ends = (
                jax.lax.broadcasted_iota(jnp.int32, (word_rows, 256), 0) * 16
                + lane * 4
                + 3
            )
            live = (ends >= desc_ref[3, i]) & (ends < desc_ref[4, i])
            mask = mask | (
                jnp.where(live, jnp.uint32(255), jnp.uint32(0)) << (lane * 8)
            )

        # Serialize the *read* too. A later tile may update different bytes
        # in exactly the same physical destination block as an earlier tile.
        @pl.when(i > 0)
        def _prior_store():
            write(i - 1).wait()

        @pl.when((desc_ref[3, i] > 3) | (desc_ref[4, i] < tile_rows))
        def _preserve():
            old = pltpu.make_async_copy(
                words.at[desc_ref[1, i], pl.ds(desc_ref[2, i], word_rows), 0, :],
                out_buf.at[slot],
                old_sem,
            )
            old.start()
            old.wait()
            out_buf[slot, :, :] = (out_buf[slot, :, :] & ~mask) | (packed & mask)

        @pl.when((desc_ref[3, i] <= 3) & (desc_ref[4, i] >= tile_rows))
        def _overwrite():
            out_buf[slot, :, :] = packed

        write(i).start()

    # Keep the two-slot period explicit; the number of pages stays dynamic.
    @pl.loop(0, pl.cdiv(count, 2))
    def _pair(pair):
        for slot in range(2):

            @pl.when(pair * 2 + slot < count)
            def _active(slot=slot):
                _page(pair * 2 + slot, slot)

    @pl.when(count > 0)
    def _drain():
        write(count - 1).wait()


def compress_and_store(
    raw_cache,
    pooled_cache,
    ape,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    num_tokens,
    kpool=4,
    *,
    max_query_length=None,
):
    """Compress newly closed K4 groups into the aliased native pooled cache.

    Raw writes must already be complete. Destination records must be unique;
    distinct records in a shared physical block are supported.
    max_query_length is a static upper bound on each request's newly supplied
    query tokens, including every verification row, not historical KV length.
    Omit it to use num_tokens as a conservative bound for ragged requests.
    Short prefill, decode and verification use the same batched implementation.
    """
    if kpool != 4 or raw_cache.shape[1:] != (128, 256):
        raise ValueError("fused pooling requires K=4, page_size=128 and D=128")
    if pooled_cache.shape != (raw_cache.shape[0], 32, 4, 256):
        raise ValueError("unexpected pooled-cache layout")
    if raw_cache.dtype != jnp.bfloat16 or pooled_cache.dtype != jnp.uint8:
        raise ValueError("expected BF16 raw cache and uint8 pooled cache")
    if not num_tokens:
        return pooled_cache
    max_q = num_tokens if max_query_length is None else max_query_length
    if not isinstance(max_q, int) or max_q < 1:
        raise ValueError("max_query_length must be a positive static integer")
    if max_q <= 32:
        desc = _request_descriptors(
            kv_lens, page_indices, cu_q_lens, distribution, 16, max_q
        )
        group_size = min(16, desc.shape[1])
        padding = (-desc.shape[1]) % group_size
        desc = jnp.pad(desc, ((0, 0), (0, padding)))
        count = desc.shape[1]
        groups = desc.reshape((7, count // group_size, group_size))
        same_quad = (groups[2, :, :, None] == groups[2, :, None, :]) & (
            groups[3, :, :, None] == groups[3, :, None, :]
        )
        live_pair = (groups[6, :, :, None] != 0) & (groups[6, :, None, :] != 0)
        index = jnp.arange(group_size)
        conflict = jnp.any(
            same_quad & live_pair & (index[:, None] < index[None, :]), axis=(1, 2)
        )
        desc = jnp.concatenate(
            (desc, jnp.repeat(conflict, group_size)[None, :].astype(jnp.int32)),
            axis=0,
        )
        ends = jnp.arange(4) * 4 + 3
        live = (ends[None, :] >= desc[4, :, None]) & (ends[None, :] < desc[5, :, None])
        mask = jnp.sum(
            jnp.where(live, jnp.uint32(255), jnp.uint32(0))
            << (jnp.arange(4, dtype=jnp.uint32) * 8)[None, :],
            axis=1,
        )[:, None]
        return pl.pallas_call(
            partial(_short_kernel, group_size=group_size),
            out_shape=jax.ShapeDtypeStruct(pooled_cache.shape, pooled_cache.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1,),
                in_specs=(
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(),
                    pl.BlockSpec(),
                ),
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=(
                    pltpu.VMEM((group_size, 16, 256), jnp.bfloat16),
                    pltpu.VMEM((group_size, 1, 256), jnp.uint32),
                    pltpu.SemaphoreType.DMA((group_size,)),
                    pltpu.SemaphoreType.DMA((group_size,)),
                    pltpu.SemaphoreType.DMA((group_size,)),
                ),
            ),
            input_output_aliases={2: 0},
            name="kpool_short_query_store",
        )(desc, raw_cache, pooled_cache, ape.astype(jnp.float32), mask)
    tile_rows = 32 if max_q < 128 else 128
    count, desc = _descriptors(
        kv_lens, page_indices, cu_q_lens, distribution, num_tokens, tile_rows
    )
    return pl.pallas_call(
        partial(_long_kernel, tile_rows=tile_rows),
        out_shape=jax.ShapeDtypeStruct(pooled_cache.shape, pooled_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.VMEM((2, tile_rows, 256), jnp.bfloat16),
                pltpu.VMEM((2, tile_rows // 16, 256), jnp.uint32),
                pltpu.VMEM((tile_rows // 4, 128), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={3: 0},
        name="kpool_long_query_store",
    )(count, desc, raw_cache, pooled_cache, ape.astype(jnp.float32))
