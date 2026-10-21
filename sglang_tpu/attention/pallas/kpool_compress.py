# SPDX-License-Identifier: Apache-2.0
"""K4/D128 pooling directly from raw pages into the native pooled cache.

Decode/verify uses fixed request/tile slots and double raw buffering. A prior
store overlaps independent destination reads and compute; aliased destinations
still serialize their RMW. The general prefill path uses larger raw transfers.
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


def _decode_descriptors(
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


def _decode_kernel(
    desc_ref,
    raw_ref,
    old_ref,
    ape_ref,
    out_ref,
    raw_buf,
    out_buf,
    read_sem,
    old_sem,
    write_sem,
    *,
    tile_rows,
    pipeline,
    single_query,
    q1_unroll,
):
    del old_ref  # Aliased output preserves every inactive record and byte.
    words = out_ref.bitcast(jnp.uint32)
    word_rows = tile_rows // 16
    count = desc_ref.shape[1]
    buffers = 2 if pipeline else 1

    def raw_read(i, slot=None):
        return pltpu.make_async_copy(
            raw_ref.at[desc_ref[0, i], pl.ds(desc_ref[1, i] * tile_rows, tile_rows), :],
            raw_buf.at[i % buffers if slot is None else slot],
            read_sem.at[i % buffers if slot is None else slot],
        )

    def destination(i):
        return words.at[desc_ref[2, i], pl.ds(desc_ref[3, i], word_rows), 0, :]

    def write(i, slot=None):
        return pltpu.make_async_copy(
            out_buf.at[i % buffers if slot is None else slot],
            destination(i),
            write_sem.at[i % buffers if slot is None else slot],
        )

    def refill(i, slot=None):
        # Clip the speculative descriptor index before checking its live bit.
        # This also covers trailing inactive slots without waiting on a DMA
        # that was never started.
        following = jnp.minimum(i + buffers, count - 1)

        @pl.when((i + buffers < count) & (desc_ref[6, following] != 0))
        def _start():
            raw_read(following, slot).start()

    for initial in range(min(buffers, count)):

        @pl.when(desc_ref[6, initial] != 0)
        def _prime(initial=initial):
            raw_read(initial).start()

    def _tile(i, fixed_slot=None):
        slot = i % buffers if fixed_slot is None else fixed_slot
        previous_slot = None if fixed_slot is None else 1 - fixed_slot
        previous = jnp.maximum(i - 1, 0)
        previous_live = (i > 0) & (desc_ref[6, previous] != 0)
        # Tiles have equal aligned extents. Equal page and quad start means
        # their entire RMW regions alias, even if different pool bytes change.
        conflict = (desc_ref[2, previous] == desc_ref[2, i]) & (
            desc_ref[3, previous] == desc_ref[3, i]
        )

        @pl.when(desc_ref[6, i] != 0)
        def _closed():
            if pipeline:

                @pl.when(previous_live & conflict)
                def _conflict_wait():
                    write(previous, previous_slot).wait()

            old = pltpu.make_async_copy(
                destination(i), out_buf.at[slot], old_sem.at[slot]
            )
            old.start()
            raw_read(i, fixed_slot).wait()
            pooled = _pool_rows(raw_buf[slot], ape_ref[...])
            if pipeline:
                # Ref loads finished; later compute uses SSA values only.
                refill(i, fixed_slot)
            rotated = _hadamard128(pooled.astype(jnp.float32))
            rotated = (rotated * (128**-0.5)).astype(jnp.bfloat16).astype(jnp.float32)
            amax = jnp.maximum(jnp.max(jnp.abs(rotated), axis=1, keepdims=True), 1e-4)
            exponent = jnp.ceil(jnp.log2(amax / 448.0))
            q = jnp.clip(rotated / jnp.exp2(exponent), -448.0, 448.0).astype(
                jnp.float8_e4m3fn
            )
            scale_bytes = jnp.where(
                jnp.arange(128)[None, :] == 0,
                (exponent + 127).astype(jnp.uint8),
                jnp.uint8(0),
            )
            packed = jnp.concatenate(
                (pltpu.bitcast(q, jnp.uint32), pltpu.bitcast(scale_bytes, jnp.uint32)),
                axis=1,
            )
            row = jax.lax.broadcasted_iota(jnp.int32, (word_rows, 256), 0)
            if single_query:
                selected = (desc_ref[5, i] - 1) // 4
                mask = jnp.where(
                    row == selected // 4,
                    jnp.uint32(255) << ((selected % 4) * 8),
                    jnp.uint32(0),
                )
            else:
                mask = jnp.zeros((word_rows, 256), jnp.uint32)
                for lane in range(4):
                    ends = row * 16 + lane * 4 + 3
                    live = (ends >= desc_ref[4, i]) & (ends < desc_ref[5, i])
                    mask |= jnp.where(live, jnp.uint32(255), jnp.uint32(0)) << (
                        lane * 8
                    )
            if pipeline:

                @pl.when(previous_live & ~conflict)
                def _nonconflict_wait():
                    # The prior store overlaps this tile's reads and compute.
                    # Drain it before starting another store, keeping at most
                    # one pending write and making buffer reuse unambiguous.
                    write(previous, previous_slot).wait()

            old.wait()
            out_buf[slot] = (out_buf[slot] & ~mask) | (packed & mask)
            write(i, fixed_slot).start()
            if not pipeline:
                write(i).wait()
                refill(i)

        @pl.when(desc_ref[6, i] == 0)
        def _inactive():
            if pipeline:

                @pl.when(previous_live)
                def _drain_previous():
                    write(previous, previous_slot).wait()

            refill(i, fixed_slot)

    if single_query and pipeline and q1_unroll:
        # Only Q=1 uses static ping/pong bindings. Keep the traced Q>1 loop
        # unchanged, including its read/modify/write dependency ordering.
        @pl.loop(0, count // 2)
        def _pair(pair):
            _tile(pair * 2, 0)
            _tile(pair * 2 + 1, 1)

        if count % 2:
            _tile(count - 1, 0)
    else:

        @pl.loop(0, count)
        def _loop(i):
            _tile(i)

    if pipeline:

        @pl.when(desc_ref[6, count - 1] != 0)
        def _drain():
            write(count - 1).wait()


def _kernel(
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

        rotated = _hadamard128(pooled.astype(jnp.float32))
        rotated = (rotated * (128**-0.5)).astype(jnp.bfloat16).astype(jnp.float32)
        amax = jnp.maximum(jnp.max(jnp.abs(rotated), axis=1, keepdims=True), 1e-4)
        exponent = jnp.ceil(jnp.log2(amax / 448.0))
        scale = jnp.exp2(exponent)
        q = jnp.clip(rotated / scale, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        # Pallas bitcast packs consecutive rows in the second-last dimension,
        # matching the cache's [quad, 4, byte] -> [quad, 1, word] Ref view.
        key_words = pltpu.bitcast(q, jnp.uint32)
        scale_bytes = jnp.where(
            jnp.arange(128)[None, :] == 0,
            (exponent + 127).astype(jnp.uint8),
            jnp.uint8(0),
        )
        scale_words = pltpu.bitcast(scale_bytes, jnp.uint32)
        packed = jnp.concatenate((key_words, scale_words), axis=1)
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
    decode=False,
    decode_tile_rows=16,
    decode_max_q=None,
    decode_pipeline=True,
    decode_q1_unroll=False,
):
    """Pool newly closed K4 groups; the returned native cache aliases its input.

    Raw writes must already be complete. Destination records must be unique;
    distinct records in a shared physical block are supported.
    Set static decode=True for decode/verify, including multiple query tokens.
    decode_max_q is a static upper bound on each request's query length, NOT
    the accepted speculative length. Omitting it uses num_tokens as a safe
    bound. The caller must include every verify row when supplying a bound.
    The general path remains preferable for large prefill query lengths.
    decode_q1_unroll opts into an experimental static-slot Q=1 pipeline;
    it has not demonstrated a stable latency gain and is disabled by default.
    """
    if kpool != 4 or raw_cache.shape[1:] != (128, 256):
        raise ValueError("fused pooling requires K=4, page_size=128 and D=128")
    if pooled_cache.shape != (raw_cache.shape[0], 32, 4, 256):
        raise ValueError("unexpected pooled-cache layout")
    if raw_cache.dtype != jnp.bfloat16 or pooled_cache.dtype != jnp.uint8:
        raise ValueError("expected BF16 raw cache and uint8 pooled cache")
    if not num_tokens:
        return pooled_cache
    if decode:
        if decode_tile_rows not in (16, 32):
            raise ValueError("decode_tile_rows must be 16 or 32")
        max_q = num_tokens if decode_max_q is None else decode_max_q
        if not isinstance(max_q, int) or max_q < 1:
            raise ValueError("decode_max_q must be a positive static integer")
        desc = _decode_descriptors(
            kv_lens, page_indices, cu_q_lens, distribution, decode_tile_rows, max_q
        )
        pipeline = decode_pipeline and desc.shape[1] > 1
        buffers = 2 if pipeline else 1
        return pl.pallas_call(
            partial(
                _decode_kernel,
                tile_rows=decode_tile_rows,
                pipeline=pipeline,
                single_query=max_q == 1,
                q1_unroll=decode_q1_unroll,
            ),
            out_shape=jax.ShapeDtypeStruct(pooled_cache.shape, pooled_cache.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1,),
                in_specs=(
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(),
                ),
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=(
                    pltpu.VMEM((buffers, decode_tile_rows, 256), jnp.bfloat16),
                    pltpu.VMEM((buffers, decode_tile_rows // 16, 256), jnp.uint32),
                    pltpu.SemaphoreType.DMA((buffers,)),
                    pltpu.SemaphoreType.DMA((buffers,)),
                    pltpu.SemaphoreType.DMA((buffers,)),
                ),
            ),
            input_output_aliases={2: 0},
            name=f"kpool_decode_store_k4_d128_r{decode_tile_rows}_p{buffers}",
        )(desc, raw_cache, pooled_cache, ape.astype(jnp.float32))
    tile_rows = 32 if num_tokens < 128 else 128
    count, desc = _descriptors(
        kv_lens, page_indices, cu_q_lens, distribution, num_tokens, tile_rows
    )
    return pl.pallas_call(
        partial(_kernel, tile_rows=tile_rows),
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
        name="kpool_compress_store_k4_d128",
    )(
        count,
        desc,
        raw_cache,
        pooled_cache,
        ape.astype(jnp.float32),
    )
