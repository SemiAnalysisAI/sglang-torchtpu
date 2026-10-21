# SPDX-License-Identifier: Apache-2.0
"""K4/D128 pooling directly from raw pages into the native pooled cache.

One tile computes 8 or 32 pools. Both raw buffers are primed; a consumed
slot is refilled while later stages use its SSA values. Output read-modify-write
starts only after the preceding store finishes, including shared destination blocks.
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


def _decode_descriptors(kv_lens, page_indices, cu_q_lens, distribution, tile_rows):
    """Fixed request slots; decode admits zero or one new token per request."""
    batch = kv_lens.shape[0]
    req = jnp.arange(batch, dtype=jnp.int32)
    q_len = cu_q_lens[1:] - cu_q_lens[:-1]
    live = (req < distribution[2]) & (q_len == 1) & (kv_lens % 4 == 0)
    pos = jnp.maximum(kv_lens - 1, 0)
    pages_per_seq = page_indices.shape[0] // batch
    # Inactive slots must still have safe metadata gathers. They issue no DMA.
    pos = jnp.where(live, pos, 0)
    return jnp.stack(
        (
            page_indices[req * pages_per_seq + pos // 128],
            (pos % 128) // tile_rows,
            page_indices[req * pages_per_seq + pos // 512],
            ((pos % 512) // tile_rows) * (tile_rows // 16),
            (pos % tile_rows) // 4,
            live.astype(jnp.int32),
        )
    ).astype(jnp.int32)


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
):
    del old_ref  # Aliased output preserves every inactive record and byte.
    words = out_ref.bitcast(jnp.uint32)
    word_rows = tile_rows // 16

    @pl.loop(0, desc_ref.shape[1])
    def _request(req):
        @pl.when(desc_ref[5, req] != 0)
        def _closed():
            raw = pltpu.make_async_copy(
                raw_ref.at[
                    desc_ref[0, req], pl.ds(desc_ref[1, req] * tile_rows, tile_rows), :
                ],
                raw_buf,
                read_sem,
            )
            dst = words.at[desc_ref[2, req], pl.ds(desc_ref[3, req], word_rows), 0, :]
            old = pltpu.make_async_copy(dst, out_buf, old_sem)
            # The preceding request's store has drained. Read the old block
            # alongside raw data, before the compute chain, even for count=1.
            raw.start()
            old.start()
            raw.wait()
            pooled = _pool_rows(raw_buf[...], ape_ref[...])
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
            selected = desc_ref[4, req]
            row = jax.lax.broadcasted_iota(jnp.int32, (word_rows, 256), 0)
            mask = jnp.where(
                row == selected // 4,
                jnp.uint32(255) << ((selected % 4) * 8),
                jnp.uint32(0),
            )
            old.wait()
            out_buf[...] = (out_buf[...] & ~mask) | (packed & mask)
            store = pltpu.make_async_copy(out_buf, dst, write_sem)
            store.start()
            # Requests can share a physical quad. Complete its RMW before
            # the next request reads the old bytes or reuses either buffer.
            store.wait()


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
):
    """Pool newly closed K4 groups; the returned native cache aliases its input.

    Raw writes must already be complete. Destination records must be unique;
    distinct records in a shared physical block are supported.
    Set static decode=True only when every request has zero or one new token.
    Small chunked-prefill buckets must use the default general path.
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
        desc = _decode_descriptors(
            kv_lens, page_indices, cu_q_lens, distribution, decode_tile_rows
        )
        return pl.pallas_call(
            partial(_decode_kernel, tile_rows=decode_tile_rows),
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
                    pltpu.VMEM((decode_tile_rows, 256), jnp.bfloat16),
                    pltpu.VMEM((decode_tile_rows // 16, 256), jnp.uint32),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            input_output_aliases={2: 0},
            name=f"kpool_decode_store_k4_d128_r{decode_tile_rows}",
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
