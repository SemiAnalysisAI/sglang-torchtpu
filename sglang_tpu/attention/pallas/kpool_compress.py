# SPDX-License-Identifier: Apache-2.0
"""K4/D128 pooling directly from raw pages into the native pooled cache.

One tile computes 32 pools from a raw page. Raw reads and output writes use
two buffers. Output read-modify-write starts only after the preceding store
finishes, including when two requests share a physical destination block.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _descriptors(kv_lens, page_indices, cu_q_lens, distribution, num_tokens):
    """One descriptor for each raw page containing a newly closed pool."""
    batch = kv_lens.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    req = jnp.minimum(jnp.searchsorted(cu_q_lens[1:], tok, side="right"), batch - 1)
    start = kv_lens[req] - (cu_q_lens[req + 1] - cu_q_lens[req])
    pos = start + tok - cu_q_lens[req]
    closing = (tok < cu_q_lens[distribution[2]]) & (pos % 4 == 3)
    first = closing & ((pos % 128 == 3) | (tok - cu_q_lens[req] < 4))
    # Each request contributes at most two boundary pages plus q_len // 128.
    capacity = min(num_tokens, num_tokens // 128 + 2 * batch)
    chosen = jnp.nonzero(first, size=capacity, fill_value=0)[0]
    r, page = req[chosen], pos[chosen] // 128
    pages_per_seq = page_indices.shape[0] // batch
    desc = jnp.stack(
        (
            page_indices[r * pages_per_seq + page],
            page_indices[r * pages_per_seq + page // 4],
            (page % 4) * 8,
            start[chosen] - page * 128,
            kv_lens[r] - page * 128,
        )
    ).astype(jnp.int32)
    return jnp.sum(first, dtype=jnp.int32)[None], desc


def _kernel(
    count_ref,
    desc_ref,
    raw_ref,
    old_ref,
    ape_ref,
    h_ref,
    out_ref,
    raw_buf,
    out_buf,
    read_sem,
    write_sem,
    old_sem,
):
    del old_ref  # out_ref aliases this input, including untouched bytes.
    words = out_ref.bitcast(jnp.uint32)
    count = count_ref[0]

    def read(i):
        return pltpu.make_async_copy(
            raw_ref.at[desc_ref[0, i]], raw_buf.at[i % 2], read_sem.at[i % 2]
        )

    def write(i):
        return pltpu.make_async_copy(
            out_buf.at[i % 2],
            words.at[desc_ref[1, i], pl.ds(desc_ref[2, i], 8)],
            write_sem.at[i % 2],
        )

    @pl.when(count > 0)
    def _prime():
        read(0).start()

    @pl.loop(0, count)
    def _page(i):
        slot = i % 2
        read(i).wait()

        @pl.when(i + 1 < count)
        def _prefetch():
            read(i + 1).start()

        raw = raw_buf[slot].astype(jnp.float32)
        keys = raw[:, :128].reshape((32, 4, 128))
        scores = raw[:, 128:].reshape((32, 4, 128)) + ape_ref[...][None, :, :]
        scores = scores - jnp.max(scores, axis=1, keepdims=True)
        weight = jnp.exp(scores)
        weight = weight / jnp.sum(weight, axis=1, keepdims=True)
        pooled = jnp.sum(weight * keys, axis=1).astype(jnp.bfloat16)
        rotated = jax.lax.dot_general(
            pooled,
            h_ref[...],
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        rotated = (rotated * (128**-0.5)).astype(jnp.bfloat16).astype(jnp.float32)
        amax = jnp.maximum(jnp.max(jnp.abs(rotated), axis=1, keepdims=True), 1e-4)
        exponent = jnp.ceil(jnp.log2(amax / 448.0))
        scale = jnp.exp2(exponent)
        q = jnp.clip(rotated / scale, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        key_bytes = pltpu.bitcast(q, jnp.uint8).astype(jnp.uint32)
        records = jnp.concatenate(
            (
                key_bytes,
                (exponent + 127).astype(jnp.uint32),
                jnp.zeros((32, 127), dtype=jnp.uint32),
            ),
            axis=1,
        ).reshape((8, 4, 256))
        packed = records[:, 0, :]
        for lane in range(1, 4):
            packed = packed | (records[:, lane, :] << (lane * 8))
        ends = jnp.arange(32, dtype=jnp.int32) * 4 + 3
        live = ((ends >= desc_ref[3, i]) & (ends < desc_ref[4, i])).reshape((8, 4, 1))
        mask = jnp.where(live[:, 0, :], jnp.uint32(255), jnp.uint32(0))
        for lane in range(1, 4):
            mask = mask | (
                jnp.where(live[:, lane, :], jnp.uint32(255), jnp.uint32(0))
                << (lane * 8)
            )

        # Serialize the *read* too. A later tile may update different bytes
        # in exactly the same physical destination block as an earlier tile.
        @pl.when(i > 0)
        def _prior_store():
            write(i - 1).wait()

        @pl.when((desc_ref[3, i] > 3) | (desc_ref[4, i] < 128))
        def _preserve():
            old = pltpu.make_async_copy(
                words.at[desc_ref[1, i], pl.ds(desc_ref[2, i], 8)],
                out_buf.at[slot],
                old_sem,
            )
            old.start()
            old.wait()
            out_buf[slot, :, 0, :] = (out_buf[slot, :, 0, :] & ~mask) | (packed & mask)

        @pl.when((desc_ref[3, i] <= 3) & (desc_ref[4, i] >= 128))
        def _overwrite():
            out_buf[slot, :, 0, :] = packed

        write(i).start()

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
):
    """Pool newly closed K4 groups; the returned native cache aliases its input.

    Raw writes must already be complete. Destination records must be unique;
    distinct records in a shared physical block are supported.
    """
    if kpool != 4 or raw_cache.shape[1:] != (128, 256):
        raise ValueError("fused pooling requires K=4, page_size=128 and D=128")
    if pooled_cache.shape != (raw_cache.shape[0], 32, 4, 256):
        raise ValueError("unexpected pooled-cache layout")
    if raw_cache.dtype != jnp.bfloat16 or pooled_cache.dtype != jnp.uint8:
        raise ValueError("expected BF16 raw cache and uint8 pooled cache")
    if not num_tokens:
        return pooled_cache
    count, desc = _descriptors(
        kv_lens, page_indices, cu_q_lens, distribution, num_tokens
    )
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(pooled_cache.shape, pooled_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(),
                pl.BlockSpec(),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.VMEM((2, 128, 256), jnp.bfloat16),
                pltpu.VMEM((2, 8, 1, 256), jnp.uint32),
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
        jnp.asarray(h, dtype=jnp.bfloat16),
    )
