# Ported from https://github.com/vllm-project/tpu-inference
# tpu_inference/kernels/experimental/batched_rpa/wrapper.py
# Commit 1d2527907b8289a13db19973d664788ead4f2392 (2026-07-09)
"""Wrapper for RPA kernel to match expected interface.

NOTE: all of the code in this directory is experimental and not fully tested!
To enable usage of this kernel in full run, you can pass the USE_BATCHED_RPA_KERNEL=1
environment variable.

Compared to the default RPA kernel, this kernel does the following:

1. Batches multiple sequences together to replace per-request flash_attention loops.

2. Enables triple-buffering via Pallas emit_pipeline

3. Precomputes expensive metadata upfront (e.g., page locations and bounds clipping) via
scheduler.py kernel. Kernel is calculated once and ammortized across different layers in a model.

Note: batched_rpa is build on top / derived from RPA3.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.layers.attention.pallas.batched_rpa import (
    configs,
    kernel,
    schedule,
    tuned_block_sizes,
    utils,
)

# Fraction of the VMEM budget a block-size pick may use. The estimator does
# not model compiler-emitted scratch, so the rest is left as headroom.
_VMEM_FRACTION = 0.8
# Decode may spend a little more, but only to cut the number of k-blocks per
# sequence (see _pick_decode_block_sizes).
_DECODE_RELAXED_VMEM_FRACTION = 0.9
# Lane cap for decode. Fixed value based on experimental results.
_DECODE_MAX_BATCH = 8


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:

    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    _, actual_num_kv_heads, _ = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head, q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_head_dim = utils.align_to(actual_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (
        jnp.pad(
            q.reshape(
                total_q_tokens,
                actual_num_kv_heads,
                num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
                (0, aligned_head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            total_q_tokens,
            actual_num_kv_heads,
            aligned_num_q_heads_per_kv_head // q_packing,
            q_packing,
            aligned_head_dim,
        )
        .swapaxes(0, 1)
    )

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2, kv_packing)
    assert num_kv_heads_x2_aligned % 2 == 0, (
        f"kv_packing={kv_packing} produces an odd aligned head count "
        f"{num_kv_heads_x2_aligned}"
    )
    new_kv_hbm = jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(
            total_q_tokens, actual_num_kv_heads_x2, actual_head_dim
        ),
        (
            (0, 0),
            (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
            (0, aligned_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        num_kv_heads_x2_aligned // kv_packing,
        kv_packing,
        aligned_head_dim,
    )

    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    num_lanes = pltpu.get_tpu_info().num_lanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim, num_lanes),
    )


def _calculate_vmem_usage(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    *,
    batch_size: int,
    n_buffer: int,
    bq_sz: int,
    bkv_sz: int,
    num_lanes: int,
) -> int:
    """Estimate the kernel's buffered and compute VMEM footprint."""
    aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    aligned_num_q_heads = aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads

    bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2, serve_cfgs.packing_kv)
    if utils.has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize

    bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
    bkv_array_size = bkv_sz * aligned_num_kv_heads_x2 * aligned_head_dim
    bo_array_size = bq_array_size

    buffer_bytes = (
        bq_array_size * q_bytes * n_buffer
        + bkv_array_size * kv_bytes * n_buffer
        + bo_array_size * out_bytes * 2
    )

    loaded_bq_size = bq_sz * model_cfgs.num_q_heads * aligned_head_dim
    loaded_bkv_size = bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim
    qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads
    compute_bytes = (
        loaded_bq_size * q_bytes + loaded_bkv_size * kv_bytes + qk_size * out_bytes
    )
    return (buffer_bytes + compute_bytes) * batch_size


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mxu_column_size = tpu_info.mxu_column_size

    def estimate_vmem_usage(
        batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int
    ) -> int:
        return _calculate_vmem_usage(
            model_cfgs,
            serve_cfgs,
            batch_size=batch_size,
            n_buffer=n_buffer,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            num_lanes=num_lanes,
        )

    def calculate_compute_buffer_time(
        batch_size: int, bq_c_sz: int, bkv_sz: int
    ) -> int:
        """Calculate computational complexity of a single compute block."""

        num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    def find_best_block_sizes(
        max_batch_size: int, max_n_buffer: int, fixed_bq_sz: int | None = None
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        # Even if we loose some potential performance, we want to avoid OOM at all
        # costs. Therefore, we conservatively only use 80% of the VMEM budget.
        capped_vmem_limit_bytes = vmem_limit_bytes * _VMEM_FRACTION

        bkv_sz = bkv_stride = mxu_column_size
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        batch_size = max_batch_size
        n_buffer = max_n_buffer

        # Step 1: Lower batch_size and/or n_buffer if even the smallest bq and bkv
        # size can trigger OOM.

        # If current batch size triggers OOM, decrease batch size until the kernel
        # fits within VMEM limit.
        while (
            estimate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            > capped_vmem_limit_bytes
        ):
            batch_size -= 1

        # As a last resort, attempt to decrease number of buffers to avoid OOM.
        while (
            estimate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            > capped_vmem_limit_bytes
        ):
            n_buffer -= 1

        # Indicates OOM was triggered even when batch_size=1 or n_buffer=1.
        # NOTE: If the function does not exit at this point even when either values
        # are zero, it will trigger infinite loop at the next while loop.
        if batch_size == 0 or n_buffer == 0:
            raise ValueError("Cannot find batch size that fits within VMEM limit.")

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        while (
            estimate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            < capped_vmem_limit_bytes
        ):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        bkv_sz -= bkv_stride
        bq_sz -= bq_stride

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError("Cannot find block sizes that fit within VMEM limit.")

        # Step 3: Given current tile size, calculate compute tile size.

        # Fixed threshold value based on hardware spec.
        # TODO(kyuyeunk): Use different threshold based on hardware and precision.
        threshold = 1500

        num_bq_c = 1
        last_valid_bq_c_sz = bq_c_sz = bq_sz
        bq_c_rem = 0

        while (
            calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz) > threshold
            or bq_c_rem != 0
        ) and num_bq_c < bq_sz:
            if bq_c_rem == 0:
                last_valid_bq_c_sz = bq_c_sz
            num_bq_c += 1
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=last_valid_bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )

    # Default to triple buffer as its almost always beneficial.
    n_buffer = 3
    decode_batch_size = _DECODE_MAX_BATCH
    # Fixed value based on experimental results.
    prefill_batch_size = 2

    decode_block_sizes = _pick_decode_block_sizes(
        estimate_vmem_usage,
        fallback=lambda: find_best_block_sizes(decode_batch_size, n_buffer, 1),
        vmem_limit_bytes=vmem_limit_bytes,
        serve_cfgs=serve_cfgs,
        max_batch_size=decode_batch_size,
        max_n_buffer=n_buffer,
    )
    prefill_block_sizes = find_best_block_sizes(prefill_batch_size, n_buffer)

    return decode_block_sizes, prefill_block_sizes


def _decode_kv_tiles(max_model_len: int, page_size: int):
    """KV tiles that split the context into 1, 2, 3, ... whole blocks.

    Yields ``(num_blocks, bkv_sz)`` widest-first. Only these widths are worth
    considering: a tile between two ladder rungs needs the same number of
    blocks as the narrower rung while paying a wider matmul on the tail of the
    last one.
    """
    seen = set()
    for num_blocks in range(1, max_model_len // page_size + 2):
        bkv_sz = min(
            utils.align_to(pl.cdiv(max_model_len, num_blocks), page_size), max_model_len
        )
        if bkv_sz < page_size:
            return
        if bkv_sz not in seen:
            seen.add(bkv_sz)
            yield num_blocks, bkv_sz


def _pick_decode_block_sizes(
    estimate_vmem_usage,
    *,
    fallback,
    vmem_limit_bytes: int,
    serve_cfgs: configs.ServingConfigs,
    max_batch_size: int,
    max_n_buffer: int,
) -> configs.BlockSizes:
    """Choose decode block sizes for the deployment's context length.

    The legacy pick maximised ``bkv_sz`` at a fixed ``(batch_size=8,
    n_buffer=3)``, ignoring the context length entirely. That leaves a lot on
    the table: the tile it lands on rarely divides the context, so each
    sequence needs several k-blocks, and the flash body does not clip its MXU
    work to ``kv_len``, so every partial block pays a full-width matmul.

    Measured on v7 (Qwen3-Coder-480B TP=8 shapes, bf16 and fp8 KV, decode at
    concurrency 8 and 64), the ordering that tracks the sweep winners is:
    fewest k-blocks per sequence first, then the largest lane count that fits,
    then the deepest buffering that fits. Within 0-3.6% of the per-shape
    optimum from 1k to 32k context.

    The tile is sized from ``max_model_len``, not from the live ``kv_len``,
    which is only known at run time. Sequences much shorter than the window
    therefore pay a wider matmul than they need: measured at 1/4 of the
    window the pick is up to 19% slower than the legacy one, against 46-134%
    faster when sequences do fill it. The downside is bounded and the upside
    is not, so this errs wide.
    """
    max_model_len = serve_cfgs.max_model_len
    page_size = serve_cfgs.page_size

    def fits_step_budget(bkv_sz: int, batch_size: int) -> bool:
        """Whether the schedule for a full batch stays inside SMEM.

        ``compute_metadata`` does not clamp its step counter, so a pick whose
        worst-case schedule is longer than the scratch writes past the end of
        it. Lowering the lane count multiplies the step count, so the picker
        has to check.
        """
        steps = pl.cdiv(
            serve_cfgs.num_seqs * pl.cdiv(max_model_len, bkv_sz), batch_size
        )
        return steps <= configs.max_steps_ub_for(
            serve_cfgs,
            batch_size=batch_size,
            bkv_p_cache=bkv_sz // page_size,
            bkv_p_new=1,
        )

    def search(vmem_fraction: float, max_blocks: int, check_steps: bool):
        limit = vmem_limit_bytes * vmem_fraction
        for num_blocks, bkv_sz in _decode_kv_tiles(max_model_len, page_size):
            if num_blocks > max_blocks:
                return None
            for batch_size in range(max_batch_size, 0, -1):
                if check_steps and not fits_step_budget(bkv_sz, batch_size):
                    continue
                for n_buffer in range(max_n_buffer, 1, -1):
                    if estimate_vmem_usage(batch_size, n_buffer, 1, bkv_sz) <= limit:
                        return num_blocks, configs.BlockSizes(
                            bq_sz=1,
                            bq_c_sz=1,
                            bkv_sz=bkv_sz,
                            batch_size=batch_size,
                            n_buffer=n_buffer,
                        )
        return None

    # Prefer a pick whose schedule fits SMEM; if none does (the schedule
    # scratch is simply too small for this many sequences at this context, a
    # limit the legacy pick hits too) fall back to the perf ordering alone.
    for check_steps in (True, False):
        found = search(_VMEM_FRACTION, max_model_len, check_steps)
        if found is not None:
            break
    if found is None:
        # Nothing on the tile ladder fits even at one page; let the legacy
        # search report the failure (or find something the ladder missed).
        return fallback()

    num_blocks, blocks = found
    if num_blocks > 1:
        # Splitting the context costs a whole extra pipeline turn per sequence
        # plus a partial last block, which is worth more than the margin. The
        # margin covers what the estimator does not model -- chiefly that K and
        # V are both live across QK and PV while ``loaded_bkv_size`` counts one
        # of them -- and decode's own tail (the qk/p intermediates) is small
        # because bq_sz is 1. Widening may also drop a buffer to pay for the
        # tile; that is the intended trade, and it never adds one.
        wider = search(_DECODE_RELAXED_VMEM_FRACTION, num_blocks - 1, check_steps)
        if wider is not None:
            blocks = wider[1]
    return blocks


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "update_kv_cache",
        "stacked",
        "dispatch_hint",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    stacked: bool = False,
    dispatch_hint: str = "auto",
) -> tuple[jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

    Args:
        queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
        keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
        values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
        kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Stores existing kv cache data where k & vs are
            concatenated along num kv heads dim.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
            page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
            sequences. distribution[2] represents total number of sequences.
        sm_scale: Softmax scale value.
        sliding_window: Size of sliding window (also known as local attention). kvs
            outside of the window is not fetched from hbm and masked out during
            computation.
        soft_cap: Cap values of softmax inputs.
        mask_value: Value to use for causal masking. Defaults to smallest
            representable value of the activation dtype.
        q_scale: Quantization scale value of queries.
        k_scale: Quantization scale value of keys.
        v_scale: Quantization scale value of values.
        chunk_prefill_size: Not used.
        decode_block_sizes: Kernel block size to use during decode.
        prefill_block_sizes: Kernel block size to use during prefill.
        vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
            size of the hardware.
        debug_mode: Not used.
        out_dtype: Dtype of output. Defaults to dtype of queries.
        use_causal_mask: Not used.
        dispatch_hint: ``"auto"`` (default) traces both the DECODE and the
            MIXED kernel. ``"decode_only"`` / ``"mixed_only"`` drop the other
            mode -- its attention kernel *and* its schedule pallas -- from the
            traced graph. A serving step is always one or the other, so the
            skipped mode's range would be empty anyway; the hint just lets the
            caller say so statically. Passing a hint that contradicts the
            traced ``distribution`` silently drops that mode's sequences.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Result of new kv cache where k & vs are
            concatenated along num kv heads dim.
    """

    if dispatch_hint not in ("auto", "decode_only", "mixed_only"):
        raise ValueError(f"Unknown {dispatch_hint=}")
    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")

    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    page_size = kv_cache.shape[1]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]

    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
    )

    q_hbm, new_kv_hbm = prepare_inputs(
        queries, keys, values, queries.dtype, kv_cache.dtype
    )

    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs, serve_cfgs, vmem_limit_bytes
    )

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
    ):
        if dispatch_hint == "decode_only" and mode != configs.RpaCase.DECODE:
            return o_hbm_alias_q_hbm, kv_cache
        if dispatch_hint == "mixed_only" and mode != configs.RpaCase.MIXED:
            return o_hbm_alias_q_hbm, kv_cache

        tuned = tuned_block_sizes.get_tuned_block_sizes(
            model_cfgs, serve_cfgs, mode, vmem_limit_bytes
        )
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or tuned or default_decode
            if stacked:
                stacked_bkv_sz = max(page_size, utils.align_to(1024, page_size))
                # Start the refit from the lane cap, not from the decode pick:
                # stacked stripes one request's k-blocks across the lanes, so
                # it wants as many as fit at its own (small) tile, whereas the
                # decode pick trades lanes away for a tile that can be tens of
                # thousands of tokens wide.
                stacked_batch_size = _DECODE_MAX_BATCH
                while (
                    stacked_batch_size > 1
                    and _calculate_vmem_usage(
                        model_cfgs,
                        serve_cfgs,
                        batch_size=stacked_batch_size,
                        n_buffer=effective_blocks.n_buffer,
                        bq_sz=effective_blocks.bq_sz,
                        bkv_sz=stacked_bkv_sz,
                        num_lanes=pltpu.get_tpu_info().num_lanes,
                    )
                    > vmem_limit_bytes
                ):
                    stacked_batch_size -= 1
                effective_blocks = dataclasses.replace(
                    effective_blocks,
                    bkv_sz=stacked_bkv_sz,
                    # The normal tuner sizes request-parallel decode before
                    # stacked mode expands each lane to a 1K KV tile. Refit
                    # the lane count so GQA shapes do not exceed device VMEM.
                    batch_size=stacked_batch_size,
                )
        else:
            effective_blocks = prefill_block_sizes or tuned or default_prefill

        cfgs = configs.RpaConfigs(
            block=effective_blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
            stacked=stacked and mode == configs.RpaCase.DECODE,
        )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
        )

        schedule_hbm = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            cfgs=cfgs,
            update_kv_cache=update_kv_cache,
        )
        return kernel.rpa_kernel(
            cu_q_lens,
            kv_lens,
            page_indices,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            cfgs=cfgs,
        )

    o_hbm_alias_q_hbm, kv_cache = run_rpa_kernel(
        configs.RpaCase.DECODE, q_hbm, kv_cache
    )
    o_hbm_alias_q_hbm, kv_cache = run_rpa_kernel(
        configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache
    )

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    return o_hbm, kv_cache
