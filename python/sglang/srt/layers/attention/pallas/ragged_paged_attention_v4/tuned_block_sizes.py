"""Auto-tuned block sizes for ragged paged attention."""

import collections
import os

from .decode_tuned_block_sizes import TUNED_BLOCK_SIZES as DECODE_BLOCKS
from .mixed_tuned_block_sizes import TUNED_BLOCK_SIZES as MIXED_BLOCKS
from .seq_tuned_block_sizes import DECODE_TUNED_BLOCK_SIZES as SEQ_DECODE_BLOCKS
from .seq_tuned_block_sizes import TUNED_BLOCK_SIZES as SEQ_MIXED_BLOCKS
from .util import (
    RPAKind,
    align_to,
    cdiv,
    get_dtype_bitwidth,
    get_dtype_packing,
    get_tuned_block_sizes_impl,
    next_power_of_2,
)

TUNED_BLOCK_SIZES_BY_KIND = {
    # for prefill, {chunk_prefill_size: tuned_block_sizes}
    RPAKind.PREFILL: collections.defaultdict(lambda: MIXED_BLOCKS),
    RPAKind.DECODE: DECODE_BLOCKS,
    RPAKind.MIXED: MIXED_BLOCKS,
}

# SEQ_ALONG_LANE uses seq-specific prefill/mixed entries (see
# seq_tuned_block_sizes.py) and a page-512-augmented decode overlay (page 512 was
# absent from the shared decode table, which KeyErrored the whole seq lookup and
# forced prefill/mixed to the untuned fallback too). The pre-existing page-128
# seq decode still resolves through the deep-copied shared decode table.
SEQ_TUNED_BLOCK_SIZES_BY_KIND = {
    RPAKind.PREFILL: collections.defaultdict(lambda: SEQ_MIXED_BLOCKS),
    RPAKind.DECODE: SEQ_DECODE_BLOCKS,
    RPAKind.MIXED: SEQ_MIXED_BLOCKS,
}


# Target KV bytes for _decode_kv_pages_per_blk_for_target's full-attention branch:
# size the KV block so ONE double-buffer VMEM slot holds ~18 MB of KV cache (the
# kernel double-buffers the KV scratch, so total KV VMEM is ~2x this, ~36 MB).
# get_tuned_block_sizes applies the decode override only to sliding-window layers,
# so this full-attention target is not used by the production caller today.
DECODE_KV_BUFFER_TARGET_BYTES = 18 * 1024 * 1024

# num_sublanes is 8 on all current TPUs (v4-v7). SEQ_ALONG_LANE aligns head_dim
# to num_sublanes * kv_packing (not 128); see kernel.get_kv_cache_shape.
_NUM_SUBLANES = 8


def _decode_kv_pages_per_blk_for_target(
    kv_dtype,
    num_kv_heads,
    head_dim,
    page_size,
    pages_per_seq,
    is_hd64,
    seq_along_lane,
    sliding_window=None,
    decode_q_len=1,
    target_bytes=DECODE_KV_BUFFER_TARGET_BYTES,
) -> int:
    """num_kv_pages_per_blk for the decode region (clamped to [1, pages_per_seq]).

    Sliding-window decode: the kernel attends to a bkv_sz-block-aligned superset of
    the window -- ``[floor((kv_len - q_len - window) / bkv_sz), cdiv(kv_len, bkv_sz))``
    (see ``bkv_idx_start`` in kernel.py). Sizing the block to ~one window
    (``cdiv(window, page_size)``) puts a block boundary inside that range, so the
    kernel runs TWO block iterations for the same in-window work -- doubling the
    per-block fixed overhead (DMA wait, stitch, per-head setup, pipeline drain). A
    block of ~2x the window (rounded up to a power-of-2 page count, so it divides
    power-of-2 contexts) swallows the window plus the floor misalignment into ONE
    iteration. That is what the autotuner picks -- e.g. 32 pages for a 2048-token /
    16-page window, measured ~1.6x faster than the 1x cap. The window working set is
    tiny, so the larger block's modest over-read is cheap vs the saved iteration.

    Full-attention decode (sliding_window is None): size the block so one
    double-buffer VMEM slot ~= target_bytes. Inverts the per-slot VMEM byte formula
    for each KV layout, mirroring ``bkv_double_buf`` / ``get_kv_cache_shape`` in
    kernel.py. ``pk`` packs sub-32-bit dtypes (32 // bitwidth); ``itemsize`` is the
    true per-element storage; the ``// pk * pk`` factors cancel.
    """
    if sliding_window is not None:
        # Window-anchored decode (env-gated, hardcoded for the 2048 local-layer
        # window-anchored decode (env-gated, hardcoded for the 2048 local-layer
        # window). The kernel anchors the single KV block at the page-aligned
        # window start (see WINDOW_ANCHORED in kernel.py), so it reads exactly
        # one window + page-misalignment + spec slack in ONE iteration:
        #   pages = cdiv(window + decode_q_len + page_size - 1, page_size)
        # e.g. 2048 -> 17 pages (q=1) / 18 (q=4), vs #962's next_pow2(2*window)=32.
        # ~16% faster decode for SWA layers; numerics match within fp8 noise. ON by
        # default; set TPU_RPA_SW_BOUND=0 to disable.
        if (
            seq_along_lane
            and sliding_window == 2048
            and os.environ.get("TPU_RPA_SW_BOUND", "1") == "1"
        ):
            anchored_pages = cdiv(
                sliding_window + decode_q_len + page_size - 1, page_size
            )
            return max(1, min(pages_per_seq, anchored_pages))
        win_pages = next_power_of_2(2 * cdiv(sliding_window, page_size))
        return max(1, min(pages_per_seq, win_pages))

    pk = get_dtype_packing(kv_dtype)
    itemsize = get_dtype_bitwidth(kv_dtype) // 8

    if seq_along_lane:
        # Lane axis = (bkv_p + 2) * page_size (2-page new-token staging slack);
        # head_dim aligned to num_sublanes * pk (no 128 padding in this layout).
        bytes_per_page = (
            page_size
            * (2 * num_kv_heads)
            * align_to(head_dim, _NUM_SUBLANES * pk)
            * itemsize
        )
        bkv_p = target_bytes // bytes_per_page - 2
    elif is_hd64:
        # hd64 packs K|V into the 128-lane head_dim; kv_heads aligned to pk.
        bytes_per_page = page_size * align_to(num_kv_heads, pk) * 128 * itemsize
        bkv_p = target_bytes // bytes_per_page
    else:
        # HEAD_ALONG_SUBLANE: K/V interleaved via 2*kv_heads; head_dim -> 128.
        bytes_per_page = (
            page_size
            * align_to(2 * num_kv_heads, pk)
            * align_to(head_dim, 128)
            * itemsize
        )
        bkv_p = target_bytes // bytes_per_page

    return max(1, min(pages_per_seq, bkv_p))


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
    chunk_prefill_size,
    sliding_window,
    is_hd64=False,
    seq_along_lane=False,
    decode_q_len=1,
) -> tuple[int, int, int, int, int, int]:
    """Search tuned values for (num_kv_pages_per_blk, num_queries_per_blk).

    For SLIDING-WINDOW decode the DECODE KV block is overridden (not looked up) with
    a window-sized block -- ~2x the window, so its attended range stays a single
    KV-block iteration (see _decode_kv_pages_per_blk_for_target). Full-attention
    decode keeps its autotuned KV block. The decode query block always tracks
    decode_q_len (num_spec + 1). Prefill and mixed always keep their autotuned
    entries.
    """
    data = (
        SEQ_TUNED_BLOCK_SIZES_BY_KIND if seq_along_lane else TUNED_BLOCK_SIZES_BY_KIND
    )
    (
        _bkv_p_decode,
        _bq_decode,
        bkv_p_prefill,
        bq_prefill,
        bkv_p_mixed,
        bq_mixed,
    ) = get_tuned_block_sizes_impl(
        data,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size,
        max_num_tokens,
        pages_per_seq,
        chunk_prefill_size,
        # The seq_along_lane layout pads head_dim to 128 so is_hd64 is False, but
        # it still serves sliding-window (local-attention) layers and has
        # window-tuned prefill/mixed entries (inherited from the shared mixed
        # table). Key prefill/mixed on the window for it too, otherwise local and
        # full layers collapse to the same (full-attention) block sizes and local
        # layers lose their speedup.
        use_sliding_window=is_hd64 or seq_along_lane,
        sliding_window=sliding_window,
        use_sliding_window_decode=is_hd64 or sliding_window is not None,
        multiply_num_kv_heads=not is_hd64,
        align_head_dim=not is_hd64,
    )

    # Override the DECODE KV block for SLIDING-WINDOW layers: reuse
    # _decode_kv_pages_per_blk_for_target to size it to ~one window, so the kernel's
    # window-anchored attended range stays a SINGLE KV-block iteration (~2x faster
    # than the autotuned full-target block, which over-reads out-of-window pages).
    # This applies to every batch size -- unlike the (disabled) full-attention
    # override, the window block is SMALL, so it helps single-token (bs1) decode too
    # rather than regressing it. Full-attention decode keeps its autotuned KV block:
    # past ~16 MB in flight it is HBM-bandwidth-bound, so the block size barely
    # matters and the autotune harness times it more reliably. The decode query
    # block always tracks decode_q_len (the decode region's static q-length,
    # num_spec + 1, which spec decode requires).
    if sliding_window is not None:
        bkv_p_decode = _decode_kv_pages_per_blk_for_target(
            kv_dtype,
            actual_num_kv_heads,
            head_dim,
            page_size,
            pages_per_seq,
            is_hd64,
            seq_along_lane,
            sliding_window=sliding_window,
            decode_q_len=decode_q_len,
        )
    else:
        bkv_p_decode = _bkv_p_decode
    bq_decode = decode_q_len

    return (
        bkv_p_decode,
        bq_decode,
        bkv_p_prefill,
        bq_prefill,
        bkv_p_mixed,
        bq_mixed,
    )
