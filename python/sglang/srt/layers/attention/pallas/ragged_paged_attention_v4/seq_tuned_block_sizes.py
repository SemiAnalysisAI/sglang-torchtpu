"""SEQ_ALONG_LANE-specific tuned block sizes.

The seq layout uses a distinct tuned-block lookup key from the head layout
(``align_head_dim`` pads head_dim to 128 and ``multiply_num_kv_heads`` doubles
kv heads), so it indexes its own entries. This module overlays seq-specific
values on top of the deep-copied shared mixed table (used for both prefill and
mixed on the seq path) so the shared table stays correct for genuine
head_dim-128 models.

Only entries that differ from the base are listed in ``_SEQ_MIXED_OVERRIDES``;
the rest fall through to the deep-copied base. Missing nested paths are created
on demand, so seq keys absent from the base table (e.g. page_size=128 / fp8-KV
combinations the head tables were never tuned for) are still populated.

Sliding-window entries (``(max_model_len, 2048)``) come from the seq_along_lane
autotune sweep (``--kv_layout=seq_along_lane``); they are what restores the
local-attention speedup on the seq path (without them local layers fell back to
the full-attention block sizes). See get_tuned_block_sizes(seq_along_lane=True).

NOTE: decode is intentionally NOT overridden here. Seq decode keeps using the
shared decode table; the autotune harness times with wall-clock around
block_until_ready, which is dominated by dispatch overhead for ~10us decode
kernels and therefore unreliable for picking decode block sizes.
"""

import copy

from .decode_tuned_block_sizes import TUNED_BLOCK_SIZES as _BASE_DECODE
from .mixed_tuned_block_sizes import TUNED_BLOCK_SIZES as _BASE_MIXED

# (device, page_size, dtypes, head_key, (max_model_len, sliding_window)) -> (bkv_p, bq)
#
# Prefill/mixed seq entries (SEQ_ALONG_LANE serves both from this table).
_SEQ_MIXED_OVERRIDES = {
    # Short prefill: seq pads head_dim to 128, so the shared table picked bq_8;
    # bq_32 (matching head) keeps the MXU fed and is ~2.6x faster on q1024 with
    # no correctness change.
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (1024, None),
    ): (8, 32),
    # seq_along_lane autotune sweep (T5-42B TP16, bf16-Q/fp8-KV, page128),
    # measured with real on-device kernel time (profiler trace).
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (8192, None),
    ): (16, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (8192, 2048),
    ): (4, 64),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (32768, None),
    ): (16, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (32768, 2048),
    ): (4, 64),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (131072, None),
    ): (16, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (131072, 2048),
    ): (4, 64),
    # Additional seq_along_lane sweep coverage for llama4x-t5 TP8 and TP4.
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (8192, None),
    ): (8, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (8192, 2048),
    ): (4, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (32768, None),
    ): (8, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (32768, 2048),
    ): (8, 16),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-32_kv_head-4_head-128",
        (8192, None),
    ): (8, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-32_kv_head-4_head-128",
        (8192, 2048),
    ): (4, 32),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-32_kv_head-4_head-128",
        (32768, None),
    ): (8, 16),
    (
        "TPU7x",
        128,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-32_kv_head-4_head-128",
        (32768, 2048),
    ): (4, 32),
    # ---------------------------------------------------------------------
    # Page-512 seq prefill/mixed (T5-42B TP16, bf16-Q/fp8-KV), measured with
    # on-device RPA kernel-op time (profiler trace; op_name_prefix="RPA").
    # Winner is uniform across q=1k..8k and across global/local: the smallest KV
    # block (bkv_p=1 -> bkv_sz=512) plus the largest query block (bq=256) keeps
    # the MXU fed and avoids masked-tail over-read -- 1.3-4x faster than the
    # get_default_block_sizes fallback (bkv_p=8/bq=32) that page 512 hit before,
    # and faster than page 128. Without these, page-512 prefill/mixed KeyErrors
    # the lookup and falls back to the untuned default. Same (bkv_p, bq) for the
    # windowed (2048, local) and non-windowed (None, global) keys.
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (8192, None),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (8192, 2048),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (32768, None),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (32768, 2048),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (131072, None),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        (131072, 2048),
    ): (1, 256),
    # Page-512 seq coverage for the attention-DP TP8 shape. Without these
    # entries the combined decode/prefill/mixed lookup falls back to the generic
    # page-512 defaults for every TP8 decode layer.
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (8192, None),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (8192, 2048),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (32768, None),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (32768, 2048),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (131072, None),
    ): (1, 256),
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-16_kv_head-2_head-128",
        (131072, 2048),
    ): (1, 256),
}


def _build_with_overrides(base, overrides):
    blocks = copy.deepcopy(base)
    for (device, page, dtypes, head_key, mml), value in overrides.items():
        node = blocks.setdefault(device, {}).setdefault(page, {})
        node = node.setdefault(dtypes, {}).setdefault(head_key, {})
        node[mml] = value
    return blocks


TUNED_BLOCK_SIZES = _build_with_overrides(_BASE_MIXED, _SEQ_MIXED_OVERRIDES)


# ---------------------------------------------------------------------------
# Page-512 seq DECODE overrides (T5-42B TP16, bf16-Q/fp8-KV).
#
# The shared decode table (decode_tuned_block_sizes.py) has no page-512 entries
# for this seq head key, so the page-512 lookup KeyErrored. That KeyError is
# fatal for the WHOLE lookup: get_tuned_block_sizes_impl does the decode +
# prefill + mixed lookups in one try/except, so a missing decode entry dropped
# prefill AND mixed to the untuned get_default_block_sizes fallback too. These
# entries make the decode lookup succeed, which is what lets the page-512
# prefill/mixed overrides above actually take effect.
#
# Value bkv_p=16 (bkv_sz=8192) for the global (sliding_window=None) keys: from
# the on-device spec-decode (decode_q_len=4) sweep it is the only fixed page-512
# block that beats page 128 at EVERY context (short and long) -- bkv_p=8 (the old
# fallback) is faster for <=4k but worse than page 128 at 16k+, and bkv_p=32 only
# wins at 16k+ while regressing short/mid. The windowed (2048) keys are required
# only so the local-layer decode lookup doesn't KeyError; their value is IGNORED
# because get_tuned_block_sizes overrides sliding-window decode with the
# window-anchored _decode_kv_pages_per_blk_for_target sizing.
_SEQ_DECODE_OVERRIDES = {
    (
        "TPU7x",
        512,
        "q_bfloat16_kv_float8_e4m3fn",
        "q_head-8_kv_head-2_head-128",
        mml_key,
    ): (16, 32)
    for mml_key in (
        (8192, None),
        (8192, 2048),
        (16384, None),
        (16384, 2048),
        (32768, None),
        (32768, 2048),
        (65536, None),
        (65536, 2048),
        (131072, None),
        (131072, 2048),
    )
}
_SEQ_DECODE_OVERRIDES.update(
    {
        (
            "TPU7x",
            512,
            "q_bfloat16_kv_float8_e4m3fn",
            "q_head-16_kv_head-2_head-128",
            mml_key,
        ): (16, 32)
        for mml_key in (
            (8192, None),
            (8192, 2048),
            (16384, None),
            (16384, 2048),
            (32768, None),
            (32768, 2048),
            (65536, None),
            (65536, 2048),
            (131072, None),
            (131072, 2048),
        )
    }
)


DECODE_TUNED_BLOCK_SIZES = _build_with_overrides(_BASE_DECODE, _SEQ_DECODE_OVERRIDES)
