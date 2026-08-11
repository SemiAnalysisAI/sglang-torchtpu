"""Measured block-size overrides for the batched-RPA kernel.

``calculate_block_sizes`` in ``wrapper.py`` picks block sizes from a VMEM-fit
heuristic. The heuristic has no way to know where the DMA/MXU trade-off
actually lands for a given model shape and context length, so a measured entry
here wins when one matches. Entries are keyed on everything that changes the
answer and nothing that does not, so an unseen deployment simply falls through
to the heuristic.

To add an entry: sweep ``(bkv_sz, batch_size, n_buffer)`` over the VMEM-fit
space for the shape, take the median of interleaved repetitions, and paste the
winner. Only add a row you have measured on the hardware named in the key, and
only when it beats the heuristic by more than run-to-run noise.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.layers.attention.pallas.batched_rpa import configs, utils

# (bq_sz, bq_c_sz, bkv_sz, batch_size, n_buffer)
TunedEntry = tuple[int, int, int, int, int]


def _token_bucket(total_q_tokens: int) -> int:
    """Round the padded token count up to a power of two.

    Decode latency depends on how many sequences share a step, but sweeping
    every token count is not practical, so entries are keyed on the bucket.
    """
    return 1 << max(0, (max(total_q_tokens, 1) - 1).bit_length())


def make_lookup_key(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    mode: configs.RpaCase,
    vmem_limit_bytes: int,
) -> tuple:
    """Build the tuned-table key for one kernel configuration.

    ``vmem_limit_bytes`` is part of the key because an entry is only valid for
    the budget it was measured under: the lookup bypasses the VMEM-fit search,
    so a smaller budget must fall back to the heuristic rather than reuse a
    pick that no longer fits.
    """
    aligned_q_per_kv = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    return (
        f"TPU{pltpu.get_tpu_info().generation}",
        serve_cfgs.page_size,
        f"q_{jnp.dtype(serve_cfgs.dtype_q).name}_kv_{jnp.dtype(serve_cfgs.dtype_kv).name}",
        f"q{aligned_q_per_kv}_kv{model_cfgs.num_kv_heads}_d{model_cfgs.head_dim}",
        serve_cfgs.max_model_len,
        mode.symbol,
        _token_bucket(serve_cfgs.total_q_tokens),
        "sw" if model_cfgs.sliding_window is not None else "none",
        vmem_limit_bytes,
    )


TUNED_BLOCK_SIZES: dict[tuple, TunedEntry] = {}


def get_tuned_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    mode: configs.RpaCase,
    vmem_limit_bytes: int,
) -> configs.BlockSizes | None:
    """Return the measured block sizes for this configuration, if any."""
    entry = TUNED_BLOCK_SIZES.get(
        make_lookup_key(model_cfgs, serve_cfgs, mode, vmem_limit_bytes)
    )
    if entry is None:
        return None
    bq_sz, bq_c_sz, bkv_sz, batch_size, n_buffer = entry
    return configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_c_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=n_buffer,
    )
