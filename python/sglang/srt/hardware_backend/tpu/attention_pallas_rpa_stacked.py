# SPDX-License-Identifier: Apache-2.0
"""Stacked Pallas ragged-paged-attention backend for the TPU torch_tpu platform.

The stacked kernel splits one request's KV work across the kernel's lanes
instead of packing one request per lane, and reads a SEQ_ALONG_LANE KV cache
(``TpuSeqAlongLaneKVPool``) rather than the fused 5D one the v3 and batched
backends use. Selected with ``--attention-backend pallas_tpu_rpa_stacked``.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    BaseTpuPallasRpaMultiStepDraftBackend,
    PallasForwardMetadata,
    RpaKernelSpec,
)

__all__ = [
    "PallasForwardMetadata",
    "TpuPallasRpaStackedAttnBackend",
    "TpuPallasRpaStackedMultiStepDraftBackend",
]


def _import_stacked_kernel():
    from sglang.srt.layers.attention.pallas.stacked_rpa.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


def _import_stacked_schedule_builder():
    from sglang.srt.layers.attention.pallas.stacked_rpa.wrapper import (
        build_schedules,
        flatten_schedules,
        schedule_leaf_shapes,
    )

    return build_schedules, flatten_schedules, schedule_leaf_shapes


def _split_schedule(
    schedule_flat, queries, keys, kv_cache, kv_lens, page_indices, **static
):
    """Rebuild the schedule leaves from the concatenated tensor.

    The kernel bridge passes only tensors, so the step's schedule arrives as
    one 1-D int32 array; the leaf shapes are a pure function of the static
    config, so the split needs nothing that is not already known here.
    """
    if schedule_flat is None:
        return None

    from sglang.srt.layers.attention.pallas.stacked_rpa.wrapper import (
        schedule_leaf_shapes,
    )

    shapes = schedule_leaf_shapes(
        queries, keys, kv_cache, kv_lens, page_indices, **static
    )
    leaves, offset = [], 0
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= dim
        leaves.append(schedule_flat[offset : offset + size].reshape(shape))
        offset += size
    return tuple(leaves)


def _invoke_stacked_kernel(
    jax_rpa,
    *,
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    attention_sink,
    sm_scale,
    sliding_window,
    soft_cap,
    k_scale,
    v_scale,
    dispatch_hint,
    schedule_flat,
):
    # cu_kv_lens is not consumed by the stacked kernel: its page table is
    # rectangular, so the page offset is derived from pages_per_seq.
    del cu_kv_lens
    return jax_rpa(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache_fused,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=attention_sink,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        use_causal_mask=True,
        update_kv_cache=True,
        dispatch_hint=dispatch_hint,
        precomputed_schedule_leaves=_split_schedule(
            schedule_flat,
            queries,
            keys,
            kv_cache_fused,
            kv_lens,
            page_indices,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            k_scale=k_scale,
            v_scale=v_scale,
            dispatch_hint=dispatch_hint,
        ),
    )


RPA_STACKED_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention_stacked",
    import_kernel=_import_stacked_kernel,
    invoke_kernel=_invoke_stacked_kernel,
    supports_attention_sink=True,
    extend_distribution_mask=(0, 0, 1),  # [0, 0, n] (MIXED-only)
    uses_draft_extend_v2=False,
    # Upstream's guidance and tuned entries target page_size 2048, but that was
    # measured on a different VMEM budget: here 2048 makes the widest useful
    # tiles overflow VMEM (batch_size=8 at bkv=8192 fails to allocate), and 256
    # is 12x faster at every shape swept on v7. See the sglang-side tuned
    # entries in the vendored tuned_block_sizes.py.
    min_page_size=256,
    supports_dispatch_hint=True,
    import_schedule_builder=_import_stacked_schedule_builder,
)


class TpuPallasRpaStackedAttnBackend(BaseTpuPallasRpaAttnBackend):
    """Stacked RPA attention backend (``--attention-backend pallas_tpu_rpa_stacked``)."""

    kernel_spec = RPA_STACKED_SPEC


class TpuPallasRpaStackedMultiStepDraftBackend(BaseTpuPallasRpaMultiStepDraftBackend):
    """One stacked-RPA backend per EAGLE draft decode step."""

    attn_backend_cls = TpuPallasRpaStackedAttnBackend
