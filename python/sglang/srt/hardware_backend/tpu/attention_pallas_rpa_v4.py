# SPDX-License-Identifier: Apache-2.0
"""Pallas v4 ragged-paged-attention backend for the TPU platform."""

# NOTE: do not add ``from __future__ import annotations`` here. The Pallas
# export path needs concrete JAX annotations; see attention_pallas_rpa_base.

from typing import Optional

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    BaseTpuPallasRpaMultiStepDraftBackend,
    RpaKernelSpec,
)

_TPU7_KV_TOKENS_PER_BLOCK = 4096
_TPU7_QUERIES_PER_BLOCK = 32


def _import_v4_kernel():
    from sglang.srt.layers.attention.pallas.ragged_paged_attention_v4 import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


def _invoke_v4_kernel(
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
):
    del cu_kv_lens  # RPA v4 derives sequence offsets from kv_lens/page_indices.
    page_size = kv_cache_fused.shape[1]
    if page_size <= 0 or _TPU7_KV_TOKENS_PER_BLOCK % page_size:
        raise ValueError(
            "RPA v4 requires the page size to divide "
            f"{_TPU7_KV_TOKENS_PER_BLOCK}; got page_size={page_size}."
        )
    return jax_rpa(
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sink,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        # TorchTPU exports Pallas through CPU-only JAX. Pass the TPU7 defaults
        # explicitly because the kernel tuner cannot inspect physical devices.
        num_kv_pages_per_block=_TPU7_KV_TOKENS_PER_BLOCK // page_size,
        num_queries_per_block=_TPU7_QUERIES_PER_BLOCK,
    )


RPA_V4_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention_v4",
    import_kernel=_import_v4_kernel,
    invoke_kernel=_invoke_v4_kernel,
    supports_attention_sink=True,
    extend_distribution_mask=(0, 1, 1),
    uses_draft_extend_v2=True,
    min_page_size=None,
)


class TpuPallasRpaV4AttnBackend(BaseTpuPallasRpaAttnBackend):
    """RPA v4 attention backend (``--attention-backend pallas_tpu_rpa_v4``)."""

    kernel_spec = RPA_V4_SPEC

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int] = None
    ):
        pass


class TpuPallasRpaV4MultiStepDraftBackend(BaseTpuPallasRpaMultiStepDraftBackend):
    """One RPA v4 backend per EAGLE draft decode step."""

    attn_backend_cls = TpuPallasRpaV4AttnBackend
