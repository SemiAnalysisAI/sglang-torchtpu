# SPDX-License-Identifier: Apache-2.0
"""Sequence-along-lane Pallas RPA backend for the TPU platform."""

# NOTE: do not add ``from __future__ import annotations`` here. The Pallas
# export path needs concrete JAX annotations; see attention_pallas_rpa_base.

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    RpaKernelSpec,
)


def _import_rpa_seq_kernel():
    from sglang.srt.layers.attention.pallas.ragged_paged_attention_seq import (
        ragged_paged_attention_prefill,
    )

    return ragged_paged_attention_prefill


def _invoke_rpa_seq_kernel(
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
    # The sequence-layout kernel derives offsets from kv_lens/page_indices.
    del cu_kv_lens

    if len(kv_cache_fused.shape) != 4:
        raise ValueError(
            "rpa_seq prefill requires its native rank-4 KV cache, "
            f"but got shape={kv_cache_fused.shape}."
        )

    page_size = kv_cache_fused.shape[-1]
    if page_size <= 0 or page_size % 128:
        raise ValueError(
            "rpa_seq prefill requires page_size to be a positive multiple of "
            f"128; got page_size={page_size}."
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
    )


RPA_SEQ_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention_seq_prefill_v1",
    import_kernel=_import_rpa_seq_kernel,
    invoke_kernel=_invoke_rpa_seq_kernel,
    supports_attention_sink=False,
    extend_distribution_mask=(0, 1, 1),
    uses_draft_extend_v2=True,
    min_page_size=None,
    seq_along_lane_min_page_size=128,
    seq_along_lane_only=True,
    kv_cache_rank=4,
)


class TpuPallasRpaSeqAttnBackend(BaseTpuPallasRpaAttnBackend):
    """Specialized rank-4 ``rpa_seq`` prefill backend."""

    kernel_spec = RPA_SEQ_SPEC

    def __init__(self, model_runner, speculative_step_id=0):
        decode_backend = model_runner.decode_attention_backend_str
        if decode_backend != "pallas_tpu_rpa_stacked":
            raise ValueError(
                "pallas_tpu_rpa_seq is prefill-only and requires "
                "--decode-attention-backend pallas_tpu_rpa_stacked."
            )
        super().__init__(model_runner, speculative_step_id)

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "pallas_tpu_rpa_seq is prefill-only. Configure "
            "--decode-attention-backend pallas_tpu_rpa_stacked."
        )
