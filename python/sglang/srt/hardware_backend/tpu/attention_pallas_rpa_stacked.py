# SPDX-License-Identifier: Apache-2.0
"""Stacked RPA backend for the TPU torch_tpu platform."""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    RpaKernelSpec,
)


def _import_stacked_kernel():
    from sglang.srt.layers.attention.pallas.stacked_rpa.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


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
):
    del cu_kv_lens, attention_sink
    return jax_rpa(
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        use_causal_mask=True,
        update_kv_cache=True,
    )


RPA_STACKED_SPEC = RpaKernelSpec(
    # Bump the suffix whenever invocation semantics change so old compiled
    # artifacts cannot be reused with a different static kernel path.
    op_name="sglang_tpu_pallas::ragged_paged_attention_stacked_decode_v1",
    import_kernel=_import_stacked_kernel,
    invoke_kernel=_invoke_stacked_kernel,
    supports_attention_sink=False,
    extend_distribution_mask=(0, 0, 0),
    uses_draft_extend_v2=False,
    min_page_size=None,
    seq_along_lane_min_page_size=128,
    seq_along_lane_only=True,
    kv_cache_rank=4,
)


class TpuPallasRpaStackedAttnBackend(BaseTpuPallasRpaAttnBackend):
    """Decode-only stacked RPA backend for use with RPA-seq prefill."""

    kernel_spec = RPA_STACKED_SPEC

    def __init__(self, model_runner, speculative_step_id=0):
        prefill_backend = getattr(model_runner, "prefill_attention_backend_str", None)
        if prefill_backend != "pallas_tpu_rpa_seq":
            raise ValueError(
                "pallas_tpu_rpa_stacked is decode-only and requires "
                "--attention-backend pallas_tpu_rpa_seq with "
                "--decode-attention-backend pallas_tpu_rpa_stacked."
            )
        super().__init__(model_runner, speculative_step_id)

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "pallas_tpu_rpa_stacked is decode-only. Configure "
            "--attention-backend pallas_tpu_rpa_seq and "
            "--decode-attention-backend pallas_tpu_rpa_stacked."
        )
