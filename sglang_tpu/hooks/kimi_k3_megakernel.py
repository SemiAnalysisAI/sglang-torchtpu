# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode megakernel attach (opt-in, SGLANG_TPU_ENABLE_K3_MEGAKERNEL).

attach_kimi_k3_megakernel
  Why: the megakernel restacks the routed experts SGLang's loader already
       quantized and allocates its decode state, so it must run after the
       loader's quant post-processing and before the KV pools are sized; core
       has no platform hook at that point, and the model file must not import
       sglang_tpu.
  How: AFTER ModelRunner.maybe_precompile_model_kernels_after_loading, which
       ModelRunner.load_model calls once the loader returns, builds the
       megakernel and re-annotates the language model with its compile
       policy. Passes through unless SGLANG_TPU_ENABLE_K3_MEGAKERNEL is set
       (validated at startup by sglang_tpu.kimi_k3_megakernel.config).
  Upstream PR or issue: none filed yet; the seam is a platform post-load-model
       hook (SRTPlatform.on_model_loaded(model_runner)).
  Remove when: that seam lands and sglang_tpu attaches through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def attach_kimi_k3_megakernel(
    result: None, model_runner: ModelRunner, *args, **kwargs
) -> None:
    """AFTER ModelRunner.maybe_precompile_model_kernels_after_loading."""
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_ENABLE_K3_MEGAKERNEL.get():
        return
    from sglang_tpu.kimi_k3_megakernel.attach import attach

    attach(
        model_runner.model,
        model_path=model_runner.model_config.model_path,
        context_length=model_runner.model_config.context_len,
    )
