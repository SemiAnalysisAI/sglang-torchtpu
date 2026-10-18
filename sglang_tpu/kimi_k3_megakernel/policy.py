# SPDX-License-Identifier: Apache-2.0
"""The compile policy that routes Kimi-K3 decode steps to the megakernel.

The megakernel attach re-annotates the language model with a subclass of the
policy it already had (``support_torch_compile``), so TpuGraphRunner installs
this one instead. It changes three things:

* ``install`` binds the runner to SGLang's pools, which exist by then;
* ``prepare_forward_batch`` runs the runner's per-step bookkeeping after the
  base policy padded the batch, so the staged rows match the compiled tile;
* ``_compile`` compiles a forward whose DECODE branch is one megakernel call
  and whose other modes are the model's own forward.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy

from .runner import KimiK3MegakernelRunner

# The forward_batch attribute carrying runner.StepInputs into the compiled
# forward; set per step, so the graph reads it as an input, not a constant.
STEP_INPUTS_ATTR = "kimi_k3_megakernel_step"


def megakernel_forward(
    forward: Callable[..., LogitsProcessorOutput], runner: KimiK3MegakernelRunner
) -> Callable[..., LogitsProcessorOutput]:
    """``forward`` with DECODE replaced by one megakernel step."""

    @functools.wraps(forward)
    def dispatch(input_ids, positions, forward_batch: ForwardBatch, *args, **kwargs):
        if forward_batch.forward_mode.is_decode():
            step = getattr(forward_batch, STEP_INPUTS_ATTR)
            logits = runner.forward_decode(input_ids, step)
            return LogitsProcessorOutput(next_token_logits=logits)
        return forward(input_ids, positions, forward_batch, *args, **kwargs)

    return dispatch


def megakernel_policy(
    base: type[DecoderTpuCompilePolicy], runner: KimiK3MegakernelRunner
) -> type[DecoderTpuCompilePolicy]:
    """``base`` (the model's own decoder policy) with decode on the megakernel."""

    class KimiK3MegakernelPolicy(base):
        def install(self) -> bool:
            installed = super().install()
            if installed:
                runner.attach_pools(
                    req_to_token_pool=self.model_runner.req_to_token_pool,
                    token_to_kv_pool=self.model_runner.token_to_kv_pool,
                )
            return installed

        def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
            super().prepare_forward_batch(forward_batch)
            mode = forward_batch.forward_mode
            if forward_batch.req_pool_indices_cpu is None:
                raise RuntimeError("the Kimi-K3 megakernel needs req_pool_indices_cpu")
            req_pool_indices = forward_batch.req_pool_indices_cpu.tolist()
            if not mode.is_decode():
                # Refused at startup; a MIXED batch would decode its decode
                # rows through SGLang over stale caches.
                if mode.is_mixed():
                    raise RuntimeError(
                        "the Kimi-K3 megakernel cannot serve MIXED batches"
                    )
                runner.check_not_decoding(req_pool_indices=req_pool_indices)
                return
            if forward_batch.seq_lens_cpu is None:
                raise RuntimeError("the Kimi-K3 megakernel needs seq_lens_cpu")
            step = runner.prepare_decode(
                req_pool_indices=req_pool_indices,
                seq_lens=forward_batch.seq_lens_cpu.tolist(),
                padded_batch_size=forward_batch.batch_size,
            )
            setattr(forward_batch, STEP_INPUTS_ATTR, step)

        def _compile(self, fn: Callable[..., LogitsProcessorOutput]):
            return super()._compile(megakernel_forward(fn, runner))

    KimiK3MegakernelPolicy.__name__ = f"KimiK3Megakernel{base.__name__}"
    KimiK3MegakernelPolicy.__qualname__ = KimiK3MegakernelPolicy.__name__
    return KimiK3MegakernelPolicy
