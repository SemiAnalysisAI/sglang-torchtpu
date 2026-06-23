"""TPU "graph runner": orchestrates annotated torch.compile boundaries.

Unlike CudaGraphRunner / NPUGraphRunner there is no graph capture or
replay here — PJRT (libtpu) caches HLO compilations internally, keyed
by tensor shape, so we get "compile once, dispatch by shape" behavior
without an explicit capture step. This class exists so the
torch.compile orchestration hangs off the same factory hook
(``current_platform.get_graph_runner_cls()``) as the in-tree runners,
matching the pattern used by ``CPUGraphRunner`` which is also just a
torch.compile wrapper.

Wired up by ``ModelRunner.init_device_graphs`` when both:

* ``server_args.enable_torch_compile`` is True (checked at the call
  site, mirroring the CPU path), and
* the model does not set ``_can_torch_compile = False`` (the same
  per-model opt-out the CUDA path honors in ``kernel_warmup``).

The runner discovers ``@support_torch_compile`` metadata and delegates the
boundary-specific install / warmup work to the annotated object's compile
policy or driver. It should not know decoder- or ViT-specific padding rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.compilation.support_torch_compile import (
    get_torch_compile_policy,
)
from sglang.srt.hardware_backend.tpu._logging import get_logger

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

# This module lives outside the tpu/ tree, so __name__ does not start with
# the TPU logger root. Pass a short suffix so the logger lands cleanly under
# that root (one setLevel covers the whole backend) instead of producing a
# doubled "...tpu.sglang.srt.model_executor.tpu_graph_runner" name.
logger = get_logger("tpu_graph_runner")


def _prebuild_tpu_fp8_dense_linear_op(model: Any) -> None:
    """Register TPU FP8 dense custom ops before Dynamo traces them."""
    modules = model.modules() if callable(getattr(model, "modules", None)) else (model,)
    for module in modules:
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        if quant_method.__class__.__name__ != "Fp8LinearMethod":
            continue

        from sglang.srt.hardware_backend.tpu.fp8_linear import (
            prebuild_tpu_fp8_linear_op,
        )

        prebuild_tpu_fp8_linear_op()
        logger.info(
            "Pre-registered TPU FP8 dense linear custom op before torch.compile."
        )
        return


class TpuGraphRunner:
    """Orchestrate ``torch.compile(backend='tpu')`` annotation policies."""

    def __init__(self, model_runner: ModelRunner) -> None:
        assert (
            model_runner.server_args.enable_torch_compile
        ), "TpuGraphRunner should only be initialized when enable_torch_compile=True."

        # Side-effectful import: registers the "tpu" dynamo backend so that
        # torch.compile(backend="tpu", ...) resolves. apply_tpu_patches does
        # this in init_backend; redo it defensively in case the runner is
        # constructed on a path that skipped that hook.
        from sglang.srt.hardware_backend.tpu.patches import (
            _register_tpu_compile_backend,
        )

        self.model_runner = model_runner
        self.model = model_runner.model

        # Per-model opt-out, matching the CUDA path in
        # ModelRunner.kernel_warmup: some Transformers-backend models
        # cannot be safely compiled (e.g. dynamic RoPE scaling).
        if not getattr(self.model, "_can_torch_compile", True):
            logger.info(
                "Model %s sets _can_torch_compile=False; skipping "
                "torch.compile install on TPU.",
                type(self.model).__name__,
            )
            # Reflect the skip to downstream code that branches on this
            # flag (kernel_warmup pre-compile loops on the CUDA side;
            # any future TPU equivalent should follow the same
            # convention).
            model_runner.server_args.enable_torch_compile = False
            return

        # Raise dynamo's recompile ceiling. torch.compile builds a fresh graph
        # per distinct input shape, and the TPU backbone is hit with many: one
        # per configured prefill token bucket plus the decode batch-size
        # buckets. dynamo's default cache_size_limit=8 is far too low — under
        # fullgraph=True it trips FailOnRecompileLimitHit (killing the
        # scheduler) as soon as warmup or a multi-shape benchmark walks past
        # ~8 shapes. Match the CUDA / CPU / piecewise graph runners, which all
        # set 1024 (see cuda_graph_runner.set_torch_compile_config).
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024
        # Fail loudly on compile-limit hits to expose latency cliffs early.
        if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
            torch._dynamo.config.fail_on_recompile_limit_hit = True

        _register_tpu_compile_backend()
        _prebuild_tpu_fp8_dense_linear_op(self.model)

        modules = (
            self.model.modules()
            if callable(getattr(self.model, "modules", None))
            else (self.model,)
        )
        installed_any = False
        for module in modules:
            policy_cls = get_torch_compile_policy(module)
            if policy_cls is None:
                continue
            if policy_cls.backend not in (None, "tpu"):
                continue

            method_name = "forward"
            method = getattr(module, method_name, None)
            if method is None:
                logger.warning(
                    "TPU compile annotation %s.%s has no such method; skipping.",
                    type(module).__name__,
                    method_name,
                )
                continue

            policy = policy_cls(
                model_runner=model_runner,
                owner=module,
                method_name=method_name,
                method=method,
                label=type(module).__name__,
            )
            installed = policy.install()
            if installed:
                installed_any = True
                policy.precompile()

        if not installed_any:
            logger.warning(
                "enable_torch_compile=True but %s declares no TPU compile "
                "policies; nothing compiled.",
                type(self.model).__name__,
            )

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """Always False — no captured graph to replay.

        PJRT/libtpu caches HLO compilations internally and dispatches
        on tensor shape, so the compiled forward is reached through the
        normal ``self.model.forward(...)`` call path, not through a
        replay step. This method exists because ModelRunner.forward
        gates the replay branch on
        ``self.graph_runner and self.graph_runner.can_run(...)`` —
        returning False keeps that branch off on TPU.
        """
        return False
