"""TPU "graph runner": installs torch.compile on the model.

Unlike CudaGraphRunner / NPUGraphRunner there is no graph capture or
replay here — PJRT (libtpu) caches HLO compilations internally, keyed
by tensor shape, so we get "compile once, dispatch by shape" behavior
without an explicit capture step. This class exists so the
torch.compile install hangs off the same factory hook
(``current_platform.get_graph_runner_cls()``) as the in-tree runners,
matching the pattern used by ``CPUGraphRunner`` which is also just a
torch.compile wrapper.

Wired up by ``ModelRunner.init_device_graphs`` when both:

* ``server_args.enable_torch_compile`` is True (checked at the call
  site, mirroring the CPU path), and
* the model does not set ``_can_torch_compile = False`` (the same
  per-model opt-out the CUDA path honors in ``kernel_warmup``).

Reassigning attributes on the model instance shadows the class methods
for that instance only — every existing call site in ``ModelRunner``
(``self.model.<name>(...)``) picks up the compiled version automatically.
"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

# This module lives outside the tpu/ tree, so __name__ does not start with
# the TPU logger root. Pass a short suffix so the logger lands cleanly under
# that root (one setLevel covers the whole backend) instead of producing a
# doubled "...tpu.sglang.srt.model_executor.tpu_graph_runner" name.
logger = get_logger("tpu_graph_runner")


def _tpu_compile_layer_limit() -> int | None:
    value = os.environ.get("SGLANG_TPU_COMPILE_LAYER_LIMIT")
    if not value:
        return None
    limit = int(value)
    if limit <= 0:
        return None
    return limit


def _wrap_with_tpu_compile(model, attr_name: str) -> bool:
    """Replace ``model.<attr_name>`` with a torch.compile + no_grad wrapper.

    Returns True if the attribute existed and was wrapped, False if the
    model does not expose it (which is the common case for the optional
    ``forward_split_prefill`` entry point).
    """
    fn = getattr(model, attr_name, None)
    if fn is None:
        return False
    signature = inspect.signature(fn)
    compiled = torch.compile(
        torch.no_grad()(fn),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )
    compiled.__signature__ = signature
    setattr(model, attr_name, compiled)
    return True


def _wrap_forward_decode_with_tpu_compile(model) -> bool:
    """Compile ``forward`` only for decode; run other modes eagerly."""
    fn = getattr(model, "forward", None)
    if fn is None:
        return False
    signature = inspect.signature(fn)
    compiled = torch.compile(
        torch.no_grad()(fn),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )
    compiled.__signature__ = signature

    @torch.no_grad()
    def decode_only_forward(input_ids, positions, forward_batch, *args, **kwargs):
        if forward_batch.forward_mode.is_decode():
            return compiled(input_ids, positions, forward_batch, *args, **kwargs)
        return fn(input_ids, positions, forward_batch, *args, **kwargs)

    decode_only_forward.__signature__ = signature
    setattr(model, "forward", decode_only_forward)
    return True


def _wrap_backbone_with_tpu_compile(model, *, decode_only: bool = False) -> bool:
    """Compile the inner language-model backbone, leaving logits eager."""
    inner_model = getattr(model, "model", None)
    fn = getattr(inner_model, "forward", None)
    if fn is None:
        return False
    signature = inspect.signature(fn)
    compiled = torch.compile(
        torch.no_grad()(fn),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )
    compiled.__signature__ = signature
    if decode_only:

        @torch.no_grad()
        def decode_only_backbone_forward(
            input_ids, positions, forward_batch, *args, **kwargs
        ):
            if forward_batch.forward_mode.is_decode():
                return compiled(input_ids, positions, forward_batch, *args, **kwargs)
            return fn(input_ids, positions, forward_batch, *args, **kwargs)

        decode_only_backbone_forward.__signature__ = signature
        inner_model.forward = decode_only_backbone_forward
    else:
        inner_model.forward = compiled
    return True


def _wrap_decoder_layers_with_tpu_compile(
    model, *, decode_only: bool = False
) -> list[str]:
    """Compile decoder-layer forwards in place and return wrapped names."""
    inner_model = getattr(model, "model", None)
    layers = getattr(inner_model, "layers", None)
    if layers is None:
        return []

    layer_limit = _tpu_compile_layer_limit()
    wrapped: list[str] = []
    for idx, layer in enumerate(layers):
        if layer_limit is not None and idx >= layer_limit:
            break
        fn = getattr(layer, "forward", None)
        if fn is None:
            continue
        signature = inspect.signature(fn)
        compiled = torch.compile(
            torch.no_grad()(fn),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        compiled.__signature__ = signature
        if decode_only:

            @torch.no_grad()
            def decode_only_layer_forward(
                *args, __fn=fn, __compiled=compiled, **kwargs
            ):
                forward_batch = kwargs.get("forward_batch")
                if forward_batch is None and len(args) >= 3:
                    forward_batch = args[2]
                if forward_batch is not None and forward_batch.forward_mode.is_decode():
                    return __compiled(*args, **kwargs)
                return __fn(*args, **kwargs)

            decode_only_layer_forward.__signature__ = signature
            layer.forward = decode_only_layer_forward
        else:
            layer.forward = compiled
        wrapped.append(f"model.layers.{idx}.forward")
    return wrapped


def _extract_forward_batch(args, kwargs):
    forward_batch = kwargs.get("forward_batch")
    if forward_batch is not None:
        return forward_batch
    for arg in args:
        if hasattr(arg, "forward_mode"):
            return arg
    return None


def _wrap_decoder_submodules_with_tpu_compile(
    model, submodule_names: tuple[str, ...], *, decode_only: bool = False
) -> list[str]:
    """Compile selected decoder-layer submodule forwards in place."""
    inner_model = getattr(model, "model", None)
    layers = getattr(inner_model, "layers", None)
    if layers is None:
        return []

    layer_limit = _tpu_compile_layer_limit()
    wrapped: list[str] = []
    for idx, layer in enumerate(layers):
        if layer_limit is not None and idx >= layer_limit:
            break
        for submodule_name in submodule_names:
            submodule = getattr(layer, submodule_name, None)
            fn = getattr(submodule, "forward", None)
            if fn is None:
                continue
            signature = inspect.signature(fn)
            compiled = torch.compile(
                torch.no_grad()(fn),
                backend="tpu",
                fullgraph=True,
                dynamic=False,
            )
            compiled.__signature__ = signature
            if decode_only:

                @torch.no_grad()
                def decode_only_submodule_forward(
                    *args, __fn=fn, __compiled=compiled, **kwargs
                ):
                    forward_batch = _extract_forward_batch(args, kwargs)
                    if (
                        forward_batch is not None
                        and forward_batch.forward_mode.is_decode()
                    ):
                        return __compiled(*args, **kwargs)
                    return __fn(*args, **kwargs)

                decode_only_submodule_forward.__signature__ = signature
                submodule.forward = decode_only_submodule_forward
            else:
                submodule.forward = compiled
            wrapped.append(f"model.layers.{idx}.{submodule_name}.forward")
    return wrapped


class TpuGraphRunner:
    """Install ``torch.compile(backend='tpu')`` on the model backbone."""

    # Entry points wrapped with torch.compile. ``forward`` is hit by
    # forward_decode / forward_extend / forward_idle.
    # ``forward_split_prefill`` is hit by the split-prefill path on
    # models that expose it (qwen2/qwen3 family, glm4, gemma); it is
    # wrapped so first invocation populates the HLO cache, but
    # ``CompilationManager`` only precompiles ``forward`` today — see
    # the warning in ``CompilationManager.__init__``.
    _COMPILE_TARGETS: tuple[str, ...] = ("forward", "forward_split_prefill")

    def __init__(self, model_runner: ModelRunner):
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
        if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
        # Fail loudly on compile-limit hits to expose latency cliffs early.
        if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
            torch._dynamo.config.fail_on_recompile_limit_hit = True

        _register_tpu_compile_backend()

        compile_granularity = os.environ.get("SGLANG_TPU_COMPILE_GRANULARITY", "model")
        if compile_granularity == "decode":
            wrapped = (
                ["forward"] if _wrap_forward_decode_with_tpu_compile(self.model) else []
            )
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = f"{type(self.model).__name__} decode"
        elif compile_granularity == "backbone":
            wrapped = (
                ["model.forward"] if _wrap_backbone_with_tpu_compile(self.model) else []
            )
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = f"{type(self.model).__name__} backbone"
        elif compile_granularity == "backbone_decode":
            wrapped = (
                ["model.forward"]
                if _wrap_backbone_with_tpu_compile(self.model, decode_only=True)
                else []
            )
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = f"{type(self.model).__name__} backbone decode"
        elif compile_granularity == "layer_decode":
            wrapped = _wrap_decoder_layers_with_tpu_compile(
                self.model, decode_only=True
            )
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = "decoder layers decode"
        elif compile_granularity == "layer":
            wrapped = _wrap_decoder_layers_with_tpu_compile(self.model)
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = "decoder layers"
        elif compile_granularity == "mlp_decode":
            wrapped = _wrap_decoder_submodules_with_tpu_compile(
                self.model, ("mlp",), decode_only=True
            )
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = "decoder MLPs decode"
        elif compile_granularity == "attn_decode":
            wrapped = _wrap_decoder_submodules_with_tpu_compile(
                self.model, ("self_attn",), decode_only=True
            )
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = "decoder attention decode"
        elif compile_granularity == "attn_mlp_decode":
            wrapped = _wrap_decoder_submodules_with_tpu_compile(
                self.model, ("self_attn", "mlp"), decode_only=True
            )
            compiled_fns = {"forward": getattr(self.model, "forward")}
            log_target = "decoder attention+MLP decode"
        elif compile_granularity == "model":
            wrapped = [
                name
                for name in self._COMPILE_TARGETS
                if _wrap_with_tpu_compile(self.model, name)
            ]
            compiled_fns = {
                name: getattr(self.model, name)
                for name in wrapped
                if name in self._COMPILE_TARGETS
            }
            log_target = type(self.model).__name__
        else:
            raise ValueError(
                "SGLANG_TPU_COMPILE_GRANULARITY must be 'model', 'layer', "
                "'decode', 'backbone', 'backbone_decode', 'layer_decode', "
                "'mlp_decode', 'attn_decode', or 'attn_mlp_decode', "
                f"got {compile_granularity!r}."
            )

        if not wrapped:
            logger.warning(
                "enable_torch_compile=True but %s exposes none of %s; "
                "nothing compiled.",
                type(self.model).__name__,
                self._COMPILE_TARGETS,
            )
            return

        logger.info(
            "Wrapped %s {%s} with torch.compile(backend='tpu', "
            "fullgraph=True, dynamic=False) inside torch.no_grad "
            "(granularity=%s).",
            log_target,
            ", ".join(wrapped),
            compile_granularity,
        )

        from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
            TpuCompiledForwardShim,
        )

        compiled_forward_shim = TpuCompiledForwardShim(
            model_runner, compiled_fns=compiled_fns
        ).install()
        model_runner.compiled_forward_shim = compiled_forward_shim

        from sglang.srt.hardware_backend.tpu.compilation.compilation_manager import (
            CompilationManager,
        )

        CompilationManager(
            model_runner=model_runner,
            compiled_fns=compiled_fns,
            t_bucketer=compiled_forward_shim.t_bucketer,
            b_bucketer=compiled_forward_shim.b_bucketer,
        ).precompile_all()

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
