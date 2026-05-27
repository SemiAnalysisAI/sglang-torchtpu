"""TPU graph compilation manager.

Provides piecewise compilation with bucket padding to support dynamic shapes.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import torch
import torch._dynamo

from sglang.srt.hardware_backend.tpu import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketizer import (
    PromptBucketizer,
)

logger = get_logger(__name__)

_dynamo_configured = False

def _configure_dynamo() -> None:
    """Configure torch._dynamo for TPU compile mode."""
    global _dynamo_configured
    if _dynamo_configured:
        return
    _dynamo_configured = True

    import torch._logging as _torch_logging
    cfg = torch._dynamo.config
    cfg.suppress_errors = False
    if hasattr(cfg, "cache_size_limit"):
        cfg.cache_size_limit = max(cfg.cache_size_limit, 1024)
    try:
        _torch_logging.set_logs(recompiles=True)
    except Exception:
        pass


def _compile_for_tpu(
    fn_or_module: Union[Callable[..., Any], torch.nn.Module],
) -> Any:
    """Wrap `fn_or_module` with `torch.compile(backend='tpu')`."""
    # TODO(wyzhang): Set `fullgraph=True` when ready.
    compiled = torch.compile(fn_or_module, backend="tpu", dynamic=False)
    logger.info(
        "wrapping %s",
        getattr(fn_or_module, "__name__", type(fn_or_module).__name__),
    )
    return compiled


class _PaddedLayer(torch.nn.Module):
    """Pads layer inputs to the next bucket size on entry, slices back
    on exit. `nn.Module` (not function) because `ModuleList` rejects
    non-Modules."""

    _tpu_input_padded = True

    def __init__(self, compiled_layer):
        super().__init__()
        # Plain attr-write (not submodule) so state_dict / lora scans
        # don't double-count the compiled child's params.
        object.__setattr__(self, "_compiled", compiled_layer)

    def forward(self, positions, hidden_states, forward_batch,
                residual=None, *args, **kwargs):
        n_real = hidden_states.shape[0]
        n_bucket = PromptBucketizer.get_instance().next_compile_bucket(n_real)
        assert n_bucket >= n_real, (
            f"next_compile_bucket returned {n_bucket} < n_real={n_real}"
        )

        if n_bucket > n_real:
            n_pad = n_bucket - n_real
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, n_pad),
            )
            positions = torch.nn.functional.pad(positions, (0, n_pad))
            if residual is not None:
                residual = torch.nn.functional.pad(
                    residual, (0, 0, 0, n_pad),
                )

        out = self._compiled(
            positions, hidden_states, forward_batch, residual,
            *args, **kwargs,
        )
        return _slice_back_to_real_len(out, n_bucket, n_real)


def _slice_back_to_real_len(out, n_bucket: int, n_real: int):
    """Slice padded outputs back to `n_real`."""
    if n_bucket == n_real:
        return out

    def _slice(t):
        if isinstance(t, torch.Tensor) and t.shape[0] == n_bucket:
            return t[:n_real]
        return t

    if isinstance(out, tuple):
        return tuple(_slice(x) for x in out)
    return _slice(out)


def _find_inner_model(model) -> "torch.nn.Module":
    """Inner backbone holding `embed_tokens`, `layers`, `norm`."""
    inner = model
    if hasattr(inner, "model") and isinstance(inner.model, torch.nn.Module):
        inner = inner.model
    if isinstance(inner, torch.nn.Module):
        return inner
    raise RuntimeError(
        f"could not locate inner backbone on {type(model).__name__} "
        f"(expected .model: nn.Module)"
    )


def _find_transformer_layers(model) -> "torch.nn.ModuleList":
    """The transformer-layer `ModuleList` we wrap per-layer."""
    inner = _find_inner_model(model)
    if (
        hasattr(inner, "layers")
        and isinstance(inner.layers, torch.nn.ModuleList)
        and len(inner.layers) > 0
    ):
        return inner.layers
    raise RuntimeError(
        f"could not locate transformer-layer ModuleList on "
        f"{type(model).__name__} (expected .model.layers: ModuleList)"
    )


def _apply_inner_padding(inner_model) -> None:
    """Pad `input_ids` + `positions`"""
    model_cls = type(inner_model)
    if getattr(model_cls, "_tpu_input_padded", False):
        return
    orig_forward = model_cls.forward

    def wrapped_forward(self, input_ids, positions, forward_batch,
                        *args, **kwargs):
        n_real = input_ids.shape[0]
        n_bucket = PromptBucketizer.get_instance().next_compile_bucket(n_real)
        assert n_bucket >= n_real, (
            f"next_compile_bucket returned {n_bucket} < n_real={n_real}"
        )

        if n_bucket > n_real:
            n_pad = n_bucket - n_real
            input_ids = torch.nn.functional.pad(input_ids, (0, n_pad))
            positions = torch.nn.functional.pad(positions, (0, n_pad))

        out = orig_forward(
            self, input_ids, positions, forward_batch, *args, **kwargs,
        )
        return _slice_back_to_real_len(out, n_bucket, n_real)

    model_cls.forward = wrapped_forward
    model_cls._tpu_input_padded = True
    logger.info(
        "wrapped %s.forward with bucket-padding (input_ids + positions)",
        model_cls.__name__,
    )


def _compile_module_attr(model, attr_name: str) -> None:
    """Wrap an inner-backbone submodule (e.g. `embed_tokens`, `norm`)"""
    inner = _find_inner_model(model)
    target = getattr(inner, attr_name, None)
    if target is None or not isinstance(target, torch.nn.Module):
        raise RuntimeError(
            f"inner backbone {type(inner).__name__} has no compilable "
            f"attribute {attr_name!r}"
        )
    if getattr(target, "_torchdynamo_orig_callable", None) is not None:
        raise RuntimeError(
            f"TPU compile wrap: `{attr_name}` on {type(inner).__name__} "
            f"is already torch.compile-wrapped. Unexpected pre-state — "
            f"refusing to double-wrap."
        )
    setattr(inner, attr_name, _compile_for_tpu(target))


_compile_wrap_installed = False


def _install_compile_wrap() -> None:
    """Install the compile wrap on the ModelRunner.

    Deferred to first request via `Scheduler.handle_generate_request`
    because `apply_tpu_patches` runs before `ModelRunner` is
    constructed.
    """
    global _compile_wrap_installed
    if _compile_wrap_installed:
        return

    from sglang.srt.managers.scheduler import Scheduler

    orig_handle = Scheduler.handle_generate_request

    installed = False

    def wrapped_handle(self, recv_req, *args, **kwargs):
        nonlocal installed
        if installed:
            return orig_handle(self, recv_req, *args, **kwargs)

        worker = (
            getattr(self, "model_worker", None)
            or getattr(self, "tp_worker", None)
        )
        if worker is None:
            raise RuntimeError(
                "TPU compile wrap: Scheduler has no `model_worker` "
                "or `tp_worker`. Sglang internals may have changed."
            )
        runner = getattr(worker, "_model_runner", None)
        if runner is None:
            raise RuntimeError(
                "TPU compile wrap: worker has no `_model_runner`. "
                "Sglang internals may have changed."
            )
        if not hasattr(runner, "model") or not isinstance(
            runner.model, torch.nn.Module
        ):
            raise RuntimeError(
                "TPU compile wrap: model_runner has no `.model` "
                "nn.Module attribute."
            )

        PromptBucketizer.initialize(runner.model_config.context_len)
        layers = _find_transformer_layers(runner.model)
        for i in range(len(layers)):
            if getattr(
                layers[i], "_torchdynamo_orig_callable", None
            ) is not None or getattr(
                layers[i], "_tpu_input_padded", False
            ):
                raise RuntimeError(
                    f"TPU compile wrap: layer {i} is already wrapped "
                    f"(_torchdynamo_orig_callable or _tpu_input_padded "
                    f"set). Unexpected pre-state — refusing to double-wrap."
                )
            layers[i] = _PaddedLayer(_compile_for_tpu(layers[i]))
        _compile_module_attr(runner.model, "embed_tokens")
        _compile_module_attr(runner.model, "norm")
        _apply_inner_padding(_find_inner_model(runner.model))
        installed = True

        return orig_handle(self, recv_req, *args, **kwargs)

    Scheduler.handle_generate_request = wrapped_handle
    _compile_wrap_installed = True
    logger.info("compile wrap will install on first generate request")


def maybe_apply_compilation_patches() -> None:
    """Compile-mode-gated TPU patches. No-op unless
    `TPU_COMPILATION_MODE` is set."""
    if not envs.TPU_COMPILATION_MODE:
        return
    _configure_dynamo()
    _install_compile_wrap()
