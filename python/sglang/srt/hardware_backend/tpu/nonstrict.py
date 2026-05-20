# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Adapted from the prototype nonstrict JIT API in PyTorch PR 183726.

"""Nonstrict JIT helpers for the SGLang TorchTPU backend.

This module intentionally lives in ``sglang-torchtpu`` rather than
``torch_tpu``.  It gives the TPU backend a JAX-like API for compiling functions
whose dynamic inputs and outputs are tensors or pytrees of tensors, while
treating named Python arguments as static compile-time constants.

The wrapper traces with ``make_fx`` and hands the resulting FX graph to a normal
Dynamo backend.  The default backend is ``"tpu"``, so this can call the
TorchTPU ``torch.compile`` backend without relying on Dynamo bytecode capture of
the original Python function.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import contextlib
import dataclasses
import functools
import inspect
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx as torch_make_fx


def _static_names(names: str | Iterable[str] | None) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def _backend_fn(backend: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(backend):
        return backend
    if backend == "tpu":
        # Import for side effects: registers the "tpu" Dynamo backend.
        import torch_tpu  # noqa: F401

    from torch._dynamo.backends.registry import lookup_backend

    return lookup_backend(backend)


@functools.lru_cache(maxsize=1)
def _tensor_metadata_fn() -> Callable[[torch.Tensor], Any]:
    try:
        from torch._inductor.codecache import extract_tensor_metadata_for_cache_key

        return extract_tensor_metadata_for_cache_key
    except Exception:
        from torch._dynamo.guards import extract_tensor_metadata

        return extract_tensor_metadata


def _freeze(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (
            type(x),
            tuple(_freeze(getattr(x, f.name)) for f in dataclasses.fields(x)),
        )
    if isinstance(x, (tuple, list, torch.Size)):
        return tuple(_freeze(v) for v in x)
    return x


def _tensor_key(x: torch.Tensor) -> Any:
    return type(x), _freeze(_tensor_metadata_fn()(x))


def _check_static(name: str, value: Any) -> Any:
    if any(isinstance(leaf, torch.Tensor) for leaf in pytree.tree_leaves(value)):
        raise TypeError(
            f"static argument {name!r} must not contain Tensors; leave Tensor "
            "arguments dynamic so tensor metadata is used for caching"
        )
    try:
        hash(value)
    except TypeError as e:
        raise TypeError(f"static argument {name!r} must be hashable") from e
    return value


def _check_tensor_leaves(values: list[Any], what: str) -> None:
    for i, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{what} must be Tensors or pytrees of Tensors; found leaf {i} "
                f"with type {type(value).__name__}"
            )


def _call(
    fn: Callable[..., Any], sig: inspect.Signature, values: dict[str, Any]
) -> Any:
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name not in values:
            continue
        value = values[name]
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(value)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs.update(value)
    return fn(*args, **kwargs)


def _flat_outputs(out: Any) -> tuple[tuple[Any, ...], pytree.TreeSpec]:
    flat_out, out_spec = pytree.tree_flatten(out)
    _check_tensor_leaves(flat_out, "outputs")
    return tuple(flat_out), out_spec


class _TraceableCollectiveRemap(torch.overrides.TorchFunctionMode):
    """Apply PyTorch's traceable collective remaps during make_fx tracing."""

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if torch.compiler.is_dynamo_compiling():
            return func(*args, **kwargs)

        if torch.distributed.is_available():
            from torch.distributed._functional_collectives import (
                REDUCE_OP_TO_STR,
                traceable_collective_remaps,
            )

            if func in traceable_collective_remaps:
                mapped_func = traceable_collective_remaps[func]
                signature = inspect.signature(func)
                kwargs = dict(signature.bind(*args, **kwargs).arguments)
                args = ()

                reduce_collectives = (
                    torch.distributed.all_reduce,
                    torch.distributed.reduce_scatter_tensor,
                    torch.distributed._reduce_scatter_base,
                )
                if func in reduce_collectives and "op" in kwargs:
                    kwargs["op"] = REDUCE_OP_TO_STR[kwargs["op"]]
                return mapped_func(*args, **kwargs)

        return func(*args, **kwargs)


@contextlib.contextmanager
def traceable_collective_remap():
    """Remap c10d collectives to traceable functional collectives.

    This is useful for callers that trace functions containing tensor-parallel
    collectives with ``make_fx`` directly. It is deliberately opt-in so
    ``jit`` and ``make_fx`` do not rewrite collective calls for users that do
    not need TPU TP graph capture.
    """
    with _TraceableCollectiveRemap():
        yield


def _bind_dynamic(
    sig: inspect.Signature,
    static_order: tuple[str, ...],
    dynamic_order: tuple[str, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], tuple[Any, ...], list[Any], pytree.TreeSpec]:
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    for name in static_order:
        _check_static(name, values[name])
    dynamic_values = tuple(values[name] for name in dynamic_order)
    flat_args, in_spec = pytree.tree_flatten(dynamic_values)
    _check_tensor_leaves(flat_args, "dynamic arguments")
    return values, tuple(values[name] for name in static_order), flat_args, in_spec


def _trace(
    fn: Callable[..., Any],
    sig: inspect.Signature,
    values: dict[str, Any],
    dynamic_order: tuple[str, ...],
    flat_args: list[Any],
    in_spec: pytree.TreeSpec,
    tracing_mode: str,
) -> tuple[torch.fx.GraphModule, pytree.TreeSpec]:
    out_spec_holder: dict[str, pytree.TreeSpec] = {}

    def flat_fn(*new_flat_args: Any) -> tuple[Any, ...]:
        new_values = dict(values)
        new_values.update(
            zip(
                dynamic_order,
                pytree.tree_unflatten(new_flat_args, in_spec),
                strict=True,
            )
        )
        flat_out, out_spec = _flat_outputs(_call(fn, sig, new_values))
        out_spec_holder["spec"] = out_spec
        return flat_out

    gm = torch_make_fx(flat_fn, tracing_mode=tracing_mode)(*flat_args)
    gm._nonstrict_jit_in_spec = in_spec
    gm._nonstrict_jit_out_spec = out_spec_holder["spec"]
    return gm, out_spec_holder["spec"]


def make_fx(
    fn: Callable[..., Any],
    *,
    static_argnames: str | Iterable[str] | None = None,
    tracing_mode: str = "fake",
) -> Callable[..., torch.fx.GraphModule]:
    """Trace ``fn`` with ``make_fx`` and named static arguments."""
    sig = inspect.signature(fn)
    static_names = frozenset(_static_names(static_argnames))
    unknown = static_names.difference(sig.parameters)
    if unknown:
        raise ValueError(f"unknown static_argnames: {sorted(unknown)}")

    static_order = tuple(name for name in sig.parameters if name in static_names)
    dynamic_order = tuple(name for name in sig.parameters if name not in static_names)

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> torch.fx.GraphModule:
        values, _, flat_args, in_spec = _bind_dynamic(
            sig, static_order, dynamic_order, args, kwargs
        )
        gm, _ = _trace(
            fn,
            sig,
            values,
            dynamic_order,
            flat_args,
            in_spec,
            tracing_mode,
        )
        return gm

    return wrapped


def _call_flat_artifact(
    artifact: Callable[..., Any],
    out_spec: pytree.TreeSpec,
    flat_args: list[Any],
) -> Any:
    flat_out = artifact(*flat_args)
    if out_spec.num_leaves == 1 and not isinstance(flat_out, (tuple, list)):
        flat_out = (flat_out,)
    return pytree.tree_unflatten(flat_out, out_spec)


class _NonstrictCompiler:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        static_argnames: str | Iterable[str] | None,
        backend: str | Callable[..., Any],
        tracing_mode: str,
    ) -> None:
        self.fn = fn
        self.sig = inspect.signature(fn)
        self.backend = backend
        self.tracing_mode = tracing_mode

        static_names = frozenset(_static_names(static_argnames))
        unknown = static_names.difference(self.sig.parameters)
        if unknown:
            raise ValueError(f"unknown static_argnames: {sorted(unknown)}")

        self.static_order = tuple(
            name for name in self.sig.parameters if name in static_names
        )
        self.dynamic_order = tuple(
            name for name in self.sig.parameters if name not in static_names
        )
        self.compile_backend = _backend_fn(backend)
        self.cache: dict[Any, tuple[Callable[..., Any], pytree.TreeSpec]] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        values, static_values, flat_args, in_spec = _bind_dynamic(
            self.sig, self.static_order, self.dynamic_order, args, kwargs
        )

        key = in_spec, static_values, tuple(_tensor_key(x) for x in flat_args)
        entry = self.cache.get(key)
        if entry is None:
            gm, out_spec = _trace(
                self.fn,
                self.sig,
                values,
                self.dynamic_order,
                flat_args,
                in_spec,
                self.tracing_mode,
            )
            entry = self.compile_backend(gm, list(flat_args)), out_spec
            self.cache[key] = entry

        compiled, out_spec = entry
        return _call_flat_artifact(compiled, out_spec, flat_args)

    def cache_size(self) -> int:
        return len(self.cache)

    def cache_clear(self) -> None:
        self.cache.clear()


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    static_argnames: str | Iterable[str] | None = None,
    backend: str | Callable[..., Any] = "tpu",
    tracing_mode: str = "real",
) -> Callable[..., Any]:
    """JIT compile a tensor-pytree function through a Dynamo backend.

    Non-static arguments must be tensors or pytrees of tensors. Static
    arguments are captured as hashable compile-time constants. The default
    backend is ``"tpu"`` for SGLang's TorchTPU platform.
    """
    if fn is None:
        return functools.partial(
            jit,
            static_argnames=static_argnames,
            backend=backend,
            tracing_mode=tracing_mode,
        )

    compiler = _NonstrictCompiler(
        fn,
        static_argnames=static_argnames,
        backend=backend,
        tracing_mode=tracing_mode,
    )

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return compiler(*args, **kwargs)

    wrapped.cache_size = compiler.cache_size  # type: ignore[attr-defined]
    wrapped.cache_clear = compiler.cache_clear  # type: ignore[attr-defined]
    return wrapped


__all__ = ["jit", "make_fx", "traceable_collective_remap"]
