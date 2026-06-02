"""A static shape shim that pads inputs and unpads outputs to maintain a fixed
set of compilation shapes.
"""

from __future__ import annotations

import abc
import functools
import inspect
from typing import Any

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketScheme,
)

logger = get_logger(__name__)

# Registry mapping model class names to custom shim rules hooks
_MODEL_SHIM_RULES: dict[str, callable] = {}


def register_model_shim_rules(model_class_name: str):
    """Decorator to register custom static shape shim rules.

    Args:
        model_class_name: The name of the target model class.
    """
    def decorator(func):
        _MODEL_SHIM_RULES[model_class_name] = func
        return func
    return decorator


# --------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------- #

def _has_nested_attr(obj: Any, path: str) -> bool:
    """Checks if a dot-separated path of attributes exists on the object."""
    parts = path.split(".")
    curr = obj
    for part in parts:
        if curr is None:
            return True
        if not hasattr(curr, part):
            return False
        curr = getattr(curr, part)
    return True


def _resolve_and_validate_tensor(
    bound: inspect.BoundArguments, path: str
) -> tuple[torch.Tensor | None, str | None, str | None]:
    """Resolves and validates a registered tensor path from bound arguments.

    Args:
        bound: The bound arguments of the function call.
        path: The dot-separated path to the target tensor.

    Returns:
        A tuple of (tensor, root_param, nested_path) if found and valid,
        otherwise (None, None, None).
    """
    parts = path.split(".", 1)
    root_param = parts[0]
    if root_param not in bound.arguments:
        return None, None, None

    val = bound.arguments[root_param]
    if len(parts) > 1:
        nested_path = parts[1]
        if not _has_nested_attr(val, nested_path):
            raise AttributeError(
                f"Registered field path '{path}' does not exist on "
                f"parameter '{root_param}'."
            )
        tensor = _get_nested_attr(val, nested_path)
    else:
        nested_path = None
        tensor = val

    if tensor is None:
        return None, None, None

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Registered path '{path}' value is not a torch.Tensor at "
            f"runtime. Got type: {type(tensor)}"
        )
    if tensor.dim() < 1:
        raise ValueError(
            f"Registered path '{path}' tensor has dimension "
            f"{tensor.dim()} < 1. Cannot apply padding on a scalar."
        )

    return tensor, root_param, nested_path


def _get_nested_attr(obj: Any, path: str) -> Any | None:
    """Traverses a dot-separated path string to retrieve a nested attribute.

    Args:
        obj: The parent object containing the attribute hierarchy.
        path: A dot-separated string of attribute names
            (e.g., "parent.child.value").

    Returns:
        The nested attribute value if found and not None, otherwise None.
    """
    parts = path.split(".")
    curr = obj
    for part in parts:
        curr = getattr(curr, part, None)
        if curr is None:
            return None
    return curr


def _set_nested_attr(obj: Any, path: str, value: Any) -> bool:
    """Traverses a dot-separated path string to set a nested attribute.

    Args:
        obj: The parent object containing the attribute hierarchy.
        path: A dot-separated string of attribute names
            (e.g., "parent.child.value").
        value: The value to set on the final nested attribute.

    Returns:
        True if the attribute was successfully set, False if any
        intermediate segment along the path was None.
    """
    parts = path.split(".")
    curr = obj
    for part in parts[:-1]:
        curr = getattr(curr, part, None)
        if curr is None:
            return False
    setattr(curr, parts[-1], value)
    return True


def _is_decode_mode(bound: inspect.BoundArguments) -> bool:
    """Detects if the current model execution mode is decode."""
    forward_batch = bound.arguments.get("forward_batch")
    if forward_batch is None:
        logger.warning(
            "Unable to detect execution mode: 'forward_batch' "
            "parameter not found. Defaulting to extend mode."
        )
        return False

    if getattr(forward_batch, "forward_mode", None) is None:
        logger.warning(
            "Unable to detect execution mode: 'forward_batch' "
            "has no 'forward_mode' attribute. Defaulting to extend mode."
        )
        return False

    return forward_batch.forward_mode.is_decode()


# --------------------------------------------------------------------- #
# Padding Strategy Definitions
# --------------------------------------------------------------------- #

class PadStrategy(abc.ABC):
    """Padding strategy interface for input tensors."""

    def _normalize_dim(self, tensor: torch.Tensor, dim: int) -> int:
        """Normalizes negative dimensions and validates them against tensor shape."""
        if dim < 0:
            dim += tensor.dim()
        assert 0 <= dim < tensor.dim(), (
            f"Dimension index {dim} is out of bounds for tensor of dimension "
            f"{tensor.dim()}."
        )
        return dim

    def matches_shape(
        self, tensor: torch.Tensor, dim: int, real_size: int
    ) -> bool:
        """Determines if the strategy matches this tensor's target dimension size.

        Args:
            tensor: The target tensor to check.
            dim: The target physical dimension index of the tensor.
            real_size: The active runtime size before padding.

        Returns:
            True if the strategy matches the tensor shape, False otherwise.
        """
        dim = self._normalize_dim(tensor, dim)
        return tensor.shape[dim] == real_size

    @abc.abstractmethod
    def pad(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        n_pad: int = 0,
        real_size: int | None = None,
    ) -> torch.Tensor:
        """Pads the target dimension of the tensor.

        Args:
            tensor: The target tensor to pad.
            dim: The target physical dimension index of the tensor.
            n_pad: The number of elements to pad along the target dimension.
            real_size: Optional active runtime size before padding.

        Returns:
            A new tensor padded along the target dimension according to the strategy.
        """
        pass


class UnpadStrategy(abc.ABC):
    """Strategy interface for unpadding/slicing output tensors."""

    @abc.abstractmethod
    def unpad(
        self, tensor: torch.Tensor, dim: int, real_size: int
    ) -> torch.Tensor:
        """Slices or reconstructs the original unpadded tensor along target dim.

        Args:
            tensor: The target padded tensor to slice.
            dim: The physical dimension index to slice along.
            real_size: The original dynamic size before padding.

        Returns:
            A sliced tensor of original dynamic shape along target dimension.
        """
        pass


class SliceStrategy(UnpadStrategy):
    """Default unpadding strategy that slices output down to real size."""

    def unpad(
        self, tensor: torch.Tensor, dim: int, real_size: int
    ) -> torch.Tensor:
        return tensor.narrow(dim, 0, real_size)


# Single reusable default instance of stateless unpadding strategy
_DEFAULT_UNPAD_STRATEGY = SliceStrategy()


class ZeroPadStrategy(PadStrategy):
    """Pads the target dimension of the tensor with zero values.

    Standard strategy for token-axis fields, flat metrics, or general indices where
    additional values are safely ignored or masked out in downstream computation.
    """

    def pad(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        n_pad: int = 0,
        real_size: int | None = None,
    ) -> torch.Tensor:
        if n_pad <= 0:
            return tensor
        dim = self._normalize_dim(tensor, dim)
        # Construct padding specification [left, right] pairs for each dim.
        pads = [[0, 0] for _ in range(tensor.dim())]
        pads[dim] = [0, n_pad]
        pads.reverse()
        pad_spec = [p for pad_pair in pads for p in pad_pair]
        return torch.nn.functional.pad(tensor, pad_spec, value=0.0)


class LastValueRepeatStrategy(PadStrategy):
    """Pads the target dimension by repeating the final element."""

    def matches_shape(
        self, tensor: torch.Tensor, dim: int, real_size: int
    ) -> bool:
        dim = self._normalize_dim(tensor, dim)
        return tensor.shape[dim] == real_size or tensor.shape[dim] == real_size + 1

    def pad(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        n_pad: int = 0,
        real_size: int | None = None,
    ) -> torch.Tensor:
        if n_pad <= 0:
            return tensor
        dim = self._normalize_dim(tensor, dim)
        # Slice the last element along the target dimension
        idx = [slice(None)] * tensor.dim()
        idx[dim] = slice(-1, None)
        last = tensor[tuple(idx)]

        # Expand the last element slice to fill the padding buffer along target dim
        expand_shape = list(tensor.shape)
        expand_shape[dim] = n_pad
        last_expanded = last.expand(expand_shape).contiguous()

        return torch.cat([tensor, last_expanded], dim=dim)


class LinearStridePadStrategy(PadStrategy):
    """Pads cumulative 1-D tensor assuming linear layout:
    ``tensor[i] = i * stride``.

    Attention backend metadata fields (such as cu_kv_lens) use this to extend
    strides for padded batch regions while avoiding device syncs.
    """

    def matches_shape(
        self, tensor: torch.Tensor, dim: int, real_size: int
    ) -> bool:
        dim = self._normalize_dim(tensor, dim)
        return tensor.shape[dim] == real_size + 1

    def pad(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        n_pad: int = 0,
        real_size: int | None = None,
    ) -> torch.Tensor:
        if n_pad <= 0:
            return tensor
        dim = self._normalize_dim(tensor, dim)
        b_real = tensor.shape[dim] - 1
        if b_real < 1:
            return tensor
        # Determine stride from the first non-zero element
        stride = int(tensor[1].item())
        device = tensor.device
        dtype = tensor.dtype
        # Generate tail: (b_real + 1 ... b_real + n_pad) * stride
        new_tail = (
            torch.arange(
                b_real + 1,
                b_real + n_pad + 1,
                device=device,
                dtype=dtype,
            )
            * stride
        )
        return torch.cat([tensor, new_tail], dim=dim)


class ScaledZeroPadStrategy(PadStrategy):
    """Pads 1-D tensor whose length is a multiple of B to B_bucket * multiple.

    Designed for fields (such as page_indices) that scale with dynamic multiples
    of B, ensuring consistent alignment of padded sub-blocks.
    """

    def matches_shape(
        self, tensor: torch.Tensor, dim: int, real_size: int
    ) -> bool:
        dim = self._normalize_dim(tensor, dim)
        return real_size >= 1 and tensor.shape[dim] % real_size == 0

    def pad(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        n_pad: int = 0,
        real_size: int | None = None,
    ) -> torch.Tensor:
        if n_pad <= 0 or not real_size or real_size < 1:
            return tensor
        dim = self._normalize_dim(tensor, dim)
        multiple = tensor.shape[dim] // real_size
        tail_n = n_pad * multiple
        pads = [[0, 0] for _ in range(tensor.dim())]
        pads[dim] = [0, tail_n]
        pads.reverse()
        pad_spec = [p for pad_pair in pads for p in pad_pair]
        return torch.nn.functional.pad(tensor, pad_spec, value=0.0)


# --------------------------------------------------------------------- #
# Shim Rules
# --------------------------------------------------------------------- #

class ShimRules:
    """Declarative registry managing input-padding and output-unpadding rules."""

    def __init__(self):
        self.input_rules: dict[
            tuple[str | None, str], tuple[int, Bucketer, PadStrategy]
        ] = {}
        self.output_rules: dict[
            tuple[str | None, str], tuple[int, Bucketer, UnpadStrategy]
        ] = {}
        # Set of dynamic entry points/targets mapped to their wrapper class
        self.wrapped_target_fn: dict[str, type] = {}
        self.t_bucketer: Bucketer | None = None
        self.b_bucketer: Bucketer | None = None

    def register_target_fn(
        self, target_fn_name: str, wrapper_cls: type | None = None
    ) -> ShimRules:
        """Registers a model compiled entry point method name to wrap.

        Args:
            target_fn_name: The name of the method on the model instance.
            wrapper_cls: The wrapper class to instantiate for this target.

        Returns:
            The ShimRules instance for fluent chaining.
        """
        if wrapper_cls is None:
            wrapper_cls = ShimWrapper
        self.wrapped_target_fn[target_fn_name] = wrapper_cls
        return self

    def register_input_field(
        self,
        target_fn: str | None,
        path: str,
        dim: int,
        bucketer: Bucketer,
        pad_strategy: PadStrategy,
    ) -> ShimRules:
        """Binds nested field dot-path to padding strategy and active bucketer.

        Args:
            target_fn: Target function name scope (e.g., "forward").
            path: A dot-separated path string (e.g., "seq_lens").
            dim: The physical dimension index of the tensor to apply padding on.
            bucketer: The Bucketer instance coordinating padding for this field.
            pad_strategy: The stateless padding strategy to execute for this path.

        Returns:
            The ShimRules instance for builder-pattern chaining.
        """
        self.input_rules[(target_fn, path)] = (dim, bucketer, pad_strategy)
        return self

    def register_output_field(
        self,
        target_fn: str | None,
        path: str,
        dim: int,
        bucketer: Bucketer,
        unpad_strategy: UnpadStrategy | None = None,
    ) -> ShimRules:
        """Registers dot-path representing returned output tensor to apply
        dynamic slicing on.

        Args:
            target_fn: Target function name scope (e.g., "forward").
            path: dot-separated path string of output tensor (e.g. "hidden_states").
            dim: The physical dimension index of the returned tensor to slice back.
            bucketer: The Bucketer instance that coordinates output shapes.
            unpad_strategy: Unpadding strategy (defaults to SliceStrategy).

        Returns:
            The ShimRules instance for builder-pattern chaining.
        """
        if unpad_strategy is None:
            unpad_strategy = _DEFAULT_UNPAD_STRATEGY
        self.output_rules[(target_fn, path)] = (dim, bucketer, unpad_strategy)
        return self


# --------------------------------------------------------------------- #
# Shim Wrapper
# --------------------------------------------------------------------- #

class ShimWrapper:
    """A callable wrapper that intercepts compiled model entry points to
    inject padding.

    Shadows targeted model methods to coordinate pre-compilation input padding
    and post-compilation output slicing. Performs inline shape verification
    and bookkeeping to ensure all inputs sharing a Bucketer have consistent
    physical dimensions.
    """

    def __init__(self, fn: Any, shim: StaticShapeShim):
        """Initializes wrapper, updating metadata to preserve original attrs.

        Args:
            fn: The original callable (e.g. compiled model forward method).
            shim: The StaticShapeShim orchestrating padding logic.
        """
        self.fn = fn
        self.shim = shim
        functools.update_wrapper(self, fn)
        # Cache signature once.
        self._sig = inspect.signature(fn)

    def __call__(self, *args, **kwargs) -> Any:
        """Intercepts method call, pads inputs, and unpads outputs.

        Performs single-pass bookkeeping and verification to resolve logical
        base sizes and check shape consistency before applying padding.
        """
        shim = self.shim
        target_fn_name = self.fn.__name__

        rules = []
        for (scope, path), rule in shim.rules.input_rules.items():
            if scope is None or scope == target_fn_name:
                rules.append((path, rule))
        if not rules:
            return self.fn(*args, **kwargs)

        bound = self._sig.bind(*args, **kwargs)
        # Original unpadded tensor to restore post-execution
        saved_tensors: dict[str, torch.Tensor] = {}
        # Maps each Bucketer to a tuple of (real_size, padded_size)
        resolved_sizes: dict[Bucketer, tuple[int, int]] = {}

        # Detect execution mode
        is_decode = _is_decode_mode(bound)

        t_bucketer = shim.rules.t_bucketer
        b_bucketer = shim.rules.b_bucketer

        for path, (dim, bucketer, strategy) in rules:
            # In decode, sequence query length is 1 (so T equals B).
            # Collapse token bucketer to batch bucketer.
            active_bucketer = bucketer
            if (
                is_decode
                and t_bucketer is not None
                and bucketer is t_bucketer
            ):
                active_bucketer = b_bucketer

            tensor, root_param, nested_path = (
                _resolve_and_validate_tensor(bound, path)
            )
            # Skip optional fields that resolve to None at runtime.
            if tensor is None:
                continue

            # Save real size on first encounter
            if active_bucketer not in resolved_sizes:
                if isinstance(strategy, ZeroPadStrategy):
                    real_size = tensor.shape[dim]
                elif isinstance(strategy, LinearStridePadStrategy):
                    real_size = max(1, tensor.shape[dim] - 1)
                else:
                    real_size = tensor.shape[dim]

                padded_size = active_bucketer.next_bucket(real_size)
                resolved_sizes[active_bucketer] = (real_size, padded_size)
            else:
                real_size, padded_size = resolved_sizes[active_bucketer]

            # Verify consistency
            if not strategy.matches_shape(tensor, dim, real_size):
                raise ValueError(
                    f"Inconsistent shape for bucketer "
                    f"'{active_bucketer.name}' "
                    f"(target_fn={target_fn_name}). "
                    f"Resolved real size: {real_size}, "
                    f"but field '{path}' along dimension {dim} "
                    f"has shape {tensor.shape[dim]}."
                )

            # Apply padding
            n_pad = padded_size - real_size
            if n_pad > 0:
                saved_tensors[path] = tensor
                padded_tensor = strategy.pad(tensor, dim, n_pad, real_size)
                if nested_path is not None:
                    _set_nested_attr(
                        bound.arguments[root_param],
                        nested_path,
                        padded_tensor,
                    )
                else:
                    bound.arguments[root_param] = padded_tensor

        try:
            out = self.fn(*bound.args, **bound.kwargs)
        finally:
            # Restore original tensors
            for path, orig_val in saved_tensors.items():
                parts = path.split(".", 1)
                root_param = parts[0]
                if len(parts) > 1:
                    nested_path = parts[1]
                    parent_obj = bound.arguments[root_param]
                    _set_nested_attr(parent_obj, nested_path, orig_val)
                else:
                    bound.arguments[root_param] = orig_val

        # Apply output slicing rules
        output_rules = []
        for (scope, path), rule in shim.rules.output_rules.items():
            if scope is None or scope == target_fn_name:
                output_rules.append((path, rule))

        for path, (dim, bucketer, strategy) in output_rules:
            # In decode, sequence query length is 1 (so T equals B).
            # Collapse token bucketer to batch bucketer.
            active_bucketer = bucketer
            if (
                is_decode
                and t_bucketer is not None
                and bucketer is t_bucketer
            ):
                active_bucketer = b_bucketer

            if active_bucketer not in resolved_sizes:
                continue

            real_size, padded_size = resolved_sizes[active_bucketer]
            if padded_size == real_size:
                continue

            # Empty path means the return value is itself the target tensor.
            # Otherwise, resolve and slice the nested attribute.
            if not path:
                if (
                    isinstance(out, torch.Tensor)
                    and out.dim() >= 1
                    and out.shape[dim] == padded_size
                ):
                    out = strategy.unpad(out, dim, real_size)
            else:
                if not _has_nested_attr(out, path):
                    raise AttributeError(
                        f"Registered output field path '{path}' "
                        f"does not exist on output object."
                    )
                tensor = _get_nested_attr(out, path)
                if tensor is not None:
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(
                            f"Registered output path '{path}' value is not a "
                            f"torch.Tensor at runtime. "
                            f"Got type: {type(tensor)}"
                        )
                    if (
                        tensor.dim() >= 1
                        and tensor.shape[dim] == padded_size
                    ):
                        sliced_tensor = strategy.unpad(tensor, dim, real_size)
                        _set_nested_attr(out, path, sliced_tensor)

        return out


# --------------------------------------------------------------------- #
# Core Static Shape Shim
# --------------------------------------------------------------------- #

class StaticShapeShim:
    """Central coordinator for compile-time shape padding.

    Manages compilation rules for dynamic shape axes (like token and batch).
    Resolves model-specific ShimRules, wraps methods with ShimWrapper, and
    orchestrates runtime dynamic padding and output unpadding.
    """

    def __init__(self, model_runner: Any):
        """Initializes the StaticShapeShim.

        Args:
            model_runner: ModelRunner hosting the compiled model.
        """
        self.model_runner = model_runner
        self.model = model_runner.model

        # Declarative rules for input-padding and output-unpadding
        self.rules = ShimRules()

    def install(self) -> StaticShapeShim:
        """Resolves model rules, registers hooks, and installs intercepts.

        Executes model strategy registration, wraps targeted methods with
        ShimWrapper intercepts, and returns the coordinator instance.

        Returns:
            The StaticShapeShim instance.
        """
        # Apply hooks dynamically based on model class name
        model_class_name = type(self.model).__name__
        rules_hook = _MODEL_SHIM_RULES.get(model_class_name)
        if rules_hook is not None:
            rules_hook(self.rules, self.model_runner)
            logger.info(
                "Applied static shape shim rules for %s.",
                model_class_name,
            )
        else:
            logger.warning(
                "No static shape rules registered for %s. No padding.",
                model_class_name,
            )
            return self

        wrapped: list[str] = []
        for attr_name, wrapper_cls in self.rules.wrapped_target_fn.items():
            fn = getattr(self.model, attr_name, None)
            if fn is None:
                logger.warning(
                    "Target function %s not found on %s.",
                    attr_name,
                    type(self.model).__name__,
                )
                continue
            if isinstance(fn, wrapper_cls):
                continue
            setattr(self.model, attr_name, wrapper_cls(fn, self))
            wrapped.append(attr_name)

        if wrapped:
            t_name = getattr(self.rules.t_bucketer, "name", "None")
            b_name = getattr(self.rules.b_bucketer, "name", "None")
            logger.info(
                "Installed compile-time input-padding shim on %s.{%s} "
                "(T-bucket: %s, B-bucket: %s).",
                type(self.model).__name__,
                ", ".join(wrapped),
                t_name,
                b_name,
            )
        return self


# --------------------------------------------------------------------- #
# Model-Specific Static Shape Padding Strategy Hooks
# --------------------------------------------------------------------- #

@register_model_shim_rules("Qwen3ForCausalLM")
def register_qwen3_strategies(rules: ShimRules, model_runner: Any) -> None:
    """Model-specific strategies for Qwen3 architecture."""
    rules.register_target_fn("forward", ShimWrapper)

    # Register and keep references to local Bucketers
    t_bucketer = Bucketer(
        name="PromptBucketer",
        scheme=PowerBucketScheme(
            min_val=64,
            max_val=model_runner.model_config.context_len,
            multiplier=8,
        ),
    )
    max_running = getattr(model_runner, "max_running_requests", None)
    assert max_running is not None and max_running > 0, (
        f"max_running_requests must be resolved to a positive integer, "
        f"but got {max_running}."
    )
    b_bucketer = Bucketer(
        name="BatchSizeBucketer",
        scheme=PowerBucketScheme(
            min_val=1,
            max_val=max_running,
            multiplier=8,
        ),
    )
    rules.t_bucketer = t_bucketer
    rules.b_bucketer = b_bucketer

    # Padding strategies
    zero_pad = ZeroPadStrategy()
    last_value = LastValueRepeatStrategy()
    stride_pad = LinearStridePadStrategy()
    scaled_zero = ScaledZeroPadStrategy()

    # Input token-axis: standard fields, zero padded
    rules.register_input_field("forward", "input_ids", 0, t_bucketer, zero_pad)
    rules.register_input_field("forward", "positions", 0, t_bucketer, zero_pad)
    rules.register_input_field(
        "forward", "input_embeds", 0, t_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.out_cache_loc", 0, t_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.out_cache_loc_swa", 0, t_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.seq_lens", 0, b_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.orig_seq_lens", 0, b_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.req_pool_indices", 0, b_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.extend_seq_lens", 0, b_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.extend_prefix_lens", 0, b_bucketer, zero_pad
    )
    rules.register_input_field(
        "forward", "forward_batch.encoder_lens", 0, b_bucketer, zero_pad
    )

    # Input batch-axis: cumulative fields. last element repeated.
    rules.register_input_field(
        "forward", "forward_batch.extend_start_loc", 0, b_bucketer, last_value
    )

    # Input attention backend: fields padded depending on semantics
    rules.register_input_field(
        "forward",
        "forward_batch.attn_backend.forward_metadata.kv_lens",
        0,
        b_bucketer,
        zero_pad,
    )
    rules.register_input_field(
        "forward",
        "forward_batch.attn_backend.forward_metadata.cu_q_lens",
        0,
        b_bucketer,
        last_value,
    )
    rules.register_input_field(
        "forward",
        "forward_batch.attn_backend.forward_metadata.cu_kv_lens",
        0,
        b_bucketer,
        stride_pad,
    )
    rules.register_input_field(
        "forward",
        "forward_batch.attn_backend.forward_metadata.page_indices",
        0,
        b_bucketer,
        scaled_zero,
    )

    # Output token-axis: slice padded token buckets back to real dynamic lengths
    rules.register_output_field("forward", "hidden_states", 0, t_bucketer)
