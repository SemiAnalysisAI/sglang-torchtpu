"""Two-axis eager input padding shim for static TPU compilation.

Wraps compiled model entry points (e.g., ``model.forward``) with an eager shim that pads
both token (T) and batch (B) dimensions to their next discrete bucket boundary.
Static TPU compilation (using PJRT / libtpu) caches HLO graphs keyed by exact tensor
shapes. Dynamic input shapes would trigger expensive runtime recompilations and
latency spikes; this shim maintains fixed compilation shapes at the TPU graph boundaries
by padding inputs to discrete bucket sizes, while restoring original shapes on output.

Coordinates declarative rules (via ShimRules) and polymorphic input/output strategies
(PadStrategy/UnpadStrategy) to ensure zero-overhead, type-safe shape alignment.
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

# Central registry mapping model class names (strings) to custom shim rules registration hooks
_MODEL_SHIM_RULES: dict[str, callable] = {}


def register_model_shim_rules(model_class_name: str):
    """Decorator to register custom static shape shim rules for a specific model class name.

    Args:
        model_class_name: The name of the target model class (e.g., "Qwen3ForCausalLM").
    """
    def decorator(func):
        _MODEL_SHIM_RULES[model_class_name] = func
        return func
    return decorator


# --------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------- #

def _get_nested_attr(obj: Any, path: str) -> Any | None:
    """Traverses a dot-separated path string to retrieve a nested attribute.

    Args:
        obj: The parent object containing the attribute hierarchy.
        path: A dot-separated string of attribute names (e.g., "parent.child.value").

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
        path: A dot-separated string of attribute names (e.g., "parent.child.value").
        value: The value to set on the final nested attribute.

    Returns:
        True if the attribute was successfully set, False if any intermediate segment
        along the path was None.
    """
    parts = path.split(".")
    curr = obj
    for part in parts[:-1]:
        curr = getattr(curr, part, None)
        if curr is None:
            return False
    setattr(curr, parts[-1], value)
    return True


# --------------------------------------------------------------------- #
# Padding Strategy Definitions
# --------------------------------------------------------------------- #

class PadStrategy(abc.ABC):
    """Polymorphic, stateless padding strategy interface for input tensors.

    Different input fields require specialized padding techniques to preserve
    algorithmic correctness when sizes are artificially inflated for static compilation.
    """

    def _normalize_dim(self, tensor: torch.Tensor, dim: int) -> int:
        """Normalizes negative dimensions and asserts they are valid for the tensor shape."""
        if dim < 0:
            dim += tensor.dim()
        assert 0 <= dim < tensor.dim(), f"Dimension index {dim} is out of bounds for tensor of dimension {tensor.dim()}."
        return dim

    def matches_shape(self, tensor: torch.Tensor, base_size: int, dim: int = 0) -> bool:
        """Determines if the strategy matches this tensor's target dimension size.

        Args:
            tensor: The target tensor to check.
            base_size: The active runtime size (T or B) before padding.
            dim: The target physical dimension index of the tensor.

        Returns:
            True if the strategy matches the tensor shape, False otherwise.
        """
        dim = self._normalize_dim(tensor, dim)
        return tensor.shape[dim] == base_size

    @abc.abstractmethod
    def pad(self, tensor: torch.Tensor, dim: int = 0, n_pad: int = 0, base_size: int | None = None) -> torch.Tensor:
        """Pads the target dimension of the tensor.

        Args:
            tensor: The target tensor to pad.
            dim: The target physical dimension index of the tensor.
            n_pad: The number of elements to pad along the target dimension.
            base_size: Optional active runtime size before padding.

        Returns:
            A new tensor padded along the target dimension according to the strategy.
        """
        pass


class UnpadStrategy(abc.ABC):
    """Polymorphic, stateless strategy interface for unpadding/slicing output tensors.

    Allows customizable output unpadding/slicing behaviors for different output fields.
    """

    @abc.abstractmethod
    def unpad(self, tensor: torch.Tensor, dim: int, real_size: int, bucket_size: int) -> torch.Tensor:
        """Slices or reconstructs the original unpadded tensor along the target dimension.

        Args:
            tensor: The target padded tensor to slice.
            dim: The physical dimension index to slice along.
            real_size: The original dynamic size before padding.
            bucket_size: The padded compile-time bucket size.

        Returns:
            A sliced tensor of original dynamic shape along the target dimension.
        """
        pass


class NarrowSliceStrategy(UnpadStrategy):
    """Default output unpadding strategy that slices/narrows the tensor down to its real size."""

    def unpad(self, tensor: torch.Tensor, dim: int, real_size: int, bucket_size: int) -> torch.Tensor:
        return tensor.narrow(dim, 0, real_size)


# Single reusable default instance of stateless unpadding strategy
_DEFAULT_UNPAD_STRATEGY = NarrowSliceStrategy()


class ZeroPadStrategy(PadStrategy):
    """Pads the target dimension of the tensor with zero values.

    Standard strategy for token-axis fields, flat metrics, or general indices where
    additional values are safely ignored or masked out in downstream computation.
    """

    def pad(self, tensor: torch.Tensor, dim: int = 0, n_pad: int = 0, base_size: int | None = None) -> torch.Tensor:
        if n_pad <= 0:
            return tensor
        dim = self._normalize_dim(tensor, dim)
        # Construct padding spec: [pad_last_dim_left, pad_last_dim_right, ..., pad_first_dim_left, pad_first_dim_right]
        pads = [[0, 0] for _ in range(tensor.dim())]
        pads[dim] = [0, n_pad]
        pads.reverse()
        pad_spec = [p for pad_pair in pads for p in pad_pair]
        return torch.nn.functional.pad(tensor, pad_spec, value=0.0)


class LastValueRepeatStrategy(PadStrategy):
    """Pads the target dimension by repeating the final element.

    Commonly used for cumulative index lists (e.g., ``extend_start_loc``) where
    subsequent operations rely on the last valid prefix index to avoid index errors.
    Supports standard base sizes or B + 1 cumulative layouts.
    """

    def matches_shape(self, tensor: torch.Tensor, base_size: int, dim: int = 0) -> bool:
        dim = self._normalize_dim(tensor, dim)
        return tensor.shape[dim] == base_size or tensor.shape[dim] == base_size + 1

    def pad(self, tensor: torch.Tensor, dim: int = 0, n_pad: int = 0, base_size: int | None = None) -> torch.Tensor:
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


class RectangularStridePadStrategy(PadStrategy):
    """Pads a cumulative 1-D tensor assuming a rectangular stride layout: ``tensor[i] = i * stride``.

    Pallas forward attention metadata fields (specifically ``cu_kv_lens``) use this to
    incrementally extend strides for batch-padded regions while avoiding expensive GPU syncs.
    """

    def matches_shape(self, tensor: torch.Tensor, base_size: int, dim: int = 0) -> bool:
        dim = self._normalize_dim(tensor, dim)
        return tensor.shape[dim] == base_size + 1

    def pad(self, tensor: torch.Tensor, dim: int = 0, n_pad: int = 0, base_size: int | None = None) -> torch.Tensor:
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
            torch.arange(b_real + 1, b_real + n_pad + 1, device=device, dtype=dtype)
            * stride
        )
        return torch.cat([tensor, new_tail], dim=dim)


class MultipliedZeroPadStrategy(PadStrategy):
    """Pads a 1-D tensor whose length is a multiple of the batch size (B) to ``B_bucket * multiple``.

    Designed for fields (such as Pallas `page_indices`) that scale with dynamic multiples
    of B, ensuring consistent alignment of padded sub-blocks.
    """

    def matches_shape(self, tensor: torch.Tensor, base_size: int, dim: int = 0) -> bool:
        dim = self._normalize_dim(tensor, dim)
        return base_size >= 1 and tensor.shape[dim] % base_size == 0

    def pad(self, tensor: torch.Tensor, dim: int = 0, n_pad: int = 0, base_size: int | None = None) -> torch.Tensor:
        if n_pad <= 0 or not base_size or base_size < 1:
            return tensor
        dim = self._normalize_dim(tensor, dim)
        multiple = tensor.shape[dim] // base_size
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
    """Declarative registry managing input-padding and output-unpadding strategies for model fields.

    Exposes simple builder-pattern methods to bind deep nested attribute paths in the model inputs
    and outputs to physical padding and unpadding strategies mapped to coordinating Bucketers.
    """

    def __init__(self):
        self.input_rules: dict[tuple[str | None, str], tuple[Bucketer, PadStrategy, int]] = {}
        self.output_rules: dict[tuple[str | None, str], tuple[Bucketer, UnpadStrategy, int]] = {}
        # Set of dynamic entry points/targets mapped to their wrapper class
        self.wrapped_target_fn: dict[str, type] = {}

        # Dynamic references to coordinating Bucketer objects representing logical dimensions
        self.t_bucketer: Bucketer | None = None
        self.b_bucketer: Bucketer | None = None

    def register_target_fn(self, target_fn_name: str, wrapper_cls: type | None = None) -> ShimRules:
        """Registers a model compiled entry point method name to wrap with the shape shim.

        Args:
            target_fn_name: The name of the method on the model instance (e.g., "forward_vision").
            wrapper_cls: The wrapper class to instantiate for this target (defaults to FuncWrapper).

        Returns:
            The ShimRules instance for fluent chaining.
        """
        if wrapper_cls is None:
            wrapper_cls = FuncWrapper
        self.wrapped_target_fn[target_fn_name] = wrapper_cls
        return self

    def register_input_field(
        self,
        target_fn: str | None,
        path: str,
        dim: int,
        strategy: PadStrategy,
        bucketer: Bucketer,
    ) -> ShimRules:
        """Binds a nested field dot-path to a padding strategy, physical dimension index, and active bucketer.

        Args:
            target_fn: Target function name scope (e.g., "forward"). If None, applies globally.
            path: A dot-separated path string (e.g., "seq_lens").
            dim: The physical dimension index of the tensor to apply padding on.
            strategy: The stateless padding strategy to execute for this path.
            bucketer: The active Bucketer instance coordinates padding for this field.

        Returns:
            The ShimRules instance for builder-pattern chaining.
        """
        self.input_rules[(target_fn, path)] = (bucketer, strategy, dim)
        return self

    def register_output_field(
        self,
        target_fn: str | None,
        path: str,
        dim: int,
        bucketer: Bucketer,
        strategy: UnpadStrategy | None = None,
    ) -> ShimRules:
        """Registers a dot-path representing a nested tensor inside the returned output to apply dynamic slicing on.

        Args:
            target_fn: Target function name scope (e.g., "forward"). If None, applies globally.
            path: A dot-separated path string representing a nested tensor inside the output (e.g., "hidden_states").
            dim: The physical dimension index of the returned tensor to slice back.
            bucketer: The active Bucketer instance that coordinates shapes for this output dimension.
            strategy: The stateless unpadding strategy to execute (defaults to NarrowSliceStrategy).

        Returns:
            The ShimRules instance for builder-pattern chaining.
        """
        if strategy is None:
            strategy = _DEFAULT_UNPAD_STRATEGY
        self.output_rules[(target_fn, path)] = (bucketer, strategy, dim)
        return self


# --------------------------------------------------------------------- #
# Function Wrapper
# --------------------------------------------------------------------- #

class FuncWrapper:
    """A callable wrapper that intercepts compiled model entry points to inject dynamic shape padding.

    Shadows targeted model methods to coordinate pre-compilation input padding and post-compilation
    output unpadding/slicing transparently. Performs inline shape verification and bookkeeping
    to ensure all inputs sharing a Bucketer have consistent physical dimensions.
    """

    def __init__(self, fn: Any, shim: StaticShapeShim):
        """Initializes the wrapper and updates metadata to preserve wrapped attributes.

        Args:
            fn: The original callable (e.g. the compiled model forward method).
            shim: The active StaticShapeShim orchestrating the padding logic.
        """
        self.fn = fn
        self.shim = shim
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs) -> Any:
        """Intercepts the method call, validates input shapes, pads inputs, executes, and unpads outputs.

        Performs a single-pass bookkeeping and consistency verification step to resolve a logical
        base size for each active Bucketer and verify all mapped input shapes match before applying
        the padding. Slices the output back to original unpadded shapes before returning.
        """
        shim = self.shim

        # Bind arguments to formal parameter names of the target function dynamically
        sig = inspect.signature(self.fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        target_fn_name = self.fn.__name__

        # Check if there are any registered rules for this target function scope
        rules = [
            (path, (bucketer, strategy, dim))
            for (target_scope, path), (bucketer, strategy, dim) in shim.rules.input_rules.items()
            if target_scope is None or target_scope == target_fn_name
        ]

        if not rules:
            return self.fn(*args, **kwargs)

        saved_tensors: dict[str, torch.Tensor] = {}
        resolved_sizes: dict[Bucketer, tuple[int, int]] = {}

        for path, (bucketer, strategy, dim) in rules:
            parts = path.split(".", 1)
            root_param = parts[0]
            if root_param not in bound.arguments:
                continue
            val = bound.arguments[root_param]
            if len(parts) > 1:
                nested_path = parts[1]
                target_val = _get_nested_attr(val, nested_path)
            else:
                nested_path = None
                target_val = val

            if target_val is None:
                continue

            if not isinstance(target_val, torch.Tensor):
                raise TypeError(
                    f"Registered path '{path}' value is not a torch.Tensor at runtime. "
                    f"Got type: {type(target_val)}"
                )
            if target_val.dim() < 1:
                raise ValueError(
                    f"Registered path '{path}' tensor has dimension {target_val.dim()} < 1. "
                    f"Cannot apply padding rules on a scalar."
                )

            # Bookkeeping & Resolution: Establish ground-truth base_size on first encounter
            if bucketer not in resolved_sizes:
                if isinstance(strategy, ZeroPadStrategy):
                    base_size = target_val.shape[dim]
                elif isinstance(strategy, RectangularStridePadStrategy):
                    base_size = max(1, target_val.shape[dim] - 1)
                else:
                    # Fallback: assume standard shape layout (base_size)
                    base_size = target_val.shape[dim]
                
                bucket_size = bucketer.next_bucket(base_size)
                resolved_sizes[bucketer] = (base_size, bucket_size)
            else:
                base_size, bucket_size = resolved_sizes[bucketer]

            # Consistency Verification: check that this field conforms to the resolved base_size
            if not strategy.matches_shape(target_val, base_size, dim):
                raise ValueError(
                    f"Inconsistent input shape for bucketer '{bucketer.name}'. "
                    f"The resolved base size is {base_size}, but field '{path}' "
                    f"along dimension {dim} has shape {target_val.shape[dim]} which "
                    f"does not match expected layout for strategy {type(strategy).__name__}."
                )

            # Apply dynamic padding
            n_pad = bucket_size - base_size
            if n_pad > 0:
                saved_tensors[path] = target_val
                padded_tensor = strategy.pad(target_val, dim, n_pad, base_size)
                if nested_path is not None:
                    _set_nested_attr(val, nested_path, padded_tensor)
                else:
                    bound.arguments[root_param] = padded_tensor

        try:
            out = self.fn(*bound.args, **bound.kwargs)
        finally:
            # Restore original unpadded tensors
            for path, orig_val in saved_tensors.items():
                parts = path.split(".", 1)
                root_param = parts[0]
                if len(parts) > 1:
                    nested_path = parts[1]
                    parent_obj = bound.arguments[root_param]
                    _set_nested_attr(parent_obj, nested_path, orig_val)
                else:
                    bound.arguments[root_param] = orig_val

        # Apply registered output slicing rules
        output_rules = [
            (path, (bucketer, strategy, dim))
            for (target_scope, path), (bucketer, strategy, dim) in shim.rules.output_rules.items()
            if target_scope is None or target_scope == target_fn_name
        ]

        for path, (bucketer, strategy, dim) in output_rules:
            if bucketer not in resolved_sizes:
                continue

            real_size, bucket_size = resolved_sizes[bucketer]
            if bucket_size == real_size:
                continue

            if not path:
                if isinstance(out, torch.Tensor) and out.dim() >= 1 and out.shape[dim] == bucket_size:
                    out = strategy.unpad(out, dim, real_size, bucket_size)
            else:
                target_val = _get_nested_attr(out, path)
                if target_val is not None:
                    if not isinstance(target_val, torch.Tensor):
                        raise TypeError(
                            f"Registered output path '{path}' value is not a torch.Tensor at runtime. "
                            f"Got type: {type(target_val)}"
                        )
                    if target_val.dim() >= 1 and target_val.shape[dim] == bucket_size:
                        sliced_tensor = strategy.unpad(target_val, dim, real_size, bucket_size)
                        _set_nested_attr(out, path, sliced_tensor)


# --------------------------------------------------------------------- #
# Core Static Shape Shim
# --------------------------------------------------------------------- #

class StaticShapeShim:
    """Central coordinator for two-axis compile-time shape padding on TPU.

    Manages the compilation rules for dynamic token (T) and batch (B) axes. Resolves
    model-specific ShimRules hooks, wraps compiled model methods with FuncWrapper intercepts,
    and orchestrates runtime dynamic padding and output unpadding.
    """

    def __init__(self, model_runner: Any):
        """Initializes the StaticShapeShim.

        Args:
            model_runner: The active ModelRunner hosting the TPU compiled model.
        """
        self.model_runner = model_runner
        self.model = model_runner.model

        # Declarative shim rules holding input-padding and output-unpadding bindings
        self.rules = ShimRules()

    def install(self) -> StaticShapeShim:
        """Resolves model rules, registers padding/unpadding hooks, and installs wrapper intercepts.

        Dynamically executes model-specific strategies registration (e.g. Qwen3), shadows targeted
        entry points on the model instance with FuncWrapper callables, and returns the coordinator.

        Returns:
            The StaticShapeShim instance itself for fluent chaining.
        """
        # Apply model-specific registered padding hooks dynamically based on model class name
        model_class_name = type(self.model).__name__
        rules_hook = _MODEL_SHIM_RULES.get(model_class_name)
        if rules_hook is not None:
            rules_hook(self.rules, self.model_runner)
            logger.info(
                "Applied model-specific static shape shim rules for %s.",
                model_class_name,
            )
        else:
            logger.info(
                "No static shape shim rules registered for %s. Static padding is disabled.",
                model_class_name,
            )
            return self

        wrapped: list[str] = []
        for attr_name, wrapper_cls in self.rules.wrapped_target_fn.items():
            fn = getattr(self.model, attr_name, None)
            if fn is None:
                continue
            if isinstance(fn, wrapper_cls):
                continue
            setattr(self.model, attr_name, wrapper_cls(fn, self))
            wrapped.append(attr_name)

        if wrapped:
            logger.info(
                "Installed two-axis input-padding shim on %s.{%s} "
                "(T-bucket from PromptBucketer, B-bucket from "
                "BatchSizeBucketer; pads forward_batch + Pallas attention "
                "metadata before the compiled forward).",
                type(self.model).__name__,
                ", ".join(wrapped),
            )
        return self


# --------------------------------------------------------------------- #
# Model-Specific Static Shape Padding Strategy Hooks
# --------------------------------------------------------------------- #

@register_model_shim_rules("Qwen3ForCausalLM")
def register_qwen3_strategies(rules: ShimRules, model_runner: Any) -> None:
    """Model-specific strategies for Qwen3 architecture."""
    rules.register_target_fn("forward", FuncWrapper)

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
    
    # Assign logical bucketer references to ShimRules for generic metadata lookup
    rules.t_bucketer = t_bucketer
    rules.b_bucketer = b_bucketer

    # Single instances of stateless strategies
    zero_pad = ZeroPadStrategy()
    last_value = LastValueRepeatStrategy()
    stride_pad = RectangularStridePadStrategy()
    multiplied_zero = MultipliedZeroPadStrategy()

    # Token-axis direct arguments (Zero Padded to the selected T-bucket)
    rules.register_input_field("forward", "input_ids", 0, zero_pad, t_bucketer)
    rules.register_input_field("forward", "positions", 0, zero_pad, t_bucketer)
    rules.register_input_field("forward", "input_embeds", 0, zero_pad, t_bucketer)

    # Token-axis nested forward_batch fields (Zero Padded to the selected T-bucket)
    rules.register_input_field("forward", "forward_batch.out_cache_loc", 0, zero_pad, t_bucketer)
    rules.register_input_field("forward", "forward_batch.out_cache_loc_swa", 0, zero_pad, t_bucketer)

    # Batch-axis standard fields inside forward_batch (Zero Padded to the selected B-bucket)
    rules.register_input_field("forward", "forward_batch.seq_lens", 0, zero_pad, b_bucketer)
    rules.register_input_field("forward", "forward_batch.orig_seq_lens", 0, zero_pad, b_bucketer)
    rules.register_input_field("forward", "forward_batch.req_pool_indices", 0, zero_pad, b_bucketer)
    rules.register_input_field("forward", "forward_batch.extend_seq_lens", 0, zero_pad, b_bucketer)
    rules.register_input_field("forward", "forward_batch.extend_prefix_lens", 0, zero_pad, b_bucketer)
    rules.register_input_field("forward", "forward_batch.encoder_lens", 0, zero_pad, b_bucketer)

    # Batch-axis cumulative fields inside forward_batch (Last element repeated to selected B-bucket)
    rules.register_input_field("forward", "forward_batch.extend_start_loc", 0, last_value, b_bucketer)

    # Pallas forward attention metadata B-fields inside forward_batch (RPA-v3 / ragged paged attention)
    rules.register_input_field("forward", "forward_batch.attn_backend.forward_metadata.kv_lens", 0, zero_pad, b_bucketer)
    rules.register_input_field("forward", "forward_batch.attn_backend.forward_metadata.cu_q_lens", 0, last_value, b_bucketer)
    rules.register_input_field("forward", "forward_batch.attn_backend.forward_metadata.cu_kv_lens", 0, stride_pad, b_bucketer)
    rules.register_input_field("forward", "forward_batch.attn_backend.forward_metadata.page_indices", 0, multiplied_zero, b_bucketer)

    # Output slicing rules (slice padded token/batch buckets back to real dynamic lengths)
    rules.register_output_field("forward", "hidden_states", 0, t_bucketer)
