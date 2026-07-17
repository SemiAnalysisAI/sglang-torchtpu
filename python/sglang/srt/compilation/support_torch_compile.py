"""Lightweight module-level torch.compile annotations.

The annotation is intentionally metadata-only. Platform-specific runners decide
how to install, pad, and precompile annotated module boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, TypeVar

import torch

from sglang.srt.speculative.spec_info import AcceptIndices, SpecInput

__all__ = [
    "CompilePolicy",
    "get_torch_compile_policy",
    "support_torch_compile",
]


_COMPILE_POLICY_ATTR = "_sglang_torch_compile_policy"
T = TypeVar("T")


class CompilePolicy(ABC):
    """Base class for annotation-driven torch.compile policies."""

    backend: ClassVar[str | None] = None
    fullgraph: ClassVar[bool] = True
    dynamic: ClassVar[bool] = False
    should_precompile: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model_runner: Any | None = None,
        owner: Any | None = None,
        method_name: str = "forward",
        method: Callable[..., Any] | None = None,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        self.model_runner = model_runner
        self.owner = owner
        self.method_name = method_name
        self.method = method
        self.name = name or type(self).__name__
        self.label = label

    @abstractmethod
    def install(self) -> bool:
        """Install runtime compile wrappers for this policy."""

    @abstractmethod
    def precompile(self) -> None:
        """Warm compile artifacts for this policy."""

    def precompile_draft(
        self, draft_attn_backend: Any, draft_extend_attn_backend: Any
    ) -> None:
        """Run the delayed speculative-draft warmup phase.

        Target and ordinary model warmup belongs in ``precompile()``. Draft
        forwards use attention backends owned by the speculative worker, which
        are created only after the model runner's initial ``precompile()`` call.
        Policies that compile draft shapes therefore implement this second-stage
        hook once those backends are available. Base is a no-op.
        """

    @abstractmethod
    def finalize_forward_outputs(self, out: Any, context: Any) -> Any:
        """Adapt padded compiled outputs back to the caller-visible shape."""

    def prepare_spec_forward_batch(self, forward_batch: Any) -> None:
        """Apply platform shaping before speculative attention planning."""

    def finalize_spec_forward_batch(
        self, outputs: Any, forward_batch: Any
    ) -> Any:
        """Finalize speculative forward outputs at their logical boundary."""
        return outputs

    def prepare_spec_verify(
        self,
        next_token_logits: torch.Tensor,
        candidates: torch.Tensor,
        retrieve_index: torch.Tensor,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
        nums: torch.Tensor,
        spec_info: SpecInput,
    ) -> Any:
        """Pad the greedy-verify *inputs* to a static shape"""
        return (
            next_token_logits,
            candidates,
            retrieve_index,
            predict,
            accept_index,
            nums,
        )

    def finalize_spec_verify(
        self,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
        nums: torch.Tensor,
        real_bs: int,
        spec_info: SpecInput,
    ) -> Any:
        """Trim the greedy-verify *outputs* back to the real bs. Base: passthrough."""
        return predict, accept_index, nums

    @abstractmethod
    def build_accepted_indices(
        self, per_request: torch.Tensor
    ) -> AcceptIndices:
        """Build logical and platform-specific accepted-index views."""


class NoopCompilePolicy(CompilePolicy):
    """Default noop policy a runner carries until a real one installs itself."""

    def install(self) -> bool:
        return False

    def precompile(self) -> None:
        pass

    def finalize_forward_outputs(self, out: Any, context: Any) -> Any:
        return out

    def build_accepted_indices(
        self, per_request: torch.Tensor
    ) -> AcceptIndices:
        return AcceptIndices.from_per_request(per_request)


def support_torch_compile(
    obj: T | None = None,
    *,
    policy: type[CompilePolicy],
) -> Callable[[T], T] | T:
    """Attach a compile policy to a module boundary.

    Preferred use is annotating the module class or instance:

    ``support_torch_compile(MyModule, policy=DecoderTpuCompilePolicy)``.

    """

    if not isinstance(policy, type) or not issubclass(policy, CompilePolicy):
        raise TypeError("support_torch_compile policy must be a CompilePolicy class")

    def decorator(target_obj: T) -> T:
        setattr(target_obj, _COMPILE_POLICY_ATTR, policy)
        return target_obj

    if obj is None:
        return decorator

    return decorator(obj)


def get_torch_compile_policy(obj: Any) -> type[CompilePolicy] | None:
    """Return the compile policy class attached to ``obj``, if any."""

    return getattr(obj, _COMPILE_POLICY_ATTR, None)
