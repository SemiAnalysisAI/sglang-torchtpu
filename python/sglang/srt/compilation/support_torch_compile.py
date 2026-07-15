"""Lightweight module-level torch.compile annotations.

The annotation is intentionally metadata-only. Platform-specific runners decide
how to install, pad, and precompile annotated module boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, TypeVar

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

    @abstractmethod
    def unpad_outputs(self, out: Any, context: Any) -> Any:
        """Adapt padded compiled outputs back to the caller-visible shape."""

    # Speculative draft-decode I/O shaping. Base is a no-op; the TPU spec policy
    # overrides these to pad/trim the compiled draft-decode forward. Kept on the
    # base so a runner can call them on any installed policy without a guard.
    def pad_draft_decode(self, forward_batch: Any) -> None:
        """Shape the speculative draft-decode batch for the compiled forward."""

    def pad_draft_extend(self, forward_batch: Any) -> None:
        """Shape the speculative draft-extend batch for the compiled forward."""

    def unpad_draft_outputs(self, outputs: Any, forward_batch: Any) -> Any:
        """Undo ``pad_draft_decode`` on the draft outputs (default: passthrough)."""
        return outputs

    def precompile_draft(
        self, draft_attn_backend: Any, draft_extend_attn_backend: Any
    ) -> None:
        """Warm the draft model's compiled graphs once the spec worker's
        attention backends exist.

        The draft forwards run on worker-owned backends created after
        ``precompile()``, so policies that precompile draft shapes implement
        this second, deferred phase. Base is a no-op.
        """

    def pad_spec_verify(
        self,
        next_token_logits: Any,
        candidates: Any,
        retrieve_index: Any,
        predict: Any,
        accept_index: Any,
        nums: Any,
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

    def unpad_spec_verify(
        self, predict: Any, accept_index: Any, nums: Any, real_bs: int
    ):
        """Trim the greedy-verify *outputs* back to the real bs. Base: passthrough."""
        return predict, accept_index, nums

    def pack_spec_accept_index(self, accept_index: Any) -> Any:
        """Static-shape packed accept index (a ``PaddedAcceptIndex``) for the
        draft-extend gathers. Base: ``None`` (the variable path is used)."""
        return None

    def pad_spec_draft_extend_input(
        self, di: Any, batch: Any, speculative_num_steps: int
    ) -> None:
        """Pad the draft-extend input tensors to a static token shape. Base: no-op."""

    def pad_draft_extend_v2(
        self, forward_batch: Any, model_runner: Any, num_draft_tokens: int
    ) -> None:
        """Shape the speculative draft-extend batch for the compiled forward."""


class NoopCompilePolicy(CompilePolicy):
    """Default noop policy a runner carries until a real one installs itself."""

    def install(self) -> bool:
        return False

    def precompile(self) -> None:
        pass

    def unpad_outputs(self, out: Any, context: Any) -> Any:
        return out


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
