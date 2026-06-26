"""TPU compiled-forward adaptation for bucketized static shapes.

This module pads ForwardBatch tensors and Python-list metadata to configured
token/batch buckets before attention metadata is built. It wraps compiled
model forwards to normalize default kwargs at the Dynamo boundary and slices
padded decoder outputs back to real request shapes.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import torch
import torch.nn.functional as F

from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketer,
)

logger = get_logger(__name__)

DEFAULT_TPU_COMPILE_MAX_BS = 64

# Fields whose token dim gets padded to the token bucket.
_TOKEN_FIELDS = ("input_ids", "positions")

# Fields whose batch dim gets padded to the bs bucket.
_BS_FIELDS = (
    "seq_lens",
    "orig_seq_lens",
    "req_pool_indices",
    "extend_seq_lens",
    "extend_prefix_lens",
    "encoder_lens",
)

# Python-list batch metadata can become Dynamo guards; pad to bucket length.
_BS_CPU_LIST_FIELDS = (
    "extend_seq_lens_cpu",
    "extend_prefix_lens_cpu",
    "extend_logprob_start_lens_cpu",
)

_REAL_Q_KEY = "_tpu_real_q"
_REAL_B_KEY = "_tpu_real_b"
_DEFAULT_KWARG_TYPES = (type(None), bool, int, float, str)


def _static_default_kwargs(fn: Callable[..., Any]) -> dict[str, Any]:
    signature = getattr(fn, "__signature__", None)
    if signature is None:
        return {}
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
        and isinstance(param.default, _DEFAULT_KWARG_TYPES)
    }


def _matches_default(value: Any, default: Any) -> bool:
    if default is None:
        return value is None
    return type(value) is type(default) and value == default


def _zero_pad_1d(t: torch.Tensor, target_size: int) -> torch.Tensor:
    """Zero-pad a 1-D tensor to target_size."""
    if t.shape[0] >= target_size:
        return t
    return F.pad(t, (0, target_size - t.shape[0]))


def _pad_forward_batch_inputs(
    forward_batch: Any,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
) -> tuple[int, int]:
    """Pad ForwardBatch tensors to next bucket shape and return original sizes."""
    if forward_batch.return_logprob:
        raise NotImplementedError(
            "return_logprob is not supported on the TPU backend yet."
        )

    is_decode = forward_batch.forward_mode.is_decode()
    real_q = forward_batch.input_ids.shape[0]
    real_b = forward_batch.seq_lens.shape[0]

    if is_decode:
        pad_t = b_bucketer.next_bucket(real_q)
        pad_b = b_bucketer.next_bucket(real_b)
    else:
        # Prefill/extend is precompiled at a fixed batch cap to avoid compiling
        # the full token_bucket x batch_bucket cross product.
        pad_b = b_bucketer.max_val
        pad_t = t_bucketer.next_bucket(max(real_q, pad_b))

    for field in _TOKEN_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_t:
            setattr(forward_batch, field, _zero_pad_1d(val, pad_t))

    # MROPE positions are (3, num_tokens); pad along the token dim.
    mrope_pos = getattr(forward_batch, "mrope_positions", None)
    if isinstance(mrope_pos, torch.Tensor) and mrope_pos.shape[-1] < pad_t:
        forward_batch.mrope_positions = F.pad(
            mrope_pos, (0, pad_t - mrope_pos.shape[-1])
        )

    for field in _BS_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_b:
            setattr(forward_batch, field, _zero_pad_1d(val, pad_b))

    for field in _BS_CPU_LIST_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, list) and len(val) < pad_b:
            setattr(forward_batch, field, val + [0] * (pad_b - len(val)))

    # mm_inputs is per request. Keep it aligned with padded extend_* CPU lists
    # so eager multimodal embedding code can index by batch position.
    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if isinstance(mm_inputs, list) and len(mm_inputs) < pad_b:
        forward_batch.mm_inputs = mm_inputs + [None] * (pad_b - len(mm_inputs))

    forward_batch.batch_size = pad_b

    return real_q, real_b


class CompiledForwardShim:
    """Adapt compiled forward calls and outputs for padded execution."""

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        policy: CompilePolicy,
    ) -> None:
        self.fn = compiled_fn
        self.policy = policy
        signature = getattr(compiled_fn, "__signature__", None)
        if signature is not None:
            self.__signature__ = signature
        self._default_kwargs = _static_default_kwargs(compiled_fn)
        functools.update_wrapper(self, compiled_fn)

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        forward_batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._default_kwargs and kwargs:
            # Dynamo treats omitted kwargs and explicit defaults as different
            # call shapes; drop explicit default values.
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if not (
                    k in self._default_kwargs
                    and _matches_default(v, self._default_kwargs[k])
                )
            }
        out = self.fn(input_ids, positions, forward_batch, *args, **kwargs)
        return self.policy.unpad_outputs(out, forward_batch)


def resolve_compile_caps(
    *,
    chunked_prefill_size: int | None,
    max_prefill_tokens: int | None,
    context_len: int | None,
    torch_compile_max_bs: int | None,
    max_running_requests: int | None,
) -> tuple[int, int]:
    """Resolve (max_t, max_b) limits for compilation bucketing."""
    candidates = [
        c
        for c in (chunked_prefill_size, max_prefill_tokens, context_len)
        if c and c > 0
    ]
    if not candidates:
        raise ValueError(
            "Cannot determine max_t for TPU compile bucketing: "
            "chunked_prefill_size, max_prefill_tokens, and context_len are all unset."
        )
    # TPU serving should compile the requested scheduler capacity by default for
    # practical benchmark sizes (for example max_running_requests=64). Avoid
    # blindly expanding to very large upstream defaults unless the user raises
    # torch_compile_max_bs or pins --tpu-compile-bs.
    max_running_compile_bs = (
        min(max_running_requests, DEFAULT_TPU_COMPILE_MAX_BS)
        if max_running_requests
        else 1
    )
    max_b = max(torch_compile_max_bs or 1, max_running_compile_bs)
    if max_running_requests and max_b < max_running_requests:
        logger.warning(
            "[compiled forward shim] TPU compile bs cap=%d < max_running_requests=%d; "
            "batches above %d are outside the TPU compile grid. Raise "
            "--torch-compile-max-bs or pin --tpu-compile-bs to cover them.",
            max_b,
            max_running_requests,
            max_b,
        )
    return min(candidates), max_b


def build_bucketers(server_args: Any, max_t: int, max_b: int) -> tuple[Bucketer, Bucketer]:
    """Build token and batch size bucketers."""
    token_paddings = server_args.tpu_compile_tokens
    if token_paddings:
        token_bucketer: Bucketer = Bucketer(
            name="TokenBucketer", buckets=list(token_paddings)
        )
    else:
        token_bucketer = PowerBucketer(
            name="TokenBucketer", min_val=64, max_val=max_t, power=2
        )

    bs_paddings = server_args.tpu_compile_bs
    if bs_paddings:
        bs_bucketer: Bucketer = Bucketer(
            name="BatchSizeBucketer", buckets=list(bs_paddings)
        )
    else:
        bs_bucketer = PowerBucketer(
            name="BatchSizeBucketer", min_val=1, max_val=max_b, power=2
        )

    return token_bucketer, bs_bucketer
