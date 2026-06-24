"""TPU compiled-forward adaptation for bucketized static shapes.

This module pads ForwardBatch tensors and Python-list metadata to
configured token/batch buckets before attention metadata is built. It
wraps compiled model forwards to normalize default kwargs at the Dynamo
boundary and slices padded outputs back to real request shapes.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable

import torch
import torch.nn.functional as F

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


def _static_default_kwargs(fn: Callable) -> dict:
    signature = getattr(fn, "__signature__", None)
    if signature is None:
        return {}
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
        and isinstance(param.default, _DEFAULT_KWARG_TYPES)
    }


def _matches_default(value, default) -> bool:
    if default is None:
        return value is None
    return type(value) is type(default) and value == default


def _zero_pad_1d(t: torch.Tensor, target_size: int) -> torch.Tensor:
    """Zero-pad a 1-D tensor to target_size."""
    if t.shape[0] >= target_size:
        return t
    return F.pad(t, (0, target_size - t.shape[0]))


def _pad_first_dim(t: torch.Tensor | None, target_rows: int) -> torch.Tensor | None:
    """Zero-pad a tensor's first (row) dim to target_rows (any ndim).

    None (an unset field) passes through unchanged.
    """
    if t is None or t.shape[0] >= target_rows:
        return t
    pad = t.new_zeros((target_rows - t.shape[0], *t.shape[1:]))
    return torch.cat([t, pad], dim=0)


def _pad_forward_batch_inputs(
    forward_batch,
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

    for field in _BS_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_b:
            setattr(forward_batch, field, _zero_pad_1d(val, pad_b))

    for field in _BS_CPU_LIST_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, list) and len(val) < pad_b:
            setattr(forward_batch, field, val + [0] * (pad_b - len(val)))

    # Pad spec_info tensors whose token dim must match input_ids after padding.
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is not None:
        hs = getattr(spec_info, "hidden_states", None)
        if isinstance(hs, torch.Tensor) and hs.ndim >= 1 and hs.shape[0] < pad_t:
            pad_rows = pad_t - hs.shape[0]
            spec_info.hidden_states = F.pad(
                hs, (0,) * (2 * (hs.ndim - 1)) + (0, pad_rows)
            )

    forward_batch.batch_size = pad_b

    return real_q, real_b


def pad_draft_decode_inputs(forward_batch, b_bucketer: Bucketer) -> int | None:
    """Pad an EAGLE draft-decode ForwardBatch to a torch.compile batch bucket."""
    real_bs = int(forward_batch.seq_lens.shape[0])
    pad_b = b_bucketer.next_bucket(real_bs)
    if pad_b == real_bs:
        return None

    forward_batch.seq_lens = _pad_first_dim(forward_batch.seq_lens, pad_b)
    forward_batch.req_pool_indices = _pad_first_dim(
        forward_batch.req_pool_indices, pad_b
    )
    forward_batch.positions = _pad_first_dim(
        forward_batch.positions, forward_batch.positions.shape[0] // real_bs * pad_b
    )
    forward_batch.out_cache_loc = _pad_first_dim(
        forward_batch.out_cache_loc,
        forward_batch.out_cache_loc.shape[0] // real_bs * pad_b,
    )

    # EagleDraftInput seeds consumed by select_top_k_tokens.
    spec_info = forward_batch.spec_info
    if spec_info is not None:
        spec_info.topk_p = _pad_first_dim(spec_info.topk_p, pad_b)
        spec_info.topk_index = _pad_first_dim(spec_info.topk_index, pad_b)
        spec_info.hidden_states = _pad_first_dim(spec_info.hidden_states, pad_b)

    forward_batch.batch_size = pad_b
    forward_batch._tpu_real_b = real_bs
    return real_bs


def _pad_first_dim_with_value(
    t: torch.Tensor | None, target_rows: int, fill_value: int
) -> torch.Tensor | None:
    """Pad a tensor's first dim to target_rows, filling new rows with fill_value.

    Used for ghost rows of bookkeeping tensors (e.g. req_pool_indices) where a
    plain zero-pad would point ghost reqs at the permanent padding slot. We
    instead reuse the last real value so the attention metadata for ghost
    rows is built from a valid req_to_token row.
    """
    if t is None or t.shape[0] >= target_rows:
        return t
    pad_shape = (target_rows - t.shape[0], *t.shape[1:])
    pad = t.new_full(pad_shape, fill_value)
    return torch.cat([t, pad], dim=0)


def pad_draft_extend_inputs(
    forward_batch,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
    *,
    draft_token_num: int | None = None,
) -> tuple[int, int] | None:
    """Pad an EAGLE draft-extend ForwardBatch to fixed (tokens, bs) buckets.

    The draft-extend forward (``forward_draft_extend_after_decode``) runs once
    per verify step on the draft model with a query total equal to the sum of
    per-seq accept lengths -- a value that varies every step. Without padding,
    every distinct (bs, sum_accept) pair compiles a fresh HLO graph.

    Per-bs bucketing (matching ``CompilationManager._precompile_draft_extend``):

    * ``pad_b = b_bucketer.next_bucket(real_b)`` -- the smallest bs bucket
      that fits the real batch.
    * ``pad_t = t_bucketer.next_bucket(max(real_q, pad_b, pad_b * draft_token_num))``
      when ``draft_token_num`` is provided -- pads to the per-bs worst case so
      the runtime shape always matches one of the precompiled (bs, t) pairs.
      When ``draft_token_num`` is ``None``, falls back to
      ``max(real_q, pad_b)`` (single-shape policy).

    Must be called *after* ``ForwardBatch.init_new`` has materialized
    ``extend_seq_lens`` / ``extend_prefix_lens`` / ``positions`` / etc. as
    tensors, and *before* ``attn_backend.init_forward_metadata`` consumes
    them. The caller (eagle_worker) bypasses the model-level
    ``prepare_forward_batch`` hook by passing ``skip_attn_backend_init=True``
    to ``forward``, so this function is the only padding site.

    Returns ``(real_q, real_b)`` so the caller can record them on the batch
    for output slicing in ``CompiledForwardShim.__call__``, or ``None`` if
    no padding was needed.
    """
    real_q = int(forward_batch.input_ids.shape[0])
    real_b = int(forward_batch.seq_lens.shape[0])

    # Per-bs bucketing: pad batch to the smallest bs bucket that fits real_b,
    # then choose the token bucket holding the per-bs worst-case q total.
    pad_b = b_bucketer.next_bucket(real_b)
    worst_q_for_pad_b = (
        pad_b * draft_token_num if draft_token_num is not None else pad_b
    )
    pad_t = t_bucketer.next_bucket(max(real_q, pad_b, worst_q_for_pad_b))
    if pad_t == real_q and pad_b == real_b:
        return None

    # Pick a ghost-slot req_pool_index that already has valid req_to_token
    # entries: reuse the last real index. (Zero would point at the permanent
    # padding slot and may not have its page indices populated for this
    # request's KV pool.) Fall back to 0 only when the batch is empty.
    ghost_pool_idx = int(forward_batch.req_pool_indices[-1].item()) if real_b > 0 else 0

    # Token-dim fields (one entry per query token).
    forward_batch.input_ids = _pad_first_dim(forward_batch.input_ids, pad_t)
    forward_batch.positions = _pad_first_dim(forward_batch.positions, pad_t)
    if forward_batch.out_cache_loc is not None:
        forward_batch.out_cache_loc = _pad_first_dim(forward_batch.out_cache_loc, pad_t)

    # Batch-dim fields (one entry per request). extend_seq_lens ghosts must be
    # zero so cu_q_lens stays monotone and the kernel sees no extra work for
    # ghost reqs. seq_lens / extend_prefix_lens are also zero-padded (the RPA
    # backend reads seq_lens but relies on the active-row `distribution` cap
    # to ignore ghost rows). req_pool_indices ghosts use a valid index so the
    # block-table gather stays in-bounds.
    forward_batch.seq_lens = _pad_first_dim(forward_batch.seq_lens, pad_b)
    if getattr(forward_batch, "orig_seq_lens", None) is not None:
        forward_batch.orig_seq_lens = _pad_first_dim(forward_batch.orig_seq_lens, pad_b)
    forward_batch.req_pool_indices = _pad_first_dim_with_value(
        forward_batch.req_pool_indices, pad_b, ghost_pool_idx
    )
    if forward_batch.extend_seq_lens is not None:
        forward_batch.extend_seq_lens = _pad_first_dim(
            forward_batch.extend_seq_lens, pad_b
        )
    if forward_batch.extend_prefix_lens is not None:
        forward_batch.extend_prefix_lens = _pad_first_dim(
            forward_batch.extend_prefix_lens, pad_b
        )
    if forward_batch.extend_start_loc is not None:
        # Ghost reqs contribute zero tokens, so they all start at real_q.
        forward_batch.extend_start_loc = _pad_first_dim_with_value(
            forward_batch.extend_start_loc, pad_b, real_q
        )

    # Python-list batch metadata can become Dynamo guards; mirror _BS_CPU_LIST_FIELDS.
    for field in _BS_CPU_LIST_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, list) and len(val) < pad_b:
            setattr(forward_batch, field, val + [0] * (pad_b - len(val)))

    # spec_info.hidden_states is the per-token target hidden feed. Pad its
    # token dim to match input_ids.
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is not None:
        hs = getattr(spec_info, "hidden_states", None)
        if isinstance(hs, torch.Tensor) and hs.ndim >= 1 and hs.shape[0] < pad_t:
            spec_info.hidden_states = _pad_first_dim(hs, pad_t)

    forward_batch.batch_size = pad_b
    setattr(forward_batch, _REAL_Q_KEY, real_q)
    setattr(forward_batch, _REAL_B_KEY, real_b)
    return real_q, real_b


def _tensor_summary(t, max_show: int = 8) -> str:
    """Compact one-line summary of a tensor: shape, dtype, head/tail values, sum.

    Designed for diff-friendly logging: two calls with identical structure
    produce identical strings; any drift in shape/dtype/values shows up
    immediately when the lines are compared.
    """
    if t is None:
        return "None"
    if not isinstance(t, torch.Tensor):
        return f"<{type(t).__name__}>"
    n = int(t.numel())
    if n == 0:
        return f"shape={tuple(t.shape)} dtype={t.dtype} empty"
    try:
        # 1-D tensor: dump head and tail. ND: dump shape + dtype + sum.
        flat = t.detach().reshape(-1)
        if n <= max_show:
            vals = flat.tolist()
            vstr = f"vals={vals}"
        else:
            head = flat[:max_show].tolist()
            tail = flat[-min(4, n - max_show) :].tolist()
            vstr = f"head={head} tail={tail}"
        # Try to print a checksum for content fingerprinting (skip on bf16
        # to keep the sum deterministic across runs at the same call).
        if t.dtype.is_floating_point:
            chk = f"sum={float(flat.float().sum().item()):.4g}"
        else:
            chk = f"sum={int(flat.long().sum().item())}"
        return f"shape={tuple(t.shape)} dtype={t.dtype} {vstr} {chk}"
    except Exception as exc:  # noqa: BLE001
        return f"shape={tuple(t.shape)} dtype={t.dtype} <readback-failed: {exc}>"


def dump_draft_extend_inputs(forward_batch, tag: str = "draft_extend") -> None:
    """Log a compact snapshot of every tensor that feeds the draft forward.

    Gated by ``SGLANG_TPU_DUMP_DRAFT_INPUTS=1``. Intended for diffing
    consecutive calls to find what is changing between forward invocations
    when a recompile is observed. Logs at WARNING level so it shows up
    even with default log_level=info, and prefixes each line with the
    rank/mode tag for easy grep.
    """
    import os

    if os.environ.get("SGLANG_TPU_DUMP_DRAFT_INPUTS") != "1":
        return
    fb = forward_batch
    mode = getattr(getattr(fb, "forward_mode", None), "name", "?")
    real_q = getattr(fb, _REAL_Q_KEY, None)
    real_b = getattr(fb, _REAL_B_KEY, None)
    bs = getattr(fb, "batch_size", None)
    lines = [
        f"[{tag} {mode}] bs={bs} real_q={real_q} real_b={real_b}",
        f"  input_ids        = {_tensor_summary(fb.input_ids)}",
        f"  positions        = {_tensor_summary(fb.positions)}",
        f"  seq_lens         = {_tensor_summary(fb.seq_lens)}",
        f"  orig_seq_lens    = {_tensor_summary(getattr(fb, 'orig_seq_lens', None))}",
        f"  extend_seq_lens  = {_tensor_summary(getattr(fb, 'extend_seq_lens', None))}",
        f"  extend_prefix_lens = {_tensor_summary(getattr(fb, 'extend_prefix_lens', None))}",
        f"  extend_start_loc = {_tensor_summary(getattr(fb, 'extend_start_loc', None))}",
        f"  req_pool_indices = {_tensor_summary(fb.req_pool_indices)}",
        f"  out_cache_loc    = {_tensor_summary(getattr(fb, 'out_cache_loc', None))}",
    ]
    # spec_info.hidden_states is the per-token target hidden feed (EAGLE3).
    spec_info = getattr(fb, "spec_info", None)
    if spec_info is not None:
        for attr in ("hidden_states", "topk_p", "topk_index", "bonus_tokens"):
            val = getattr(spec_info, attr, None)
            if val is not None:
                lines.append(f"  spec_info.{attr:14s} = {_tensor_summary(val)}")
    # Attention backend's per-call kernel inputs (what Pallas actually sees).
    attn = getattr(fb, "attn_backend", None)
    fmd = getattr(attn, "forward_metadata", None)
    if fmd is not None:
        for attr in (
            "kv_lens",
            "cu_q_lens",
            "cu_kv_lens",
            "page_indices",
            "distribution",
        ):
            val = getattr(fmd, attr, None)
            if val is not None:
                lines.append(f"  attn.{attr:14s}       = {_tensor_summary(val)}")
    logger.warning("\n".join(lines))


class CompiledForwardShim:
    """Adapt compiled forward calls and outputs for padded execution."""

    def __init__(self, compiled_fn):
        self.fn = compiled_fn
        self._compilation_manager = None
        self._model_label = ""
        signature = getattr(compiled_fn, "__signature__", None)
        if signature is not None:
            self.__signature__ = signature
        self._default_kwargs = _static_default_kwargs(compiled_fn)
        functools.update_wrapper(self, compiled_fn)

    def set_compilation_manager(self, manager, model_label: str = "") -> None:
        """Set the compilation manager for serving-time recompilation checks.

        Args:
            manager: CompilationManager instance.
            model_label: human-readable model identifier for log context
                (e.g. "LlamaForCausalLM").
        """
        self._compilation_manager = manager
        self._model_label = model_label

    def __call__(self, input_ids, positions, forward_batch, *args, **kwargs):
        if self._default_kwargs and kwargs:
            # Dynamo treats omitted kwargs and explicit defaults as
            # different call shapes; drop explicit default values.
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if not (
                    k in self._default_kwargs
                    and _matches_default(v, self._default_kwargs[k])
                )
            }
        out = self.fn(input_ids, positions, forward_batch, *args, **kwargs)

        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        if real_q is None or real_b is None:
            return out

        next_token_logits = getattr(out, "next_token_logits", None)
        if next_token_logits is not None and real_b < next_token_logits.shape[0]:
            # TARGET_VERIFY outputs bs * draft_token_num logits (one per
            # draft token across all requests).  Slicing to real_b would
            # drop the verify tokens.  Use real_q instead.
            fm = getattr(forward_batch, "forward_mode", None)
            logit_rows = (
                real_q if (fm is not None and fm.is_target_verify()) else real_b
            )
            if logit_rows < next_token_logits.shape[0]:
                out.next_token_logits = next_token_logits[:logit_rows]

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is not None:
            # Capture-LAST (EAGLE draft extend) emits one hidden row per
            # sequence, i.e. a batch-dim tensor padded to the compile bucket, so
            # it must be sliced back to the real batch size. Every other capture
            # mode emits one row per token (real_q).
            cap = getattr(forward_batch, "capture_hidden_mode", None)
            n_real = real_b if (cap is not None and cap.is_last()) else real_q
            if n_real < hidden_states.shape[0]:
                out.hidden_states = hidden_states[:n_real]

        # Restore real batch_size so downstream consumers (sampler,
        # logprob, scheduler hooks) don't iterate over ghost slots.
        forward_batch.batch_size = real_b

        # Check for serving-time XLA recompilation (once per forward call).
        # Gated by SGLANG_TPU_CHECK_RECOMPILATION=1 (off by default).
        if self._compilation_manager is not None:
            fm = getattr(forward_batch, "forward_mode", "N/A")
            self._compilation_manager.check_recompilation(
                f"{self._model_label} {fm.name if hasattr(fm, 'name') else fm}"
            )

        return out


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


class TpuCompiledForwardShim:
    """Installs bucket padding and compiled-forward adaptation."""

    def __init__(self, model_runner, compiled_fns: dict[str, Callable]):
        self.model_runner = model_runner
        self.model = model_runner.model
        self._compiled_fns = compiled_fns

        server_args = model_runner.server_args
        max_t, max_b = resolve_compile_caps(
            chunked_prefill_size=server_args.chunked_prefill_size,
            max_prefill_tokens=server_args.max_prefill_tokens,
            context_len=model_runner.model_config.context_len,
            torch_compile_max_bs=getattr(server_args, "torch_compile_max_bs", None),
            max_running_requests=model_runner.max_running_requests,
        )

        self.t_bucketer, self.b_bucketer = build_bucketers(
            server_args, max_t=max_t, max_b=max_b
        )

    def install(self) -> TpuCompiledForwardShim:
        # Speculative decoding bypasses this prepare_forward_batch padding: the
        # draft runner calls forward(..., skip_attn_backend_init=True). The two
        # bypass sites are handled separately:
        #   * Draft decode: padded explicitly by pad_draft_decode_inputs() (this
        #     module), called before the draft attn metadata is built.
        #   * Draft extend (forward_draft_extend_after_decode): runs unpadded --
        #     its query total = sum of per-seq accept lengths varies every step,
        #     so uniform bucket padding can't pin a static shape. It recompiles
        #     per accept-total (a bounded set, served from the persistent compile
        #     cache after warmup).
        # TODO(pyc): pad draft extend to a fixed token bucket.
        existing_prep = getattr(self.model, "prepare_forward_batch", None)
        t_bucketer = self.t_bucketer
        b_bucketer = self.b_bucketer

        def _prepare(forward_batch):
            if existing_prep is not None:
                existing_prep(forward_batch)
            real_q, real_b = _pad_forward_batch_inputs(
                forward_batch, t_bucketer, b_bucketer
            )
            setattr(forward_batch, _REAL_Q_KEY, real_q)
            setattr(forward_batch, _REAL_B_KEY, real_b)

        self.model.prepare_forward_batch = _prepare

        for attr, compiled_fn in self._compiled_fns.items():
            setattr(
                self.model,
                attr,
                CompiledForwardShim(compiled_fn),
            )

        logger.info(
            "[compiled forward shim] Installed compiled-forward adaptation on %s.{%s} "
            "(t-bucketer=%s, b-bucketer=%s).",
            type(self.model).__name__,
            ", ".join(self._compiled_fns),
            self.t_bucketer.name,
            self.b_bucketer.name,
        )
        return self

    def set_compilation_manager(self, manager, model_label: str = "") -> None:
        """Propagate the compilation manager to all installed forward shims."""
        for attr in self._compiled_fns:
            shim = getattr(self.model, attr, None)
            if isinstance(shim, CompiledForwardShim):
                shim.set_compilation_manager(manager, model_label=model_label)


def build_bucketers(server_args, max_t: int, max_b: int) -> tuple[Bucketer, Bucketer]:
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
