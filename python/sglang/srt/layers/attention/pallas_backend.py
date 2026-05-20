"""Pallas ragged-paged-attention backend for the TPU torch_tpu platform.

This backend wraps the JAX/Pallas v3 ragged_paged_attention kernel
(``sglang.srt.layers.attention.pallas.ragged_paged_attention_v3``) and
exposes it as a PyTorch custom op via ``torch_tpu._internal.pallas.jax_op``
so it can be invoked from PyTorch and traced through ``torch.compile``
with ``backend="tpu"``.

Design notes
------------

* The kernel writes K/V into the fused KV cache itself. The backend therefore
  ignores ``save_kv_cache``: the cache is always updated.
* The fused KV cache must be allocated by
  :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`.
* All attention metadata (cu_q_lens, cu_kv_lens, page_indices, distribution)
  is built once per forward pass in :meth:`init_forward_metadata` and reused
  by every layer.
* Only Qwen3-dense-style models are exercised today (causal, GQA, no
  speculative decoding, no SWA). The kernel itself supports more, so
  enabling extra features is a matter of plumbing flags through the
  ``_KernelHandle`` cache key.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here.
# ``torch_tpu._internal.pallas.jax_op`` inspects the wrapped function's
# annotations at registration time (``param.annotation is jax.Array``); under
# PEP 563 those would be plain strings and validation would reject them.
# Forward-reference types used only at type-check time are spelled as
# explicit string literals (e.g. ``"RadixAttention"``).

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class PallasForwardMetadata:
    """Per-forward-pass metadata shared by every attention layer."""

    # i32[max_num_seqs]  total KV length per sequence (after appending new K/V)
    kv_lens: torch.Tensor
    # i32[max_num_seqs + 1]  cumulative sum of query lengths (starts at 0)
    cu_q_lens: torch.Tensor
    # i32[max_num_seqs + 1]  cumulative sum of page-aligned KV lengths
    cu_kv_lens: torch.Tensor
    # i32[max_num_seqs * pages_per_seq]  flat page-indices look-up table
    page_indices: torch.Tensor
    # i32[3]  (decode_end, prefill_end, mixed_end) sequence-index ranges
    distribution: torch.Tensor


# Pad the *batch* axis of the per-sequence metadata up to a single fixed
# size so torch.compile sees exactly one ``kv_lens.shape[0]`` value, never
# multiple. Mirrors torchtpu-vllm's
#   ``seq_lens = self.seq_lens_cpu[:self.num_reqs_max_model_len]``
# pattern (see tpu_runner.py:1374/1386). With multiple SEQ_BUCKETS the
# model body's compile key becomes ``(input_ids_bucket × kv_lens_bucket ×
# forward_mode)`` — a 9 × 7 × 2 = 126-cell grid — and only the small
# subset the workload happens to hit at runtime gets compiled, leaving
# stalls every time the scheduler produces a new (token, seq) tuple.
# Collapsing to one bucket makes it 9 × 1 × 2 = 18, all of which we can
# enumerate at startup. The Pallas RPA kernel already ignores entries
# beyond ``distribution[2] = num_active``, so padded slots cost nothing
# at runtime.
SEQ_BUCKETS: tuple[int, ...] = (64,)


def _next_bucket(n: int, buckets: tuple[int, ...]) -> int:
    """Smallest bucket >= n; clamps at the largest bucket (will recompile
    if a workload exceeds it — by design, so it stays loud)."""
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


_PALLAS_RPA_OP = None


# --------------------------------------------------------------------------- #
# Compiled metadata builders.
#
# Building the per-forward attention metadata (kv_lens / cu_q_lens / cu_kv_lens
# / page_indices / distribution) currently fires ~15 separate eager tensor ops
# per call. Each one becomes its own XLA dispatch on TPU, adding ~50–100µs of
# host overhead per dispatch — i.e. ~1 ms wasted per decode step. Wrapping the
# tensor-only inner computation in ``torch.compile`` collapses all of those
# into a single XLA program with a single dispatch, mirroring the way
# torchtpu-vllm wraps its post-forward steps and sglang-jax keeps the analog
# under jax.jit. We pick separate functions for decode vs extend so neither
# graph carries a Python branch on ``forward_mode``.
# --------------------------------------------------------------------------- #


def _build_decode_metadata_kernel(
    seq_lens_padded: torch.Tensor,       # [padded_bs] int32  (0 in padded slots)
    extend_seq_lens_padded: torch.Tensor, # [padded_bs] int32 (1 in real slots, 0 in padded)
    req_pool_indices_padded: torch.Tensor, # [padded_bs] int64
    req_to_token: torch.Tensor,          # [max_reqs+1, max_ctx] int32
    pages_per_seq: int,
    page_size: int,
):
    padded_bs = seq_lens_padded.shape[0]
    cu_q_lens = torch.zeros(padded_bs + 1, dtype=torch.int32, device=seq_lens_padded.device)
    cu_q_lens[1:] = torch.cumsum(extend_seq_lens_padded, dim=0, dtype=torch.int32)
    cu_kv_lens = torch.arange(
        padded_bs + 1, dtype=torch.int32, device=seq_lens_padded.device
    ) * (pages_per_seq * page_size)
    per_seq = req_to_token[req_pool_indices_padded]
    page_indices = (
        per_seq[:, : pages_per_seq * page_size : page_size] // page_size
    ).to(torch.int32)
    page_indices = page_indices.reshape(padded_bs * pages_per_seq).contiguous()
    num_active = (seq_lens_padded > 0).sum().to(torch.int32)
    distribution = num_active.repeat(3).to(torch.int32)  # decode: [N, N, N]
    return seq_lens_padded, cu_q_lens, cu_kv_lens, page_indices, distribution


def _build_extend_metadata_kernel(
    seq_lens_padded: torch.Tensor,
    extend_seq_lens_padded: torch.Tensor,
    req_pool_indices_padded: torch.Tensor,
    req_to_token: torch.Tensor,
    pages_per_seq: int,
    page_size: int,
):
    padded_bs = seq_lens_padded.shape[0]
    cu_q_lens = torch.zeros(padded_bs + 1, dtype=torch.int32, device=seq_lens_padded.device)
    cu_q_lens[1:] = torch.cumsum(extend_seq_lens_padded, dim=0, dtype=torch.int32)
    cu_kv_lens = torch.arange(
        padded_bs + 1, dtype=torch.int32, device=seq_lens_padded.device
    ) * (pages_per_seq * page_size)
    per_seq = req_to_token[req_pool_indices_padded]
    page_indices = (
        per_seq[:, : pages_per_seq * page_size : page_size] // page_size
    ).to(torch.int32)
    page_indices = page_indices.reshape(padded_bs * pages_per_seq).contiguous()
    num_active = (seq_lens_padded > 0).sum().to(torch.int32)
    zero = torch.zeros((), dtype=torch.int32, device=seq_lens_padded.device)
    distribution = torch.stack([zero, num_active, num_active]).to(torch.int32)  # [0, N, N]
    return seq_lens_padded, cu_q_lens, cu_kv_lens, page_indices, distribution


# Lazily-compiled wrappers: built on first use so unit tests that don't have
# torch_tpu installed can still import this module on CPU. The compiled
# variants are only attached when the runtime device is "tpu".
_compiled_decode_metadata = None
_compiled_extend_metadata = None


def _get_compiled_metadata_kernels(device: torch.device):
    """Return ``(decode_fn, extend_fn)`` compiled with the ``"tpu"`` backend,
    or the eager Python originals on non-TPU devices."""
    global _compiled_decode_metadata, _compiled_extend_metadata
    if device.type != "tpu":
        return _build_decode_metadata_kernel, _build_extend_metadata_kernel
    if _compiled_decode_metadata is None:
        _compiled_decode_metadata = torch.compile(
            _build_decode_metadata_kernel,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        _compiled_extend_metadata = torch.compile(
            _build_extend_metadata_kernel,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
    return _compiled_decode_metadata, _compiled_extend_metadata


def _build_pallas_kernel_op():
    """Lazily register the ragged_paged_attention kernel as a torch custom op.

    Only one custom op is registered per process; the kernel's
    Pallas-internal block-size selection is shape-aware, so shape-dependent
    specialisation happens inside the kernel rather than at the
    ``torch.library`` boundary. We override ``register_fake`` to return
    shape-matching empties without re-tracing the JAX function — this is
    what lets the dynamic token dimension flow through ``torch.compile``.
    """
    global _PALLAS_RPA_OP
    if _PALLAS_RPA_OP is not None:
        return _PALLAS_RPA_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
        ragged_paged_attention as _jax_rpa,
    )

    op_name = "sglang_tpu_pallas::ragged_paged_attention"

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        sm_scale: float = 1.0,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
    ) -> tuple[jax.Array, jax.Array]:
        return _jax_rpa(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            None,  # custom_mask
            None,  # attention_sink
            causal=1,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

    # Donate ``kv_cache_fused`` (index 3) so XLA can reuse its storage for
    # the updated cache output. Index 0 (queries) is *not* donated because
    # SGLang's ``q`` is a fresh tensor allocated by the projection — donating
    # it would invalidate the projection's output buffer.
    custom_op = _torch_pallas.jax_op(op_name, _wrapped, donate_argnums=(3,))

    def _fake_impl(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        *args,
        **kwargs,
    ):
        # The kernel returns ``(output, updated_kv_cache_fused)`` with the
        # same dtypes/shapes as ``queries`` and ``kv_cache_fused``. Returning
        # ``empty_like`` here lets ``torch.compile`` handle dynamic token
        # dimensions without invoking the (expensive, shape-specialised)
        # JAX tracer.
        return torch.empty_like(queries), torch.empty_like(kv_cache_fused)

    custom_op.register_fake(_fake_impl)

    _PALLAS_RPA_OP = custom_op
    return custom_op


class PallasAttnBackend(AttentionBackend):
    """SGLang attention backend backed by the Pallas v3 ragged-paged-attention kernel."""

    def __init__(self, model_runner: "ModelRunner") -> None:
        super().__init__()
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool

        self.forward_metadata: Optional[PallasForwardMetadata] = None
        # Register the custom op eagerly so the (Python-introspection-heavy)
        # ``jax_op`` registration does not happen inside a ``torch.compile``
        # traced region — dynamo can't handle ``inspect.signature`` on the
        # wrapped JAX function. The forward path then just looks the op up.
        self._op = _build_pallas_kernel_op()

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._build_metadata(forward_batch)

    def _build_metadata(self, forward_batch: ForwardBatch) -> PallasForwardMetadata:
        """Build per-forward-pass metadata for the Pallas kernel.

        Layout choice
        -------------
        The kernel uses ``cu_kv_lens[i]`` only as a *page-indices offset*
        (``start_kv_page_idx = cdiv(cu_kv_lens[i], page_size)``). It uses
        ``kv_lens[i]`` for the actual sequence length, and ``page_indices``
        is read as a flat 1D table of size
        ``sum_i cdiv(kv_lens[i], page_size)`` in the *ragged* world.

        We use a *rectangular* layout: every sequence reserves
        ``pages_per_seq`` slots in ``page_indices``, and we set
        ``cu_kv_lens[i] = i * pages_per_seq * page_size``. The kernel reads
        ``page_indices[i * pages_per_seq + kv_p_start]`` which is exactly
        the page for token ``kv_p_start`` of sequence ``i``.

        Batch-axis bucketing
        --------------------
        Every metadata tensor (``kv_lens``, ``cu_q_lens``, ``cu_kv_lens``,
        ``page_indices``, ``distribution``) is padded along the sequence
        axis to one of :data:`SEQ_BUCKETS`. Trailing padded sequences have
        ``seq_lens=0`` / ``extend_seq_lens=0`` and ``req_pool_indices=0``;
        because ``req_to_token[0]`` is all-zero, their page indices point at
        block 0 (the dummy-padding block of the fused KV cache). The kernel
        only loops up to ``distribution[2] = num_active`` so those padded
        slots are never visited. The padding makes the metadata shape
        constant across calls so ``torch.compile`` does not re-trace the
        attention layer per batch-size.
        """
        page_size = self.page_size
        device = forward_batch.seq_lens.device
        real_bs = int(forward_batch.seq_lens.shape[0])
        padded_bs = _next_bucket(real_bs, SEQ_BUCKETS)
        req_to_token = self.req_to_token_pool.req_to_token
        max_context_len = req_to_token.shape[1]
        pages_per_seq = max_context_len // page_size

        # Pad the per-sequence input tensors here (cheap, only 3 ops).
        seq_lens_padded = self._pad_to(forward_batch.seq_lens, padded_bs)

        is_extend = forward_batch.forward_mode.is_extend()
        if is_extend:
            assert forward_batch.extend_seq_lens is not None, (
                "extend_seq_lens is required for the Pallas attention backend in "
                "extend mode."
            )
            extend_seq_lens_padded = self._pad_to(
                forward_batch.extend_seq_lens, padded_bs
            )
        else:
            # Decode: q_len == 1 per *real* sequence, 0 for padded slots.
            extend_seq_lens_padded = torch.zeros(
                padded_bs, dtype=torch.int32, device=device
            )
            extend_seq_lens_padded[:real_bs] = 1

        req_pool_indices_padded = self._pad_to(
            forward_batch.req_pool_indices.to(torch.int64), padded_bs
        )

        # Dispatch the rest (cumsum, arange, gather, reshape, distribution
        # build) as a single compiled XLA program. On TPU this collapses ~12
        # individual eager-mode dispatches into one, which is what removes
        # the per-step host overhead we saw in xprof.
        decode_fn, extend_fn = _get_compiled_metadata_kernels(device)
        fn = extend_fn if is_extend else decode_fn
        (
            kv_lens_out,
            cu_q_lens,
            cu_kv_lens,
            page_indices,
            distribution,
        ) = fn(
            seq_lens_padded,
            extend_seq_lens_padded,
            req_pool_indices_padded,
            req_to_token,
            pages_per_seq,
            page_size,
        )

        return PallasForwardMetadata(
            kv_lens=kv_lens_out,
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            page_indices=page_indices,
            distribution=distribution,
        )

    @staticmethod
    def _pad_to(t: torch.Tensor, length: int) -> torch.Tensor:
        """Right-pad a 1D tensor with zeros to ``length``. No-op if equal."""
        n = t.shape[0]
        if n == length:
            return t.to(torch.int32) if t.dtype != torch.int32 else t
        if n > length:
            raise ValueError(
                f"Tensor with {n} entries does not fit in bucket of size {length}; "
                "increase the largest SEQ_BUCKETS entry."
            )
        out = torch.zeros(length, dtype=t.dtype, device=t.device)
        out[:n] = t
        return out.to(torch.int32) if out.dtype != torch.int32 else out

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # During torch.compile, rotary_emb may produce a 3D query; flatten so
        # the view below matches the kernel's expected layout. This mirrors
        # TorchNativeAttnBackend.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        return self._forward(q, k, v, layer, forward_batch)

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.forward_metadata is not None, (
            "Pallas attention backend was not initialised for this forward pass; "
            "did you forget to call init_forward_metadata()?"
        )

        num_tokens = q.shape[0]
        q = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(num_tokens, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(num_tokens, layer.tp_v_head_num, layer.v_head_dim)

        kv_cache = forward_batch.token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)

        op = self._op

        sliding_window = getattr(layer, "sliding_window_size", None)
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None

        soft_cap = layer.logit_cap if getattr(layer, "logit_cap", 0) else None

        out, updated_kv = op(
            q,
            k,
            v,
            kv_cache,
            self.forward_metadata.kv_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            float(layer.scaling),
            sliding_window,
            soft_cap,
        )
        # ``donate_argnums=(3,)`` lets XLA alias the input KV cache with the
        # output; in that common path the explicit ``copy_`` below is a
        # storage-aliased no-op. When XLA chooses *not* to alias (e.g. on
        # platforms / kernels where donation cannot be honoured) this copy
        # is what propagates the updated cache back into the pool. It also
        # gives dynamo a concrete in-place mutation edge between the kernel
        # output and the long-lived KV buffer.
        kv_cache.copy_(updated_kv)

        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------ #
    # Capability flags
    # ------------------------------------------------------------------ #

    def support_triton(self) -> bool:
        return False

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 0
