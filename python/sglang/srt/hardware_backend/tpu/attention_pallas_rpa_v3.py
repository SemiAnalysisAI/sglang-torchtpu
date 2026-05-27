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


_PALLAS_RPA_OP = None


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
        custom_mask: Optional[jax.Array] = None,
        sm_scale: float = 1.0,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
        causal: int = 1,
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
            custom_mask,
            None,  # attention_sink
            causal=causal,
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


class TpuPallasRpaV3AttnBackend(AttentionBackend):
    """SGLang attention backend backed by the Pallas v3 ragged-paged-attention kernel.

    Sibling to :class:`TpuPallasAttnBackend` (``attention_pallas.py``), which
    wraps the Pallas flash + paged-attention kernels. Selected via
    ``--attention-backend pallas_tpu_rpa_v3`` (the TPU platform default).
    Requires the fused KV layout supplied by
    :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`;
    :func:`patches._patch_mha_kv_pool_for_rpa_v3` swaps that pool in when this
    backend is active.
    """

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
        """
        page_size = self.page_size
        device = forward_batch.seq_lens.device
        bs = int(forward_batch.seq_lens.shape[0])
        req_to_token = self.req_to_token_pool.req_to_token
        max_context_len = req_to_token.shape[1]
        pages_per_seq = max_context_len // page_size

        seq_lens = forward_batch.seq_lens.to(torch.int32)

        if forward_batch.forward_mode.is_extend():
            assert forward_batch.extend_seq_lens is not None, (
                "extend_seq_lens is required for the Pallas attention backend in "
                "extend mode."
            )
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
        else:
            # Decode: q_len == 1 per sequence.
            extend_seq_lens = torch.ones(bs, dtype=torch.int32, device=device)

        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)

        cu_q_lens = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        cu_q_lens[1:] = torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)

        # cu_kv_lens[i] = i * pages_per_seq * page_size so the kernel addresses
        # ``page_indices[i * pages_per_seq + j]`` for token j of seq i.
        cu_kv_lens = torch.arange(bs + 1, dtype=torch.int32, device=device) * (
            pages_per_seq * page_size
        )

        per_seq = req_to_token[req_pool_indices]  # [bs, max_ctx]
        # ``req_to_token.shape[1]`` may not be an exact multiple of
        # ``page_size`` (SGLang adds extra slack tokens), in which case
        # ``per_seq[:, ::page_size]`` would emit one extra trailing column.
        # Trim to exactly ``pages_per_seq`` entries per row so the flat
        # ``page_indices`` shape matches the rectangular layout the kernel
        # expects.
        page_indices = (
            per_seq[:, : pages_per_seq * page_size : page_size] // page_size
        ).to(torch.int32)
        page_indices = page_indices.reshape(bs * pages_per_seq).contiguous()

        num_active = torch.full((), bs, dtype=torch.int32, device=device)
        if forward_batch.forward_mode.is_decode():
            distribution = num_active.repeat(3).to(torch.int32)
        else:
            zero = torch.zeros((), dtype=torch.int32, device=device)
            distribution = torch.stack([zero, num_active, num_active]).to(torch.int32)

        return PallasForwardMetadata(
            kv_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            page_indices=page_indices,
            distribution=distribution,
        )

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

        # Spec-decoding verify step: forward_batch.spec_info.custom_mask carries
        # the tree mask in the [pattern_total_kv_len] layout the kernel expects
        # (per-sequence q_len*kv_len row-major, matching the FULL_MASK output
        # of build_tree_kernel_efficient). Cast bool -> int32 so the kernel's
        # `< 1` test reads 1=keep / 0=mask.
        custom_mask = None
        causal = 1
        spec_info = getattr(forward_batch, "spec_info", None)
        cm = getattr(spec_info, "custom_mask", None) if spec_info is not None else None
        if cm is not None and cm.numel() > 0:
            custom_mask = cm.to(torch.int32) if cm.dtype == torch.bool else cm
            causal = 0

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
            custom_mask,
            float(layer.scaling),
            sliding_window,
            soft_cap,
            causal,
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
