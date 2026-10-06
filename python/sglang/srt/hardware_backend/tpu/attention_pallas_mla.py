# SPDX-License-Identifier: Apache-2.0
"""Pallas MLA (Multi-head Latent Attention) backend for the TPU torch_tpu platform.

Wraps the JAX/Pallas MLA v2 ragged-paged-attention kernel
(``sglang.srt.layers.attention.pallas.mla_ragged_paged_attention_v2``, vendored
from sglang-jax, itself adapted from vllm/tpu-inference) and exposes it as a
PyTorch custom op, mirroring ``attention_pallas_rpa_v3``. Selected via
``--attention-backend pallas_tpu_mla``; used by MLA models (DeepSeek V2/V3).

Design notes
------------

* Only the *absorbed* MLA path is supported: the DeepSeek dispatch handler for
  this backend returns ``AttnForwardMethod.MLA`` for every forward mode, and
  ``pallas_tpu_mla`` is registered in ``FORWARD_ABSORB_CORE_ATTENTION_BACKENDS``
  so ``forward_absorb_core`` calls ``attn_mqa`` in the split-rope form::

      attn_mqa(q_nope_out,  # [T, H, kv_lora_rank]     (q_nope · W_UK)
               k_nope,      # [T, 1, kv_lora_rank]     (latent, post layernorm)
               k_nope,      # v (ignored; same latent)
               forward_batch,
               q_rope=q_pe, # [T, H, qk_rope_head_dim] (post rotary)
               k_rope=k_pe) # [T, 1, qk_rope_head_dim] (post rotary)

  which maps 1:1 onto the kernel inputs (ql_nope, q_pe, new_kv_c, new_k_pe).
  The kernel attends over the latent cache (MQA) and returns o_latent
  ``[T, H, kv_lora_rank]``; the model folds W_UV into it afterwards.
* The kernel writes the new latent KV into the paged cache itself (the cache
  is donated to the pallas call), so ``save_kv_cache`` is ignored and the pool
  (:class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMLATokenToKVPool`)
  rejects out-of-band writes.
* All attention metadata is identical to the RPA v3 backend's (the MLA v2
  kernel was adapted to the same ragged ``cu_kv_lens``-offset page-indices
  addressing), so this class reuses ``TpuPallasRpaV3AttnBackend``'s
  ``init_forward_metadata`` machinery wholesale via inheritance.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    _quantize_kv_to_fp8,
)
from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
    TpuPallasRpaV3AttnBackend,
)
from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

_PALLAS_MLA_OP = None

_MLA_DECODE_BATCH_SIZE = 4


def _build_pallas_mla_kernel_op():
    """Lazily register the MLA v2 kernel as a torch custom op.

    Mirrors ``attention_pallas_rpa_v3._build_pallas_kernel_op``: one custom op
    per process; block-size selection is shape-aware inside the kernel; the
    fake impl returns a shape-matching empty so the dynamic token dimension
    flows through ``torch.compile``.
    """
    global _PALLAS_MLA_OP
    if _PALLAS_MLA_OP is not None:
        return _PALLAS_MLA_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.layers.attention.pallas.mla_ragged_paged_attention_v2 import (
        mla_ragged_paged_attention as _jax_mla,
    )

    op_name = "sglang_tpu_pallas::mla_ragged_paged_attention"

    def _wrapped(
        ql_nope: jax.Array,
        q_pe: jax.Array,
        new_kv_c: jax.Array,
        new_k_pe: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        sm_scale: float = 1.0,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> tuple[jax.Array, jax.Array]:
        return _jax_mla(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            decode_batch_size=_MLA_DECODE_BATCH_SIZE,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(10, 11, 12),
        donate_argnums=(4,),
        input_output_aliases={4: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("kv_cache_fused",),
        device_types=["tpu"],
    )
    def custom_op(
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        new_kv_c: torch.Tensor,
        new_k_pe: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        sm_scale: float = 1.0,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> torch.Tensor:
        # JAX is CPU-only here, so Mosaic needs the TPU generation patched in.
        # Inside the custom op like the RPA base: Dynamo can't trace the patch's lock.
        with tpu_info_for_cpu_export():
            output, _updated_kv = jax_kernel(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                sm_scale,
                k_scale,
                v_scale,
            )
        return output

    def _fake_impl(
        ql_nope: torch.Tensor,
        *args,
        **kwargs,
    ):
        return torch.empty_like(ql_nope)

    custom_op.register_fake(_fake_impl)

    _PALLAS_MLA_OP = custom_op
    return custom_op


class TpuPallasMlaAttnBackend(TpuPallasRpaV3AttnBackend):
    """SGLang attention backend backed by the Pallas MLA v2 kernel.

    Inherits the full per-forward metadata machinery (kv_lens / cu_q_lens /
    cu_kv_lens / page_indices / distribution) from the RPA v3 backend — the
    MLA kernel consumes the identical layout — and overrides only the op
    registration and the per-layer forward.
    """

    def _build_kernel_op(self):
        # Hook called by ``super().__init__()`` to populate ``self._op``. Both
        # forward modes go through ``_forward_mla``, so register the MLA v2
        # kernel op here — the parent never builds the unused RPA v3 op.
        return _build_pallas_mla_kernel_op()

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_rpa_distribution_cpu(is_decode: bool, real_bs: int) -> torch.Tensor:
        """MLA v2 ``distribution`` semantics differ from RPA v3 for extend.

        The kernel runs three passes over ``(i, j, k) = distribution``:
        decode ``[0:i]``, prefill-only ``[i:j]`` and mixed ``[j:k]`` — but its
        PREFILL pass is dead code (kernel.py TODO), so extend batches must be
        routed through MIXED: ``[0, 0, N]`` (sglang-jax mla_backend.py does
        the same). The parent's RPA v3 shape ``[0, N, N]`` would leave every
        prefill sequence unprocessed — attention output uninitialised and the
        latent cache never written. Decode is ``[N, N, N]`` in both kernels.
        """
        num_active = torch.tensor(real_bs, dtype=torch.int32)
        if is_decode:
            return num_active.repeat(3)
        zero = torch.zeros((), dtype=torch.int32)
        return torch.stack([zero, zero, num_active])

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
        return self._forward_mla(q, k, layer, forward_batch, **kwargs)

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
        return self._forward_mla(q, k, layer, forward_batch, **kwargs)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self.forward_metadata.dsa_token_slots = self._build_dsa_token_slots(
            forward_batch
        )
        if envs.SGLANG_TPU_DSA_INDEXER_KERNEL.get():
            from sglang.srt.hardware_backend.tpu.dsa_indexer_tpu import (
                register_dsa_indexer_ops,
            )

            register_dsa_indexer_ops()

    def _build_dsa_token_slots(self, forward_batch: ForwardBatch):
        """i32 [bs, max_context]: token slot of position s of each request, on device.

        Built here, like the page table, because req_to_token is host-resident
        on TPU and cannot be indexed inside the compiled forward.
        """
        from sglang.srt.hardware_backend.tpu.kv_pool import (
            TpuFusedDSAMLATokenToKVPool,
        )

        if not isinstance(self.token_to_kv_pool, TpuFusedDSAMLATokenToKVPool):
            return None
        if not forward_batch.forward_mode.is_decode():
            return None
        # Only the token-exact gather and the torch indexer read the slots.
        if (
            envs.SGLANG_TPU_DSA_PAGE_LEVEL.get()
            and envs.SGLANG_TPU_DSA_INDEXER_KERNEL.get()
        ):
            return None
        req_to_token = self.req_to_token_pool.req_to_token
        if req_to_token.device.type == "cpu":
            req = forward_batch.req_pool_indices_cpu.to(torch.int64)
        else:
            req = forward_batch.req_pool_indices.to(torch.int64)
        slots = req_to_token[req].to(torch.int32)
        return self._value_metadata(
            name="dsa_token_slots", value=slots, device=forward_batch.seq_lens.device
        )

    def _forward_dsa_decode(
        self,
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        kv_cache,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Token-exact DSA decode: attend only the indexer's top-k positions.

        The MLA kernel still owns the cache write; it runs over the tail page
        only. The selected latent rows (which include the new token, since
        the indexer scores its own key) are then gathered and attended here.
        """
        md = self.forward_metadata
        if envs.SGLANG_TPU_DSA_PAGE_LEVEL.get():
            page_lens, pages = _dsa_page_level_metadata(
                topk_indices, md.kv_lens, md.page_indices, self.page_size
            )
            return self._op(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                kv_cache,
                page_lens,
                pages,
                md.cu_q_lens,
                md.cu_kv_lens,
                md.distribution,
                float(layer.scaling),
                None,
                None,
            )
        tail_lens, tail_pages = _dsa_tail_page_metadata(
            md.kv_lens, md.page_indices, self.page_size
        )
        self._op(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache,
            tail_lens,
            tail_pages,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
            float(layer.scaling),
            None,
            None,
        )
        slots = md.dsa_token_slots.gather(1, topk_indices.clamp(min=0).to(torch.int64))
        # Gather from the contiguous padded rows, then trim: indexing the
        # trimmed (strided) view materializes the whole layer cache first.
        buf = self.token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)
        rows = buf.view(-1, buf.shape[-1]).index_select(0, slots.reshape(-1))
        rows = rows[:, : self.token_to_kv_pool.kv_cache_dim].view(
            slots.shape[0], slots.shape[1], -1
        )
        q = torch.cat([ql_nope, q_pe], dim=-1)
        out = dsa_sparse_attention(
            q, rows, topk_indices, float(layer.scaling), ql_nope.shape[-1]
        )
        return out.to(ql_nope.dtype)

    def _forward_mla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert self.forward_metadata is not None, (
            "Pallas MLA attention backend was not initialised for this forward "
            "pass; did you forget to call init_forward_metadata()?"
        )
        assert q_rope is not None and k_rope is not None, (
            "pallas_tpu_mla requires the split-rope absorbed call form "
            "(q_rope=/k_rope= kwargs). Is 'pallas_tpu_mla' registered in "
            "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS?"
        )

        # For the MLA attn_mqa layer: v_head_dim == kv_lora_rank, and
        # qk_head_dim == kv_lora_rank + qk_rope_head_dim.
        kv_lora_rank = layer.v_head_dim
        rope_dim = layer.qk_head_dim - layer.v_head_dim
        num_tokens = q.shape[0]

        # q arrives as [T, H, kv_lora_rank] from forward_absorb_core but is
        # NON-contiguous (bmm(...).transpose(0, 1)); k / k_rope are slices of
        # latent_cache ([T, 1, kv_lora_rank] / [T, 1, rope_dim], row stride
        # 576). ``reshape`` to the same logical shape keeps those strides, and
        # the torch->JAX pallas bridge hands the raw buffer to the kernel, so
        # every input must be made dense explicitly or the kernel reads
        # scrambled data.
        ql_nope = q.reshape(num_tokens, layer.tp_q_head_num, kv_lora_rank).contiguous()
        q_pe = q_rope.reshape(num_tokens, layer.tp_q_head_num, rope_dim).contiguous()
        new_kv_c = k.reshape(num_tokens, kv_lora_rank).contiguous()
        new_k_pe = k_rope.reshape(num_tokens, rope_dim).contiguous()

        # fp8 KV cache: the kernel writes new_kv_c / new_k_pe into the cache
        # itself and requires them in the cache dtype. Quantize per-tensor with
        # the layer's k_scale (1.0 when the checkpoint has none) and pass the
        # same scale for K and V: MLA's V is the same latent tensor, so it
        # must be dequantized with the scale it was quantized with. Q stays
        # bf16 (the combination upstream tests).
        if self.kv_cache_quant_dtype is not None:
            k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
            v_scale = k_scale
            new_kv_c = _quantize_kv_to_fp8(new_kv_c, k_scale, self.kv_cache_quant_dtype)
            new_k_pe = _quantize_kv_to_fp8(new_k_pe, k_scale, self.kv_cache_quant_dtype)
        else:
            k_scale = None
            v_scale = None

        # Backend-cached pool: upstream moved token_to_kv_pool off ForwardBatch.
        kv_cache = self.token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)

        topk_indices = kwargs.get("topk_indices")
        if topk_indices is not None and forward_batch.forward_mode.is_decode():
            if self.kv_cache_quant_dtype is not None:
                raise NotImplementedError("TPU DSA decode needs a bf16 KV cache")
            out = self._forward_dsa_decode(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                kv_cache,
                layer,
                forward_batch,
                topk_indices,
            )
            return out.reshape(num_tokens, layer.tp_q_head_num * kv_lora_rank)

        out = self._op(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache,
            self.forward_metadata.kv_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            float(layer.scaling),
            k_scale,
            v_scale,
        )

        # o_latent [T, H, kv_lora_rank] -> [T, H * kv_lora_rank]; the model
        # views it back to 3D and folds W_UV.
        return out.reshape(num_tokens, layer.tp_q_head_num * kv_lora_rank)


def _dsa_page_level_metadata(topk_indices, kv_lens, page_indices, page_size):
    """Page table of only the pages holding top-k positions, newest page last.

    Ported from sglang-jax ``sparse_mla_page_level``: the dense MLA kernel then
    attends whole selected pages (a superset of the top-k) and its in-place
    write of the new latent lands on the true slot of the last listed page.
    """
    num_seqs = kv_lens.shape[0]
    if topk_indices.shape[0] != num_seqs:
        raise ValueError("page-level DSA decode expects one query per sequence")
    table = page_indices.view(num_seqs, -1)
    pages_per_seq = table.shape[1]
    page_ids = torch.arange(pages_per_seq, device=table.device)
    newest = (kv_lens.to(torch.int64) - 1).clamp(min=0)
    new_page = newest // page_size
    picked = torch.where(
        topk_indices >= 0, topk_indices.to(torch.int64) // page_size, pages_per_seq
    )
    hit = torch.zeros(
        (num_seqs, pages_per_seq + 1), dtype=torch.int32, device=table.device
    ).scatter(1, picked, 1)[:, :pages_per_seq]
    hit = hit.bool() & (page_ids[None, :] != new_page[:, None])
    num_hit = hit.sum(dim=1)
    hit_pages = torch.where(hit, page_ids[None, :], pages_per_seq).sort(dim=1).values
    local = torch.where(
        page_ids[None, :] < num_hit[:, None], hit_pages, new_page[:, None]
    ).clamp(max=pages_per_seq - 1)
    lens = torch.where(
        kv_lens > 0,
        (num_hit * page_size + newest % page_size + 1).to(kv_lens.dtype),
        kv_lens,
    )
    return lens, table.gather(1, local).reshape(-1).contiguous()


def _dsa_tail_page_metadata(kv_lens, page_indices, page_size):
    """Metadata that points each sequence at only the page holding its newest token.

    The MLA kernel then writes the new latent in place (donated cache) and
    touches at most one page; its attention output is not used.
    """
    num_seqs = kv_lens.shape[0]
    table = page_indices.view(num_seqs, -1)
    last = (kv_lens.to(torch.int64) - 1).clamp(min=0)
    tail_page = table.gather(1, (last // page_size).unsqueeze(1))
    table = torch.cat([tail_page, table[:, 1:]], dim=1)
    tail_lens = torch.where(
        kv_lens > 0, (last % page_size + 1).to(kv_lens.dtype), kv_lens
    )
    return tail_lens, table.reshape(-1).contiguous()


def dsa_sparse_attention(q, latent_rows, topk_indices, sm_scale, kv_lora_rank):
    """Attention of each query over its own selected latent rows.

    q [B, H, kv_lora_rank + rope], latent_rows [B, K, kv_lora_rank + rope],
    topk_indices [B, K] (-1 = not selected). Returns [B, H, kv_lora_rank] fp32.
    """
    scores = torch.einsum("bhd,bkd->bhk", q.float(), latent_rows.float()) * sm_scale
    selected = (topk_indices >= 0).unsqueeze(1)
    # Finite floor, not -inf: padded rows select nothing and must not give NaN.
    scores = torch.where(selected, scores, torch.full_like(scores, -1e30))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhk,bkd->bhd", probs, latent_rows[..., :kv_lora_rank].float())
