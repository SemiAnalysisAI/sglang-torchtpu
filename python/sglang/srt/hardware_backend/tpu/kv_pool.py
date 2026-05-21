"""TPU fused KV cache pool used by the Pallas ragged-paged-attention backend.

The Pallas v3 kernel expects KV cache laid out as a single 5D buffer per
layer with shape::

    [num_blocks, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]

where ``num_kv_heads_x2`` is ``2 * num_kv_heads`` (K and V interleaved) and
``kv_packing = 32 // bits_per_dtype_elem`` is the number of dtype elements
that fit in a 32-bit lane (e.g. 2 for bfloat16). The kernel performs the
KV-cache write itself during attention, so SGLang's ``set_kv_buffer`` is a
no-op on the pallas path — calling it would only undo the kernel's writes
when ``save_kv_cache=True``.

To stay compatible with the rest of SGLang (radix cache, HiCache, etc.)
the pool still exposes ``get_key_buffer`` / ``get_value_buffer`` as logical
views into the fused buffer; the contiguous storage is the 5D fused
buffer.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, get_tensor_size_bytes

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def _torch_dtype_packing(dtype: torch.dtype, packing_bits: int = 32) -> int:
    """Number of ``dtype`` elements that fit in a ``packing_bits``-bit lane."""
    bits = torch.empty((), dtype=dtype).element_size() * 8
    if packing_bits % bits != 0:
        raise ValueError(
            f"dtype bitwidth must divide {packing_bits}, but got {bits} for {dtype}."
        )
    return packing_bits // bits


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class TpuFusedMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool with the fused 5D layout consumed by the Pallas kernel.

    Only the ``num_blocks * page_size`` axis is addressable by ``loc``; we
    keep an additional pad of one ``page_size`` so that padded tokens
    (``loc == 0``) write into a dummy page, mirroring the upstream MHA pool
    behaviour.
    """

    HEAD_DIM_ALIGNMENT = 128

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        # We deliberately do *not* call ``MHATokenToKVPool.__init__`` here:
        # its ``_create_buffers`` allocates the legacy two-buffer (K / V) layout
        # which is incompatible with the Pallas kernel. Instead we initialize
        # the ``KVCache`` base directly, then build the fused buffers ourselves.
        from sglang.srt.mem_cache.memory_pool import KVCache  # local import to avoid cycles

        KVCache.__init__(
            self,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )
        if self.head_dim != self.v_head_dim:
            raise NotImplementedError(
                "TpuFusedMHATokenToKVPool requires head_dim == v_head_dim "
                f"(got {self.head_dim} vs {self.v_head_dim}). Asymmetric K/V "
                "head dims are not supported by the Pallas kernel."
            )

        self.padded_head_dim = _cdiv(self.head_dim, self.HEAD_DIM_ALIGNMENT) * self.HEAD_DIM_ALIGNMENT
        self.kv_packing = _torch_dtype_packing(self.store_dtype)
        # K and V are interleaved along the kv-heads axis, then packed in 32-bit lanes.
        self.num_kv_heads_x2 = _cdiv(self.head_num * 2, self.kv_packing) * self.kv_packing

        if self.size % self.page_size != 0:
            raise ValueError(
                f"size ({self.size}) must be divisible by page_size ({self.page_size}) "
                "for the TPU fused KV pool."
            )
        # Pad with one extra page so loc == 0 (the padded-token slot) is harmless.
        self.num_blocks = self.size // self.page_size + 1

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        # No alt_stream on TPU.
        self.alt_stream = None
        self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # Bookkeeping used by store-cache helpers; the fused buffer keeps a
        # single contiguous row per slot of width ``num_kv_heads_x2 * head_dim``.
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = True

    # ------------------------------------------------------------------ #
    # Buffer creation / accessors
    # ------------------------------------------------------------------ #

    def _create_buffers(self):
        kv_heads_packed = self.num_kv_heads_x2 // self.kv_packing
        shape = (
            self.num_blocks,
            self.page_size,
            kv_heads_packed,
            self.kv_packing,
            self.padded_head_dim,
        )
        logger.info(
            "Allocating fused TPU KV buffers: layers=%d, shape per layer=%s, dtype=%s",
            self.layer_num,
            shape,
            self.store_dtype,
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    torch.zeros(shape, dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]

        # Mirrors used by the rest of SGLang. These are recomputed lazily; the
        # fused buffer is the source of truth.
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [int(np.prod(x.shape[1:])) * x.dtype.itemsize for x in self.kv_buffer],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        size_bytes = 0
        for buf in self.kv_buffer:
            size_bytes += get_tensor_size_bytes(buf)
        # Report as a (k_size, v_size) tuple for compatibility with the
        # base-class accounting; the buffers are fused so we split evenly.
        return size_bytes // 2, size_bytes // 2

    # ``get_fused_kv_buffer`` is the canonical accessor on TPU; the Pallas
    # backend uses it directly.
    def get_fused_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_buffer[layer_id - self.start_layer]

    def _logical_kv_view(self, layer_id: int) -> torch.Tensor:
        """Return a logical [num_tokens, num_kv_heads*2, head_dim] view.

        Used by HiCache / radix cache helpers that want to read K and V back
        out separately. Note the head dim has padding to a multiple of 128.
        """
        buf = self.kv_buffer[layer_id - self.start_layer]
        n_blocks, page_size, packed, packing, hd = buf.shape
        return buf.view(n_blocks * page_size, packed * packing, hd)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        view = self._logical_kv_view(layer_id)
        # K heads are the even indices of the interleaved KV axis.
        return view[:, 0 : 2 * self.head_num : 2, : self.head_dim]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        view = self._logical_kv_view(layer_id)
        return view[:, 1 : 2 * self.head_num : 2, : self.head_dim]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def _get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_value_buffer(layer_id)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """No-op on the Pallas path.

        The Pallas ragged-paged-attention kernel writes K and V into the fused
        buffer itself during the attention step. The attention backend is
        responsible for clearing ``save_kv_cache`` so this method is never
        called for normal forward passes; the override here is defensive and
        also covers backfill paths (e.g. radix cache replay) which still
        invoke ``set_kv_buffer`` explicitly.
        """
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if cache_k is None:
            return

        # Convert separate (K, V) → fused interleaved layout and scatter into
        # the fused buffer along the logical token axis. Used by radix-cache
        # backfill; not hit in the steady-state forward.
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k.div(k_scale)
            if v_scale is not None:
                cache_v = cache_v.div(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        view = self._logical_kv_view(layer_id)
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.head_dim)
        # Pad head dim out to padded_head_dim if needed.
        if self.padded_head_dim != self.head_dim:
            pad = self.padded_head_dim - self.head_dim
            cache_k = torch.nn.functional.pad(cache_k, (0, pad))
            cache_v = torch.nn.functional.pad(cache_v, (0, pad))
        # Interleave K and V along the heads axis: [..., K0, V0, K1, V1, ...].
        n_tokens = cache_k.shape[0]
        fused = torch.empty(
            (n_tokens, 2 * self.head_num, self.padded_head_dim),
            dtype=cache_k.dtype,
            device=cache_k.device,
        )
        fused[:, 0::2] = cache_k
        fused[:, 1::2] = cache_v
        # Pad the kv-heads axis to ``num_kv_heads_x2`` if the model has an
        # odd-shape that requires packing alignment.
        if fused.shape[1] != self.num_kv_heads_x2:
            extra = self.num_kv_heads_x2 - fused.shape[1]
            fused = torch.nn.functional.pad(fused, (0, 0, 0, extra))
        view[loc] = fused

    # ------------------------------------------------------------------ #
    # Misc compatibility
    # ------------------------------------------------------------------ #

    def get_contiguous_buf_infos(self):
        kv_data_ptrs = [buf.data_ptr() for buf in self.kv_buffer]
        kv_data_lens = [buf.nbytes for buf in self.kv_buffer]
        kv_item_lens = [buf[0].nbytes * self.page_size for buf in self.kv_buffer]
        return kv_data_ptrs, kv_data_lens, kv_item_lens
