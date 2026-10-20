# SPDX-License-Identifier: Apache-2.0
"""Logical-to-Physical Unified KV cache pool for hybrid Full + SWA models on TPU.

Shares ``G = gcd(num_full_layers, num_swa_layers)`` physical 5D Pallas KV buffers
across both Full-Attention and Sliding-Window Attention layers with zero
external fragmentation and zero internal buffer-padding waste:

* Full layer ``i`` uses ``buf[i % G]`` at group ``g = i // G`` across
  ``M_full = num_full_layers // G`` groups.
* SWA layer ``j`` uses ``buf[j % G]`` at group ``g = j // G`` across
  ``M_swa = num_swa_layers // G`` groups.
* Each Full logical page ``p`` claims ``M_full`` arbitrary free physical page
  rows from the shared pool: ``p`` for Group ``0``, and
  ``full_group_pages[g - 1, p]`` for Group ``g in [1, M_full)``.
* Each SWA logical page ``p`` claims ``M_swa`` arbitrary free physical page rows
  from the shared pool: ``p`` for Group ``0``, and ``swa_group_pages[g - 1, p]``
  for Group ``g in [1, M_swa)``.
* Physical page ``0`` is reserved as the dummy-write sink for padded tokens.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang_tpu.kv_pool import (
    GB,
    TpuFusedMHATokenToKVPool,
    _cdiv,
    _torch_dtype_packing,
)
from sglang_tpu.unified_kv_heads import merge_swa_kv_heads, split_swa_output

logger = logging.getLogger(__name__)


def num_shared_kv_buffers(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of shared 5D buffers ``G = gcd(num_full_layers, num_swa_layers)``."""
    if num_full_layers <= 0 or num_swa_layers <= 0:
        raise ValueError(
            f"Unified KV pool requires positive full ({num_full_layers}) and "
            f"SWA ({num_swa_layers}) layer counts."
        )
    return math.gcd(num_full_layers, num_swa_layers)


def num_full_groups(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of Full groups ``M_full = num_full_layers // gcd(B, S)``."""
    return num_full_layers // num_shared_kv_buffers(num_full_layers, num_swa_layers)


def num_swa_groups(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of SWA groups ``M_swa = num_swa_layers // gcd(B, S)``."""
    return num_swa_layers // num_shared_kv_buffers(num_full_layers, num_swa_layers)


def _fused_page_shape(
    page_size: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    store_dtype: torch.dtype,
) -> Tuple[int, int, int, int]:
    """Return per-page 4D shape ``(page_size, kv_heads_packed, kv_packing, padded_head_dim)``."""
    if v_head_dim > head_dim:
        raise NotImplementedError(
            "TpuUnifiedSWAKVPool requires v_head_dim <= head_dim "
            f"(got v_head_dim={v_head_dim} > head_dim={head_dim})."
        )
    padded_head_dim = (
        _cdiv(head_dim, TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT)
        * TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
    )
    kv_packing = _torch_dtype_packing(store_dtype)
    num_kv_heads_x2 = _cdiv(head_num * 2, kv_packing) * kv_packing
    kv_heads_packed = num_kv_heads_x2 // kv_packing
    return (page_size, kv_heads_packed, kv_packing, padded_head_dim)


def resolve_unified_page_shape(
    *,
    page_size: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    store_dtype: torch.dtype,
) -> Tuple[Tuple[int, int, int, int], bool]:
    """Validate Full/SWA page compatibility and return ``(page_shape, swa_kv_heads_merged)``."""
    full_page_shape = _fused_page_shape(
        page_size, head_num, head_dim, v_head_dim, store_dtype
    )
    swa_page_shape = _fused_page_shape(
        page_size, swa_head_num, swa_head_dim, swa_v_head_dim, store_dtype
    )
    if full_page_shape == swa_page_shape:
        return full_page_shape, False

    if swa_head_num > 1 and swa_v_head_dim == swa_head_dim:
        merged_dim = swa_head_num * swa_head_dim
        merged_swa_shape = _fused_page_shape(
            page_size, 1, merged_dim, merged_dim, store_dtype
        )
        if full_page_shape == merged_swa_shape:
            return full_page_shape, True

    raise ValueError(
        "TpuUnifiedSWAKVPool requires Full and SWA layers to share an identical "
        f"per-page fused shape ( directly or after merging SWA KV heads ), "
        f"but got full={full_page_shape} (heads={head_num}, dim={head_dim}) vs "
        f"swa={swa_page_shape} (heads={swa_head_num}, dim={swa_head_dim})."
    )


@functools.cache
def _unified_pool_classes(pool_cls: type) -> Tuple[type, type]:
    """Return ``(backing_cls, view_cls)`` over ``pool_cls`` so custom pool subclasses
    keep their buffer layout and attention-facing methods."""
    if pool_cls is TpuFusedMHATokenToKVPool:
        return _UnifiedBackingPool, _SharedBufferView
    return (
        type(
            f"_UnifiedBacking{pool_cls.__name__}", (_UnifiedBackingPool, pool_cls), {}
        ),
        type(
            f"_SharedBufferView{pool_cls.__name__}", (_SharedBufferView, pool_cls), {}
        ),
    )


class _UnifiedBackingPool(TpuFusedMHATokenToKVPool):
    """Allocates the shared physical 5D KV buffers with page 0 reserved as dummy sink."""


class _SharedBufferView(TpuFusedMHATokenToKVPool):
    """Zero-allocation ``TpuFusedMHATokenToKVPool`` view over shared physical buffers."""

    def __init__(
        self,
        *,
        shared_buffers: List[torch.Tensor],
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        v_head_dim: Optional[int],
        device: str,
        enable_memory_saver: bool,
        allocation_label: str,
        layer_groups: Optional[List[int]] = None,
        group_pages: Optional[torch.Tensor] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        layer_num = len(shared_buffers)
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
        self.store_dtype = self.dtype
        self.allocation_label = allocation_label
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.kv_pack_heads = 1
        self.padded_head_dim = (
            _cdiv(self.head_dim, self.HEAD_DIM_ALIGNMENT) * self.HEAD_DIM_ALIGNMENT
        )
        self.kv_packing = _torch_dtype_packing(self.store_dtype)
        self.num_kv_heads_x2 = (
            _cdiv(self.head_num * 2, self.kv_packing) * self.kv_packing
        )
        self.num_blocks = (
            shared_buffers[0].shape[0]
            if shared_buffers
            else self.size // self.page_size + 1
        )
        self.kv_buffer = list(shared_buffers)
        self.layer_groups = (
            list(layer_groups) if layer_groups is not None else [0] * layer_num
        )
        self.group_pages = group_pages
        self._has_group_offsets = group_pages is not None and any(
            g > 0 for g in self.layer_groups
        )
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [int(np.prod(x.shape[1:])) * x.dtype.itemsize for x in self.kv_buffer],
            device=self.device,
        )
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = None
        self._kv_copy_config = None
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = True

    def _map_group_token_indices(
        self, group: int, indices_cpu: torch.Tensor
    ) -> torch.Tensor:
        if group == 0 or self.group_pages is None:
            return indices_cpu
        pages = indices_cpu // self.page_size
        offsets = indices_cpu % self.page_size
        phys_pages = self.group_pages[group - 1, pages].to(torch.long)
        return phys_pages * self.page_size + offsets

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        if not self._has_group_offsets:
            return super().get_cpu_copy(
                indices, mamba_indices=mamba_indices, req_pool_index=req_pool_index
            )
        indices_cpu = indices.to(device="cpu", dtype=torch.long)
        dev = self.kv_buffer[0].device
        self._synchronize_device()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            layer_kv = self.kv_buffer[layer_id].flatten(0, 1)
            layer_indices = self._map_group_token_indices(
                self.layer_groups[layer_id], indices_cpu
            ).to(device=dev)
            for i in range(0, len(layer_indices), chunk_size):
                chunk_indices = layer_indices[i : i + chunk_size]
                kv_cpu = layer_kv.index_select(0, chunk_indices).to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        self._synchronize_device()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        if not self._has_group_offsets:
            return super().load_cpu_copy(
                kv_cache_cpu,
                indices,
                mamba_indices=mamba_indices,
                req_pool_index=req_pool_index,
            )
        from sglang_tpu.kv_page_write import write_pages_torch

        indices = indices.to(device="cpu", dtype=torch.long)
        if indices.numel() == 0:
            return
        page = self.page_size
        blocks = indices // page
        offsets = indices % page
        uniq_blocks, block_pos = torch.unique(blocks, return_inverse=True)
        full = torch.bincount(block_pos, minlength=uniq_blocks.numel()) == page
        full_rows = full[block_pos]
        full_pos = torch.cumsum(full.long(), 0) - 1
        part_pos = torch.cumsum((~full).long(), 0) - 1
        device = self.kv_buffer[0].device
        full_uniq_cpu = uniq_blocks[full]
        part_uniq_cpu = uniq_blocks[~full]
        full_blocks = full_uniq_cpu.to(device)
        part_blocks = part_uniq_cpu.to(device)
        full_page_pos = full_pos[block_pos][full_rows]
        full_offsets = offsets[full_rows]
        part_page_pos = part_pos[block_pos][~full_rows].to(device)
        part_offsets = offsets[~full_rows].to(device)

        self._synchronize_device()
        for layer_id in range(self.layer_num):
            rows = torch.cat(kv_cache_cpu[layer_id], dim=0)
            assert rows.shape[0] == indices.numel()
            dst = self.kv_buffer[layer_id]
            group = self.layer_groups[layer_id]
            if group and self.group_pages is not None:
                layer_full_blocks = self.group_pages[group - 1, full_uniq_cpu].to(
                    dtype=torch.long, device=device
                )
                layer_part_blocks = self.group_pages[group - 1, part_uniq_cpu].to(
                    dtype=torch.long, device=device
                )
            else:
                layer_full_blocks = full_blocks
                layer_part_blocks = part_blocks
            if layer_full_blocks.numel():
                pages = torch.empty(
                    (layer_full_blocks.numel(), page, *rows.shape[1:]), dtype=rows.dtype
                )
                pages[full_page_pos, full_offsets] = rows[full_rows]
                write_pages_torch(
                    layer_full_blocks, pages.to(device, non_blocking=True), dst
                )
            if layer_part_blocks.numel():
                pages = dst[layer_part_blocks]
                pages[part_page_pos, part_offsets] = rows[~full_rows].to(
                    device, non_blocking=True
                )
                write_pages_torch(layer_part_blocks, pages, dst)
        self._synchronize_device()


class TpuUnifiedSWAKVPool(SWAKVPool):
    """Unified Hybrid Full/SWA KV pool backed by ``G = gcd(B, S)`` shared 5D buffers."""

    def __init__(
        self,
        size: int,
        size_swa: Optional[int] = None,
        *,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        device: str,
        token_to_kv_pool_class: type = TpuFusedMHATokenToKVPool,
        full_kv_pool_class: Optional[type] = None,
        swa_kv_pool_class: Optional[type] = None,
        full_kv_pool_kwargs: Optional[dict] = None,
        swa_kv_pool_kwargs: Optional[dict] = None,
        start_layer: int = 0,
        enable_memory_saver: bool = False,
        **kwargs,
    ):
        del full_kv_pool_class, swa_kv_pool_class
        del full_kv_pool_kwargs, swa_kv_pool_kwargs
        if size_swa is None:
            size_swa = size

        self.size = size
        self.size_swa = size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = start_layer
        self.page_size = page_size
        self.layer_transfer_counter = None
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None

        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        v_head_dim = kwargs.get("v_head_dim") or head_dim
        swa_head_num = kwargs.get("swa_head_num") or head_num
        swa_head_dim = kwargs.get("swa_head_dim") or head_dim
        swa_v_head_dim = kwargs.get("swa_v_head_dim") or swa_head_dim

        _, self.swa_kv_heads_merged = resolve_unified_page_shape(
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            store_dtype=dtype,
        )
        self.swa_raw_head_num = swa_head_num
        self.swa_raw_head_dim = swa_head_dim

        if self.swa_kv_heads_merged:
            swa_view_head_num = 1
            swa_view_head_dim = swa_head_num * swa_head_dim
            swa_view_v_head_dim = swa_head_num * swa_v_head_dim
        else:
            swa_view_head_num = swa_head_num
            swa_view_head_dim = swa_head_dim
            swa_view_v_head_dim = swa_v_head_dim

        self.num_shared_buffers = num_shared_kv_buffers(
            self.full_layer_nums, self.swa_layer_nums
        )
        self.num_full_groups = num_full_groups(
            self.full_layer_nums, self.swa_layer_nums
        )
        self.num_swa_groups = num_swa_groups(self.full_layer_nums, self.swa_layer_nums)
        num_blocks = size // page_size + 1
        # Logical-to-physical companion page tables for Full groups 1 .. M_full - 1
        # and SWA groups 1 .. M_swa - 1:
        self.full_group_pages = torch.zeros(
            (max(0, self.num_full_groups - 1), num_blocks),
            dtype=torch.int32,
            device="cpu",
        )
        self.swa_group_pages = torch.zeros(
            (max(0, self.num_swa_groups - 1), num_blocks),
            dtype=torch.int32,
            device="cpu",
        )

        backing_cls, view_cls = _unified_pool_classes(token_to_kv_pool_class)
        self._backing_pool = backing_cls(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            layer_num=self.num_shared_buffers,
            device=device,
            enable_memory_saver=enable_memory_saver,
            allocation_label="Unified",
        )
        backing_buffers = self._backing_pool.kv_buffer

        full_shared_buffers = [
            backing_buffers[i % self.num_shared_buffers]
            for i in range(self.full_layer_nums)
        ]
        full_layer_groups = [
            i // self.num_shared_buffers for i in range(self.full_layer_nums)
        ]
        self.full_kv_pool = view_cls(
            shared_buffers=full_shared_buffers,
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            device=device,
            enable_memory_saver=enable_memory_saver,
            allocation_label="Full",
            layer_groups=full_layer_groups,
            group_pages=self.full_group_pages if self.num_full_groups > 1 else None,
        )
        swa_shared_buffers = [
            backing_buffers[j % self.num_shared_buffers]
            for j in range(self.swa_layer_nums)
        ]
        swa_layer_groups = [
            j // self.num_shared_buffers for j in range(self.swa_layer_nums)
        ]
        self.swa_kv_pool = view_cls(
            shared_buffers=swa_shared_buffers,
            size=self.size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=swa_view_head_num,
            head_dim=swa_view_head_dim,
            v_head_dim=swa_view_v_head_dim,
            device=device,
            enable_memory_saver=enable_memory_saver,
            allocation_label="SWA",
            layer_groups=swa_layer_groups,
            group_pages=self.swa_group_pages if self.num_swa_groups > 1 else None,
        )
        # HiCache sizes host memory from these weights; both views alias the
        # shared backing buffers, so they are counted once.
        self.full_kv_pool.host_capacity_bytes = sum(
            self._backing_pool.get_kv_size_bytes()
        )
        self.swa_kv_pool.host_capacity_bytes = 0

        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        self._full_layer_group: Dict[int, int] = {}
        self._swa_layer_group: Dict[int, int] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
            self._full_layer_group[global_layer_id] = (
                full_attn_layer_id // self.num_shared_buffers
            )
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
            self._swa_layer_group[global_layer_id] = (
                swa_layer_id // self.num_shared_buffers
            )

        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            "TpuUnifiedSWAKVPool mem usage: %.2f GB, unified size: %d tokens "
            "(shared buffers=%d, full_groups=%d, swa_groups=%d, swa_heads_merged=%s, size_swa_arg=%d)",
            self.mem_usage,
            self.size,
            self.num_shared_buffers,
            self.num_full_groups,
            self.num_swa_groups,
            self.swa_kv_heads_merged,
            size_swa,
        )

    def full_page_group(self, layer_id: int) -> int:
        """Return Full group index ``g in [0, M_full)`` for ``layer_id`` (0 for SWA layers)."""
        return self._full_layer_group.get(layer_id, 0)

    def swa_page_group(self, layer_id: int) -> int:
        """Return SWA group index ``g in [0, M_swa)`` for ``layer_id`` (0 for Full layers)."""
        return self._swa_layer_group.get(layer_id, 0)

    def get_state(self) -> list[torch.Tensor]:
        """Return the ``G`` unique backing buffers (without duplicating Full/SWA views)."""
        return self._backing_pool.get_state()

    def set_state(self, state: list[torch.Tensor]) -> None:
        """Update backing buffers and refresh both Full and SWA views."""
        self._backing_pool.set_state(state)
        self.full_kv_pool.set_state(
            [state[i % self.num_shared_buffers] for i in range(self.full_layer_nums)]
        )
        self.swa_kv_pool.set_state(
            [state[j % self.num_shared_buffers] for j in range(self.swa_layer_nums)]
        )

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        """Report physical KV bytes once from the backing pool."""
        return self._backing_pool.get_kv_size_bytes()

    def get_state_buf_infos(self):
        """SWA shares the physical buffers already returned by ``get_contiguous_buf_infos``."""
        return [], [], []
