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

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.runtime_context import get_disagg
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_free_fence import HeldPages
from sglang_tpu.kv_pool import (
    GB,
    TpuFusedMHATokenToKVPool,
    _cdiv,
    _torch_dtype_packing,
)
from sglang_tpu.unified_kv_heads import merge_swa_kv_heads, split_swa_output

logger = logging.getLogger(__name__)

_EVICTION_POLICIES = ("full_first", "swa_first", "static_shadow", "compact")


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


class _SharedPageStore:
    """Shared CPU physical page pool (1 .. num_pages) drawn from by Full and SWA with zero fragmentation."""

    def __init__(self):
        self.free_pages: Optional[torch.Tensor] = None
        self.page_is_free: Optional[torch.Tensor] = None
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0
        self.held_pages = HeldPages()

    def unheld_free_pages(self) -> int:
        return (
            0 if self.free_pages is None else len(self.free_pages)
        ) + self.num_staged_pages

    def num_free_pages(self) -> int:
        return self.unheld_free_pages() + self.held_pages.num_pages


class _SharedPagesAllocator(TpuPagedTokenToKVPoolAllocator):
    """One side (Full or SWA) of the zero-fragmentation Logical-to-Physical unified allocator.

    Both sides share a single flat ``free_pages`` list in ``_SharedPageStore``.
    - Full Attention claims 1 physical page ID per logical page.
    - SWA claims 1 primary physical page ID ``p`` (for Group 0) plus ``M - 1``
      arbitrary companion physical page IDs (for Groups ``1 .. M - 1``), recorded
      in ``group_pages[:, p]`` and released together when ``p`` is freed.
    """

    def __init__(
        self,
        *,
        page_store: _SharedPageStore,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
        group_pages: Optional[torch.Tensor] = None,
    ):
        self._page_store = page_store
        self.allocated_pages = 0
        self._group_pages = group_pages
        self._pages_per_id = 1 if group_pages is None else group_pages.shape[0] + 1
        self._free_pages_before_take: Optional[torch.Tensor] = None
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

    @property
    def free_pages(self) -> Optional[torch.Tensor]:
        return self._page_store.free_pages

    @free_pages.setter
    def free_pages(self, value: Optional[torch.Tensor]) -> None:
        self._page_store.free_pages = value

    @property
    def staged_pages(self) -> list[torch.Tensor]:
        return self._page_store.staged_pages

    @staged_pages.setter
    def staged_pages(self, value: list[torch.Tensor]) -> None:
        self._page_store.staged_pages = value

    @property
    def num_staged_pages(self) -> int:
        return self._page_store.num_staged_pages

    @num_staged_pages.setter
    def num_staged_pages(self, value: int) -> None:
        self._page_store.num_staged_pages = value

    @property
    def held_pages(self) -> HeldPages:
        return self._page_store.held_pages

    @held_pages.setter
    def held_pages(self, value: HeldPages) -> None:
        self._page_store.held_pages = value

    @property
    def page_is_free(self) -> Optional[torch.Tensor]:
        return self._page_store.page_is_free

    @page_is_free.setter
    def page_is_free(self, value: Optional[torch.Tensor]) -> None:
        self._page_store.page_is_free = value

    def available_size(self) -> int:
        return (
            self._page_store.num_free_pages() // self._pages_per_id
        ) * self.page_size

    def allocated_count(self) -> int:
        return (self.allocated_pages // self._pages_per_id) * self.page_size

    def alloc(self, need_size: int):
        return self._count_taken_pages(super().alloc, need_size)

    def alloc_extend(self, *args, **kwargs):
        return self._count_taken_pages(super().alloc_extend, *args, **kwargs)

    def alloc_decode(self, *args, **kwargs):
        return self._count_taken_pages(super().alloc_decode, *args, **kwargs)

    def _count_taken_pages(self, take, *args, **kwargs):
        free_before = self._page_store.num_free_pages()
        out = take(*args, **kwargs)
        if self._group_pages is not None:
            self._take_group_pages()
        self.allocated_pages += free_before - self._page_store.num_free_pages()
        return out

    def _ensure_free_pages(self, num_pages: int) -> bool:
        if self._group_pages is None:
            return super()._ensure_free_pages(num_pages)
        ok = super()._ensure_free_pages(num_pages * self._pages_per_id)
        self._free_pages_before_take = self.free_pages if ok else None
        return ok

    def _take_group_pages(self) -> None:
        before = self._free_pages_before_take
        self._free_pages_before_take = None
        if before is None:
            return
        num_taken = len(before) - len(self.free_pages)
        if num_taken == 0:
            return
        num_extra = self._group_pages.shape[0] * num_taken
        extra = self.free_pages[:num_extra]
        self._mark_handed_out(extra)
        self._group_pages[:, before[:num_taken].long()] = extra.view(-1, num_taken)
        self.free_pages = self.free_pages[num_extra:]

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        page_ids = tuple(
            ids[ids > 0]
            for ids in (
                p.reshape(-1).to(device="cpu", dtype=torch.int32) for p in page_ids
            )
        )
        if self._group_pages is not None and page_ids:
            ids = torch.cat([p.to(torch.int64) for p in page_ids])
            if ids.numel() > 0:
                page_ids = (*page_ids, self._group_pages[:, ids].reshape(-1))
        total_released = sum(ids.numel() for ids in page_ids)
        if total_released == 0:
            return
        self.allocated_pages -= total_released
        super()._release_page_ids(*page_ids)

    def clear(self) -> None:
        super().clear()
        self.allocated_pages = 0


class TpuUnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Logical-to-Physical Unified KV allocator for Full + SWA attention on TPU.

    Both Full and SWA allocate from a single shared physical page list
    (``_SharedPageStore``). A Full logical page consumes ``M_full = num_full_groups``
    arbitrary physical page IDs (recorded in ``kvcache.full_group_pages``);
    an SWA logical page consumes ``M_swa = num_swa_groups`` arbitrary physical page
    IDs (recorded in ``kvcache.swa_group_pages``), eliminating both external and
    internal fragmentation.
    """

    def __init__(
        self,
        size: int,
        size_swa: Optional[int] = None,
        *,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: TpuUnifiedSWAKVPool,
        need_sort: bool,
        req_to_token_pool=None,
        static_full_tokens: Optional[int] = None,
    ):
        del size_swa
        if page_size <= 1:
            raise ValueError(
                f"TpuUnifiedSWATokenToKVPoolAllocator requires page_size > 1, got {page_size}."
            )
        assert isinstance(kvcache, TpuUnifiedSWAKVPool)
        self._page_store = _SharedPageStore()
        self._full_groups = getattr(kvcache, "num_full_groups", 1)
        self._swa_groups = getattr(kvcache, "num_swa_groups", 1)
        self.need_sort = need_sort
        self._kvcache = kvcache
        self._req_to_token_pool = req_to_token_pool
        super().__init__(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
            req_to_token_pool=req_to_token_pool,
        )
        assert not self._swa_req_ring
        self._eviction_policy, self._full_share, self._swa_share = self._eviction_setup(
            kvcache, size // page_size, static_full_tokens
        )
        self._compact_retain_tokens = (
            tpu_envs.SGLANG_TPU_UNIFIED_KV_COMPACT_RETAIN_TOKENS.get()
        )
        self.full_attn_allocator = _SharedPagesAllocator(
            page_store=self._page_store,
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache.full_kv_pool,
            need_sort=need_sort,
            group_pages=kvcache.full_group_pages if self._full_groups > 1 else None,
        )
        self.swa_attn_allocator = _SharedPagesAllocator(
            page_store=self._page_store,
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache.swa_kv_pool,
            need_sort=need_sort,
            group_pages=kvcache.swa_group_pages if self._swa_groups > 1 else None,
        )
        self.clear()

    def _eviction_setup(
        self,
        kvcache: TpuUnifiedSWAKVPool,
        total_pages: int,
        static_full_tokens: Optional[int],
    ) -> Tuple[str, int, int]:
        """Resolve ``SGLANG_TPU_UNIFIED_KV_EVICTION`` and each side's static share
        in physical page IDs."""
        policy = tpu_envs.SGLANG_TPU_UNIFIED_KV_EVICTION.get()
        if policy not in _EVICTION_POLICIES:
            raise ValueError(
                f"SGLANG_TPU_UNIFIED_KV_EVICTION={policy!r}; expected one of "
                f"{', '.join(_EVICTION_POLICIES)}"
            )
        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            policy = "full_first"
        if static_full_tokens is None:
            if policy not in ("full_first", "swa_first"):
                policy = "full_first"
            return policy, total_pages, 0
        full_pages = static_full_tokens // self.page_size
        full_share = full_pages * self._full_groups
        # Because G = gcd(B, S), total_pages * G == full_pages * B + swa_pages * S,
        # so swa_pages * M_swa == total_pages - full_pages * M_full.
        swa_share = max(0, total_pages - full_share)
        logger.info(
            "Unified KV eviction under pressure: %s (static shares: %d full, "
            "%d SWA page ids)",
            policy,
            full_share,
            swa_share,
        )
        return policy, full_share, swa_share

    def clear(self) -> None:
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.free_group = None
        self.swa_free_group = []
        self.swa_page_ids_group = []

    def _total_pages(self) -> int:
        return self.size_full // self.page_size

    def _free_page_count(self) -> int:
        return self._page_store.num_free_pages()

    def _page_ids_needed(self, num_full_pages: int, num_swa_pages: int) -> int:
        return self._full_groups * num_full_pages + self._swa_groups * num_swa_pages

    def max_extend_tokens(self) -> int:
        """Maximum joint Full + SWA tokens allocatable in one extend without eviction."""
        return (
            self._free_page_count() // (self._full_groups + self._swa_groups)
        ) * self.page_size

    def available_size(self) -> int:
        return self.max_extend_tokens()

    def full_available_size(self) -> int:
        return (self._free_page_count() // self._full_groups) * self.page_size

    def swa_available_size(self) -> int:
        return (self._free_page_count() // self._swa_groups) * self.page_size

    def _conserve_full_available_size(self) -> int:
        return self._size_full - self.full_attn_allocator.allocated_count()

    def _conserve_swa_available_size(self) -> int:
        return self._size_swa - self.swa_attn_allocator.allocated_count()

    def conserve_full_available_size(self) -> int:
        return self._conserve_full_available_size()

    def conserve_swa_available_size(self) -> int:
        return self._conserve_swa_available_size()

    @property
    def current_full_capacity(self) -> int:
        return self.full_available_size() + self.full_attn_allocator.allocated_count()

    @property
    def current_swa_capacity(self) -> int:
        return self.swa_available_size() + self.swa_attn_allocator.allocated_count()

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        del full_capacity, swa_capacity
        return (
            (self.current_full_capacity, self.full_available_size()),
            (self.current_swa_capacity, self.swa_available_size()),
        )

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        from sglang.srt.mem_cache.prefill_budget import SharedSWAPrefillBudget

        return SharedSWAPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        if min(num_full_pages, num_swa_pages) < 0:
            return False
        return (
            self._page_ids_needed(num_full_pages, num_swa_pages)
            <= self._free_page_count()
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        out_indices_cpu: Optional[torch.Tensor] = None,
    ):
        num_new_pages = int((seq_lens_cpu % self.page_size == 1).sum())
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None
        return super().alloc_decode(
            seq_lens,
            seq_lens_cpu,
            last_loc,
            out_indices_cpu=out_indices_cpu,
        )

    def reclaim_plan(
        self,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        empty_pool: bool = False,
    ) -> Optional[Tuple[int, int]]:
        if full_tokens < 0 or swa_tokens < 0:
            return None
        page_size = self.page_size
        full_pages = (math.ceil(full_tokens) + page_size - 1) // page_size
        swa_pages = (math.ceil(swa_tokens) + page_size - 1) // page_size
        need_pages = self._page_ids_needed(full_pages, swa_pages)

        if empty_pool:
            return (0, 0) if need_pages <= self._total_pages() else None

        free_pages = self._free_page_count()
        if need_pages <= free_pages:
            return (0, 0)

        short_pages = need_pages - free_pages
        m_full = self._full_groups
        m_swa = self._swa_groups
        max_full_pages = min(
            self.full_attn_allocator.allocated_pages // m_full,
            max(0, int(full_evictable_tokens)) // page_size,
        )
        max_swa_pages = min(
            self.swa_attn_allocator.allocated_pages // m_swa,
            max(0, int(swa_evictable_tokens)) // page_size,
        )
        if self._page_ids_needed(max_full_pages, max_swa_pages) < short_pages:
            return None

        # Prefer evicting Full prefix pages first, then SWA pages if needed.
        full_reclaim_pages = min(max_full_pages, _cdiv(short_pages, m_full))
        rem_short = short_pages - full_reclaim_pages * m_full
        swa_reclaim_pages = _cdiv(rem_short, m_swa) if rem_short > 0 else 0
        return (
            full_reclaim_pages * page_size,
            swa_reclaim_pages * page_size,
        )

    def can_reserve(
        self,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        empty_pool: bool = False,
        require_token_slack: bool = False,
    ) -> bool:
        if full_tokens < 0 or swa_tokens < 0:
            return False
        if require_token_slack and (
            full_tokens >= self.size_full or swa_tokens >= self.size_swa
        ):
            return False
        return (
            self.reclaim_plan(
                full_tokens,
                swa_tokens,
                full_evictable_tokens=full_evictable_tokens,
                swa_evictable_tokens=swa_evictable_tokens,
                empty_pool=empty_pool,
            )
            is not None
        )

    def max_new_tokens_for_memory(
        self,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
    ) -> int | None:
        del token_capacity
        from sglang.srt.mem_cache.prefill_budget import estimate_swa_kv_tokens

        def fits(candidate: int) -> bool:
            return self.can_reserve(
                input_tokens + candidate + self.page_size,
                estimate_swa_kv_tokens(
                    input_tokens,
                    candidate,
                    sliding_window_size=sliding_window_size,
                    page_size=self.page_size,
                    allocation_limit=chunk_size,
                ),
                empty_pool=True,
                require_token_slack=True,
            )

        if not fits(0):
            return None
        if fits(max_new_tokens):
            return max_new_tokens
        lo, hi = 0, max_new_tokens
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _evict_shortfall(self, tree_cache, full_pages: int, swa_pages: int) -> None:
        need_pages = self._page_ids_needed(full_pages, swa_pages)
        if need_pages <= self._free_page_count():
            return
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        page_size = self.page_size
        m_full = self._full_groups
        m_swa = self._swa_groups
        if self._eviction_policy == "full_first":
            short = need_pages - self._free_page_count()
            tree_cache.evict_for_alloc(
                EvictParams(
                    num_tokens=_cdiv(short, m_full) * page_size,
                    swa_num_tokens=_cdiv(short, m_swa) * page_size,
                )
            )
            return
        if self._eviction_policy == "swa_first":
            if (
                isinstance(tree_cache, UnifiedRadixCache)
                and get_disagg().disaggregation_mode != "decode"
            ):
                tree_cache.evict_unmatchable_device_leaves(
                    lambda: self._free_page_count() >= need_pages
                )
            for with_full in (False, True):
                short = need_pages - self._free_page_count()
                if short <= 0:
                    return
                tree_cache.evict_for_alloc(
                    EvictParams(
                        num_tokens=_cdiv(short, m_full) * page_size if with_full else 0,
                        swa_num_tokens=_cdiv(short, m_swa) * page_size,
                    )
                )
            return

        def shortfall() -> int:
            return need_pages - self._free_page_count()

        def done() -> bool:
            return shortfall() <= 0

        def overshoots() -> Tuple[int, int]:
            return (
                self.full_attn_allocator.allocated_pages
                + m_full * full_pages
                - self._full_share,
                self.swa_attn_allocator.allocated_pages
                + m_swa * swa_pages
                - self._swa_share,
            )

        def evict(full_ids: int, swa_ids: int) -> None:
            full_tokens = _cdiv(max(full_ids, 0), m_full) * page_size
            swa_tokens = _cdiv(max(swa_ids, 0), m_swa) * page_size
            if full_tokens and swa_tokens:
                tree_cache.evict(
                    EvictParams(num_tokens=full_tokens, swa_num_tokens=swa_tokens)
                )
            elif full_tokens or swa_tokens:
                tree_cache.evict_for_alloc(
                    EvictParams(num_tokens=full_tokens, swa_num_tokens=swa_tokens)
                )

        compact = (
            self._eviction_policy == "compact"
            and isinstance(tree_cache, UnifiedRadixCache)
            and get_disagg().disaggregation_mode != "decode"
        )
        if compact:
            tree_cache.evict_unmatchable_device_leaves(done)
        over_full, over_swa = overshoots()
        if compact and over_swa > 0 and not done():
            evict(min(over_full, shortfall()), 0)
            tree_cache.compact_swa(self._compact_retain_tokens, done)
            over_full, over_swa = 0, overshoots()[1]
        if not done():
            evict(min(over_full, shortfall()), min(over_swa, shortfall()))
        full_held, swa_held = (
            self.full_attn_allocator.allocated_pages + m_full * full_pages,
            self.swa_attn_allocator.allocated_pages + m_swa * swa_pages,
        )
        swa_nearer = swa_held * self._full_share > full_held * self._swa_share
        for kind_is_swa in (swa_nearer, not swa_nearer):
            if done():
                return
            if kind_is_swa and compact:
                tree_cache.compact_swa(self._compact_retain_tokens, done)
                if done():
                    return
            if kind_is_swa:
                evict(0, shortfall())
            else:
                evict(shortfall(), 0)

    def evict_to_free_tokens(
        self, tree_cache, num_tokens: int, *, swa_num_tokens: Optional[int] = None
    ) -> bool | None:
        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return None
        required_swa = num_tokens if swa_num_tokens is None else swa_num_tokens
        need_full_pages = _cdiv(max(0, int(num_tokens)), self.page_size)
        need_swa_pages = _cdiv(max(0, int(required_swa)), self.page_size)
        if hasattr(tree_cache, "evict_for_alloc"):
            self._evict_shortfall(tree_cache, need_full_pages, need_swa_pages)
        else:
            plan = self.reclaim_plan(
                num_tokens,
                required_swa,
                full_evictable_tokens=tree_cache.full_evictable_size(),
                swa_evictable_tokens=tree_cache.swa_evictable_size(),
            )
            if plan is None:
                return False
            full_reclaim, swa_reclaim = plan
            if full_reclaim or swa_reclaim:
                tree_cache.evict(
                    EvictParams(num_tokens=full_reclaim, swa_num_tokens=swa_reclaim)
                )
        return self.new_pages_available(need_full_pages, need_swa_pages)

    def reclaim_for_prealloc(
        self, tree_cache, full_tokens: int, swa_tokens: int
    ) -> str | None:
        ready = self.evict_to_free_tokens(
            tree_cache, full_tokens, swa_num_tokens=swa_tokens
        )
        if ready is None:
            need_full_pages = _cdiv(max(0, int(full_tokens)), self.page_size)
            need_swa_pages = _cdiv(max(0, int(swa_tokens)), self.page_size)
            ready = self.new_pages_available(need_full_pages, need_swa_pages)
        if ready:
            return None
        return (
            "Unified FULL/SWA reclamation insufficient: "
            f"needed=({full_tokens}, {swa_tokens})"
        )

    def debug_print(self) -> str:
        return (
            f"#free-pages: {self._free_page_count()}, "
            f"#full-pages: {self.full_attn_allocator.allocated_pages}, "
            f"#swa-pages: {self.swa_attn_allocator.allocated_pages}, "
        )

