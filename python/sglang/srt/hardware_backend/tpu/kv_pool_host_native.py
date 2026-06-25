"""Native (raiden-free) host KV pool for TPU HiCache.

Counterpart to ``TpuRaidenHostKVCache`` that uses no raiden: per-layer pinned
torch host buffers mirroring the fused device KV layout, moved with plain torch
ops (D2H ``.to("cpu")``, H2D ``index_put_`` into the live fused buffer). The H2D
mutates the buffer's shared storage in place and is PJRT dependency-tracked, so
the consuming forward auto-waits on it -- no completion gate is needed; the
returned event is already complete. Allocates/frees whole pages, matching
``TpuRaidenHostKVCache``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

logger = logging.getLogger(__name__)

NATIVE_LAYOUT = "fused_native"


class _CompletedEvent:
    """Stand-in for the CUDA finish-event the ack queues expect.

    Native transfers are dependency-tracked, so completion is implicit: the
    consuming forward orders after the pending write via PJRT. ``query`` is
    always ``True`` and ``synchronize`` is a no-op.
    """

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def record(self, *args, **kwargs) -> None:
        return None

    def wait(self, *args, **kwargs) -> None:
        return None


_DONE = _CompletedEvent()


class TpuNativeHostKVCache(HostKVCache):
    """Page-granular host KV pool using pinned torch buffers + native copies."""

    device_pool: "TpuFusedMHATokenToKVPool"

    def __init__(
        self,
        device_pool: "TpuFusedMHATokenToKVPool",
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        # Skip HostKVCache.__init__ (psutil RAM check + its own buffer alloc);
        # replicate just the sizing, then build pinned per-layer host buffers.
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = NATIVE_LAYOUT
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num

        fused0 = device_pool.get_fused_kv_buffer(self.start_layer)
        self._page_shape = tuple(fused0.shape[1:])  # (page_size, kv_heads, kvp, hd)

        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        assert self.size > device_pool.size, (
            "Host KV pool must be larger than the device pool "
            f"(host={self.size} <= device={device_pool.size})."
        )

        self.host_kv = self.init_kv_buffer()

        self.lock = threading.RLock()
        self.clear()

    # ------------------------------------------------------------------ #
    # Sizing / buffers
    # ------------------------------------------------------------------ #

    def get_size_per_token(self) -> int:
        fused = self.device_pool.get_fused_kv_buffer(self.start_layer)
        page_bytes = int(np.prod(fused.shape[1:])) * fused.element_size()
        per_token = page_bytes // self.page_size
        return per_token * self.layer_num

    def init_kv_buffer(self) -> List[torch.Tensor]:
        shape = (self.page_num,) + self._page_shape
        buffers = []
        pinned = self.pin_memory
        for _ in range(self.layer_num):
            try:
                t = torch.zeros(shape, dtype=self.dtype, pin_memory=pinned)
            except Exception as e:  # pinning needs an initialized PJRT client
                logger.warning(
                    "Native host KV pin_memory=%s failed (%s); falling back to "
                    "unpinned host buffer.",
                    pinned,
                    repr(e)[:80],
                )
                pinned = False
                t = torch.zeros(shape, dtype=self.dtype)
            buffers.append(t)
        self.pin_memory = pinned
        logger.info(
            "Native host KV pool ready: %d layers x %s %s, %d pages (pinned=%s)",
            self.layer_num,
            shape,
            self.dtype,
            self.page_num,
            self.pin_memory,
        )
        return buffers

    # ------------------------------------------------------------------ #
    # Page-granular allocation (identical to the raiden pool)
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        with self.lock:
            self.free_pages = deque(range(self.page_num))

    def available_size(self) -> int:
        return len(self.free_pages) * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "host alloc size must be a multiple of page_size"
        num_pages = need_size // self.page_size
        with self.lock:
            if num_pages > len(self.free_pages):
                return None
            pages = [self.free_pages.popleft() for _ in range(num_pages)]
        base = np.repeat(
            np.array(pages, dtype=np.int64) * self.page_size, self.page_size
        )
        offsets = np.tile(np.arange(self.page_size, dtype=np.int64), num_pages)
        return torch.from_numpy(base + offsets)

    def free(self, indices: torch.Tensor) -> int:
        page_ids = torch.unique(indices.cpu() // self.page_size).tolist()
        with self.lock:
            self.free_pages.extend(page_ids)
        return len(indices)

    # ------------------------------------------------------------------ #
    # Transfer (token indices -> page-id index tensors -> native copies)
    # ------------------------------------------------------------------ #

    def _page_ids(self, token_indices: torch.Tensor) -> np.ndarray:
        ids = token_indices.cpu().numpy()
        return (ids[:: self.page_size] // self.page_size).astype(np.int64)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> "_CompletedEvent":
        dev_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        dev_idx = torch.as_tensor(dev_pages, dtype=torch.long, device="tpu")
        host_idx = torch.as_tensor(host_pages, dtype=torch.long)
        for i in range(self.layer_num):
            dev_buf = device_pool.get_fused_kv_buffer(self.start_layer + i)
            self.host_kv[i][host_idx] = dev_buf[dev_idx].to("cpu")
        return _DONE

    def load_all_layer(
        self, device_pool, host_indices, device_indices
    ) -> "_CompletedEvent":
        host_pages = self._page_ids(host_indices)
        dev_pages = self._page_ids(device_indices)
        dev_idx = torch.as_tensor(dev_pages, dtype=torch.long, device="tpu")
        host_idx = torch.as_tensor(host_pages, dtype=torch.long)
        for i in range(self.layer_num):
            dev_buf = device_pool.get_fused_kv_buffer(self.start_layer + i)
            # Eager in-place scatter into the live fused buffer. Mutates the
            # shared storage; PJRT orders the consuming forward after it.
            dev_buf[dev_idx] = self.host_kv[i][host_idx].to("tpu")
        return _DONE

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        raise NotImplementedError(
            "TpuNativeHostKVCache uses all-layer loads via load_all_layer()."
        )

    # ------------------------------------------------------------------ #
    # L3 storage seams (not yet supported on TPU)
    # ------------------------------------------------------------------ #

    def get_data_page(self, index, flat: bool = True):
        raise NotImplementedError("L3 storage not yet supported on TPU native path")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("L3 storage not yet supported on TPU native path")

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError("L3 storage not yet supported on TPU native path")
