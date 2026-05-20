"""TPU paged memory allocator.

CPU-resident free-page bookkeeping. Mirrors what sglang-jax (numpy
free_pages) and torchtpu-vllm (CPU block_table) both do for the same
underlying reason.

Why CPU bookkeeping on TPU
--------------------------
PJRT compiles a fresh HLO graph for every unique tensor shape it sees.
The upstream allocator keeps ``self.free_pages`` as a device tensor
that shrinks by one entry every page-boundary allocation
(``self.free_pages = self.free_pages[num_new_pages:]``). The gather
``self.free_pages[start_new_pages]`` inside ``alloc_decode`` therefore
sees a different input shape every ``page_size`` decode tokens, which
triggers a ~25 ms TPU recompile per page boundary (~25% of long-decode
wall time at page_size=32).

This subclass keeps the free/release pools as numpy arrays on the
host. ``alloc_decode`` precomputes the assigned page-IDs on CPU and
H2D-copies a fixed-shape ``(bs,)`` tensor into the device-side
formula. The compiled HLO graph has constant input shapes regardless
of how many pages have been consumed, so PJRT reuses its cache.

The fix only affects the TPU subclass; the parent
:class:`PagedTokenToKVPoolAllocator` is unchanged for other backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def _np_to_device(arr: np.ndarray, device) -> torch.Tensor:
    """Synchronous H2D copy. ``arr.copy()`` so the source numpy buffer
    is owned by the resulting torch CPU tensor's storage (and stays
    alive across the H2D), and ``.to(device)`` without
    ``non_blocking=True`` because torch_tpu's PJRT path mishandles
    fire-and-forget transfers from temporary CPU memory.
    """
    return torch.from_numpy(arr.copy()).to(device)


# Alias so the rare prefill-time call site reads clearly.
_np_to_device_oneshot = _np_to_device


class TpuPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.roundup = page_size - 1
        # Parent's __init__ called clear(); replace its device-tensor
        # pools with numpy versions.
        self.clear()

    # ------------------------------------------------------------------
    # Pool lifecycle — numpy arrays on host
    # ------------------------------------------------------------------

    def clear(self) -> None:
        # Slot 0 is reserved as the dummy-padding page, so page IDs start at 1.
        self.free_pages = np.arange(1, self.num_pages + 1, dtype=np.int64)
        self.release_pages = np.empty((0,), dtype=np.int64)
        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self) -> int:
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def merge_and_sort_free(self) -> None:
        if len(self.release_pages) > 0:
            self.free_pages = np.sort(
                np.concatenate([self.free_pages, self.release_pages])
            )
            self.release_pages = np.empty((0,), dtype=self.release_pages.dtype)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def alloc(self, need_size: int):
        # Page-aligned bulk allocation, returns contiguous device indices.
        assert (
            need_size % self.page_size == 0
        ), "The allocation size should be page-aligned"
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages_np = self.free_pages[:num_pages].copy()
        self.free_pages = self.free_pages[num_pages:]

        # Expand to per-token indices on CPU, then one synchronous H2D.
        out_indices_np = (
            out_pages_np[:, None] * self.page_size
            + np.arange(self.page_size, dtype=np.int64)
        ).reshape(-1)
        return _np_to_device_oneshot(out_indices_np, self.device)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if num_new_pages is None:
            seq_lens_np = seq_lens_cpu.numpy()
            prefix_lens_np = prefix_lens_cpu.numpy()
            num_new_pages_item = int(
                (
                    (seq_lens_np + self.roundup) // self.page_size
                    - (prefix_lens_np + self.roundup) // self.page_size
                ).sum()
            )
        else:
            num_new_pages_item = int(num_new_pages)

        if self.need_sort and num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return None

        # Compute the per-token page indices entirely on host, then do a
        # single H2D copy. The shared ``alloc_extend_naive`` iterates over
        # ``range(len(prefix_lens))`` and reads scalar elements off its
        # tensor inputs each iteration — if those tensors are on TPU, the
        # XLA backend lowers each ``//`` to a one-element
        # ``tt_jit_floor_divide_div`` HLO module per call, which is both
        # slow (~64 tiny compiles per extend) and has been observed to
        # halt the SparseCore with a UserFatal on Qwen3-32B at TP=8,
        # concurrency=64. ``prefix_lens_cpu`` / ``seq_lens_cpu`` are
        # already mirrors of the device tensors so the only extra D2H is
        # the tiny ``last_loc`` vector (one int32 per sequence).
        if num_new_pages_item > 0:
            free_pages_cpu = torch.from_numpy(
                self.free_pages[:num_new_pages_item].copy()
            )
        else:
            free_pages_cpu = torch.empty((0,), dtype=torch.int64)

        out_indices_cpu = torch.empty(
            (extend_num_tokens,),
            dtype=torch.int32,
        )
        alloc_extend_naive(
            prefix_lens_cpu,
            seq_lens_cpu,
            last_loc.cpu(),
            free_pages_cpu,
            out_indices_cpu,
            self.page_size,
            "cpu",
        )
        out_indices = out_indices_cpu.to(self.device)

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        # Everything below is computed on CPU; we never index into
        # ``self.free_pages`` from a device kernel, so its size doesn't
        # affect any HLO.
        seq_lens_np = seq_lens_cpu.numpy()
        need_new_pages_np = (seq_lens_np % self.page_size == 1).astype(np.int32)
        num_new_pages = int(need_new_pages_np.sum())

        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        bs = int(seq_lens.shape[0])
        # ``per_req_new_page[i]`` = page-ID assigned to req i if it
        # needs a fresh page, else 0 (multiplied by 0 in the formula
        # below — the sentinel is harmless).
        per_req_new_page_np = np.zeros(bs, dtype=np.int64)
        if num_new_pages > 0:
            new_page_ids_np = self.free_pages[:num_new_pages]
            start_new_pages_np = np.cumsum(need_new_pages_np) - need_new_pages_np
            mask = need_new_pages_np.astype(bool)
            per_req_new_page_np[mask] = new_page_ids_np[start_new_pages_np[mask]]
            self.free_pages = self.free_pages[num_new_pages:]

        # Two fixed-shape ``(bs,)`` synchronous H2D copies. bs is
        # bucketed by the attention backend, so PJRT sees a stable input
        # shape and reuses its compiled graph for the device formula
        # below regardless of context length / number of pages
        # consumed so far.
        per_req_new_page = _np_to_device(per_req_new_page_np, self.device)
        need_new_pages = _np_to_device(need_new_pages_np, self.device)

        # Stable-shape device formula:
        #   need_new_pages[i] == 1  →  out_indices[i] = new_page * page_size
        #   need_new_pages[i] == 0  →  out_indices[i] = last_loc + 1
        out_indices = (last_loc + 1) * (1 - need_new_pages) + (
            per_req_new_page * self.page_size * need_new_pages
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        return out_indices.int()

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            # D2H copy of the freed token indices, then everything
            # stays on host until merge_and_sort_free flushes them
            # back into self.free_pages.
            free_page_indices_np = np.unique(
                free_index.cpu().numpy() // self.page_size
            )
            if self.need_sort:
                self.release_pages = np.concatenate(
                    [free_page_indices_np, self.release_pages]
                )
            else:
                self.free_pages = np.concatenate(
                    [free_page_indices_np, self.free_pages]
                )
        else:
            # Aggregated frees still come in as device tensors; flush
            # at free_group_end -> free.
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(np.unique(self.free_pages)) == len(self.free_pages)
