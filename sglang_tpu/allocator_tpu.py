# SPDX-License-Identifier: Apache-2.0
"""TPU paged memory allocator.

CPU-resident free-page bookkeeping.

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

This subclass keeps the free/release pools as host-resident **CPU torch
tensors** (adapted from the numpy-based CPU-resident allocator in
``sglang-jax``; ported to torch tensors for the torch_tpu backend).
``alloc_decode`` precomputes the
assigned page-IDs on CPU and ships a fixed-shape ``(bs,)`` tensor to the
device-side formula with a direct ``cpu_tensor.to(device)`` — no
``from_numpy`` round-trip. The compiled HLO graph has constant input
shapes regardless of how many pages have been consumed, so PJRT reuses
its cache.

The fix only affects the TPU subclass; the parent
:class:`PagedTokenToKVPoolAllocator` is unchanged for other backends.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.utils import get_num_new_pages
from sglang_tpu.kv_free_fence import HeldPages


def alloc_extend_vectorized_cpu(
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fill ``out_indices`` like ``alloc_extend_naive`` without a per-request loop.

    Each request contributes three contiguous runs of token slots: the rest of
    its current partial page (continuing from ``last_loc``), whole new pages,
    and a trailing partial page. All inputs are CPU tensors.

    Example with page_size=4, prefix_lens=[3, 0, 5], seq_lens=[9, 2, 8],
    last_loc=[22, -1, 48], and free_pages=[10, 27, 35, 42]:
      prefix_page_end = [4, 0, 8]
      num1 = [1, 0, 3]  # Continue the current page.
      num2 = [4, 0, 0]  # Fill whole new pages.
      num3 = [1, 2, 0]  # Fill a final partial new page.
    Request 0 gets [23, 40, 41, 42, 43, 108], request 1 gets [140, 141],
    and request 2 gets [49, 50, 51]. Free page 42 is unused.
    """
    prefix_lens = prefix_lens.to(torch.int64)
    seq_lens = seq_lens.to(torch.int64)
    last_loc = last_loc.to(torch.int64)
    free_pages = free_pages.to(torch.int64)

    extend_lens = seq_lens - prefix_lens
    total = int(extend_lens.sum())
    if total == 0:
        return

    prefix_page_end = (prefix_lens + page_size - 1) // page_size * page_size
    num1 = torch.minimum(seq_lens, prefix_page_end) - prefix_lens
    remaining = extend_lens - num1
    num2 = remaining // page_size * page_size
    num3 = remaining - num2
    num_new_pages = num2 // page_size + (num3 > 0).to(torch.int64)
    page_start = torch.cumsum(num_new_pages, 0) - num_new_pages

    # Segment k of request i (k = 0: partial current page, 1: full new pages,
    # 2: partial last page) covers ``seg_len[i, k]`` slots. Segments 0 and 2
    # each continue one page from a base slot; segment 1 spans several pages
    # that need not be contiguous, so its slots look their page up directly.
    seg_len = torch.stack([num1, num2, num3], dim=1)
    zeros = torch.zeros_like(num3)
    if free_pages.numel() > 0:
        last_page_idx = (page_start + num_new_pages - 1).clamp(
            min=0, max=free_pages.numel() - 1
        )
        last_page_base = torch.where(
            num3 > 0, free_pages[last_page_idx] * page_size, zeros
        )
    else:
        last_page_base = zeros
    seg_base = torch.stack([last_loc + 1, zeros, last_page_base], dim=1)

    seg_len_flat = seg_len.reshape(-1)
    seg_base_flat = seg_base.reshape(-1)
    seg_start = torch.cumsum(seg_len_flat, 0) - seg_len_flat
    within = torch.arange(total, dtype=torch.int64) - torch.repeat_interleave(
        seg_start, seg_len_flat, output_size=total
    )
    values = (
        torch.repeat_interleave(seg_base_flat, seg_len_flat, output_size=total) + within
    )
    seg_id = torch.repeat_interleave(
        torch.arange(seg_len_flat.numel(), dtype=torch.int64),
        seg_len_flat,
        output_size=total,
    )
    is_full_seg = seg_id % 3 == 1
    if bool(is_full_seg.any()):
        req_of_slot = seg_id // 3
        page_idx = (page_start[req_of_slot] + within // page_size).clamp(
            max=free_pages.numel() - 1
        )
        full_value = free_pages[page_idx] * page_size + within % page_size
        values = torch.where(is_full_seg, full_value, values)
    out_indices[:total] = values.to(out_indices.dtype)


class TpuPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    # Set by the non-spec host-padding policy; wrapped allocators retain
    # their existing device-return contract.
    defer_decode_upload: bool = False

    def clear(self) -> None:
        # Slot 0 is reserved as the dummy-padding page, so page IDs start at 1.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int32, device="cpu"
        )
        # None outside a free_group_begin()/free_group_end() pair, as in the parent.
        self.free_group = None
        self.free_page_ids_group = []
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0
        # Pages freed while a forward was in flight; see kv_free_fence.py.
        self.held_pages = HeldPages()
        # True while a page sits in free_pages, staged_pages or held_pages; a
        # second release would let a later allocation hand it to two owners.
        self.page_is_free = torch.ones(self.num_pages + 1, dtype=torch.bool)
        self.page_is_free[0] = False

    def available_size(self):
        return super().available_size() + self.held_pages.num_pages * self.page_size

    def _ensure_free_pages(self, num_pages: int) -> bool:
        """Make ``free_pages`` hold at least ``num_pages``, taking held pages
        when short."""
        if (
            self.kv_free_fence is not None
            and not self.kv_free_fence.in_forward_write_target
        ):
            self.kv_free_fence.stats.non_forward_allocs += 1
        if num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        short = num_pages - len(self.free_pages)
        if short <= 0:
            return True
        if short > self.held_pages.num_pages:
            return False
        if self.kv_free_fence.in_forward_write_target:
            # Device compute runs in launch order, so this forward writes last.
            self.kv_free_fence.stats.pages_borrowed += short
            self.free_pages = torch.cat(
                (self.held_pages.borrow(short), self.free_pages)
            )
        else:
            # Other allocations may be DMA targets, which are not ordered after
            # the forward that may still write these.
            self.kv_free_fence.wait_for_in_flight()
            self.merge_and_sort_free()
        return num_pages <= len(self.free_pages)

    def take_free_pages(self, num_pages: int) -> torch.Tensor | None:
        """Pop ``num_pages`` free page ids (host int32) in free-list order, or
        None when the pool cannot supply them."""
        if not self._ensure_free_pages(num_pages):
            return None
        pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self._mark_handed_out(pages)
        return pages

    def alloc(self, need_size: int):
        # Page-aligned bulk allocation, returns contiguous device indices.
        assert need_size % self.page_size == 0, (
            "The allocation size should be page-aligned"
        )
        out_pages = self.take_free_pages(need_size // self.page_size)
        if out_pages is None:
            return None

        # Expand to per-token indices on CPU, then one async H2D.
        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, dtype=torch.int32)
        ).reshape(-1)
        return out_indices.to(self.device, non_blocking=True)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
        out_indices_cpu: torch.Tensor = None,
        return_host: bool = False,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if num_new_pages is None:
            num_new_pages_item = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        else:
            num_new_pages_item = num_new_pages

        free_pages_cpu = self.take_free_pages(num_new_pages_item)
        if free_pages_cpu is None:
            return None

        # Compute the per-token page indices entirely on host, then do a
        # single H2D copy. Batch-wide tensor operations avoid the per-request
        # Python loop in ``alloc_extend_naive``. ``prefix_lens_cpu`` and
        # ``seq_lens_cpu`` are already mirrors of the device tensors, so only
        # ``last_loc`` needs a D2H copy when it is not already on the host.

        if out_indices_cpu is None:
            out_indices_cpu = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int32,
                device="cpu",
            )
        alloc_extend_vectorized_cpu(
            prefix_lens_cpu,
            seq_lens_cpu,
            last_loc.cpu(),
            free_pages_cpu,
            out_indices_cpu,
            self.page_size,
        )
        out_indices = (
            out_indices_cpu
            if return_host
            else out_indices_cpu.to(self.device, non_blocking=True)
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        out_indices_cpu: torch.Tensor | None = None,
    ):
        """``out_indices_cpu``: optional caller-owned host (bs,) buffer filled
        with the returned locations, so host consumers skip a D2H readback.
        Requires a host-resident ``last_loc``."""
        if out_indices_cpu is not None and last_loc.device.type != "cpu":
            raise ValueError("out_indices_cpu requires a host-resident last_loc")
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        # Everything below is computed on CPU torch tensors; we never
        # index into ``self.free_pages`` from a device kernel, so its
        # size doesn't affect any HLO.
        need_new_pages = (seq_lens_cpu % self.page_size == 1).to(torch.int32)
        num_new_pages = int(need_new_pages.sum())

        new_page_ids = self.take_free_pages(num_new_pages)
        if new_page_ids is None:
            return None

        bs = int(seq_lens.shape[0])
        # ``per_req_new_page[i]`` = page-ID assigned to req i if it
        # needs a fresh page, else 0 (multiplied by 0 in the formula
        # below — the sentinel is harmless).
        per_req_new_page = torch.zeros(bs, dtype=torch.int32, device="cpu")
        if num_new_pages > 0:
            start_new_pages = torch.cumsum(need_new_pages, 0) - need_new_pages
            mask = need_new_pages.to(torch.bool)
            per_req_new_page[mask] = new_page_ids[start_new_pages[mask].long()]

        #   need_new_pages[i] == 1  →  out_indices[i] = new_page * page_size
        #   need_new_pages[i] == 0  →  out_indices[i] = last_loc + 1
        if last_loc.device.type == "cpu":
            # Host-resident: last_loc is host, so run the formula on CPU and
            # H2D once (no per-bucket device compile of the allocator formula).
            host_indices = (last_loc + 1) * (1 - need_new_pages) + (
                per_req_new_page * self.page_size * need_new_pages
            )
            if out_indices_cpu is not None:
                out_indices_cpu.copy_(host_indices)
            out_indices = (
                host_indices
                if self.defer_decode_upload
                else host_indices.to(self.device, non_blocking=True)
            )
        else:
            # Device formula on fixed-shape (bs,) inputs (bs is bucketed).
            per_req_new_page_dev = per_req_new_page.to(self.device, non_blocking=True)
            need_new_pages_dev = need_new_pages.to(self.device, non_blocking=True)
            out_indices = (last_loc + 1) * (1 - need_new_pages_dev) + (
                per_req_new_page_dev * self.page_size * need_new_pages_dev
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        return out_indices.int()

    def _mark_handed_out(self, page_ids: torch.Tensor) -> None:
        if page_ids.numel() == 0:
            return
        ids = page_ids.long()
        if not bool(self.page_is_free[ids].all()):
            raise RuntimeError(
                "KV page allocator handed out pages that are not free: "
                f"{ids[~self.page_is_free[ids]].tolist()}"
            )
        self.page_is_free[ids] = False

    def _mark_released(self, page_ids: torch.Tensor) -> None:
        if page_ids.numel() == 0:
            return
        ids = page_ids.long()
        already_free = self.page_is_free[ids]
        if bool(already_free.any()):
            raise RuntimeError(
                "KV page allocator released pages that are already free: "
                f"{ids[already_free].tolist()}"
            )
        unique_ids, counts = torch.unique(ids, return_counts=True)
        if unique_ids.numel() != ids.numel():
            raise RuntimeError(
                "KV page allocator released pages twice in one call: "
                f"{unique_ids[counts > 1].tolist()}"
            )
        self.page_is_free[ids] = True

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        """Keep every page-id release path on the host pool's int32 dtype."""
        normalized = tuple(
            page_id.to(device="cpu", dtype=torch.int32) for page_id in page_ids
        )
        self._mark_released(torch.cat(normalized))
        tag = (
            None
            if self.kv_free_fence is None
            else self.kv_free_fence.hold_tag(
                num_pages=sum(page_id.numel() for page_id in normalized)
            )
        )
        if tag is None:
            super()._release_page_ids(*normalized)
        else:
            self.held_pages.push(tag, torch.cat(normalized))

    def release_held_pages(self, completed: int) -> None:
        released = self.held_pages.pop_released(completed)
        if released:
            super()._release_page_ids(*released)

    def free(self, free_index: torch.Tensor) -> None:
        # Page pools are host-resident int32 tensors. Scheduler locations can
        # be int64, so normalize before the parent derives and reinserts page
        # IDs; otherwise torch.cat promotes the pool and the next decode-page
        # assignment fails against its int32 destination.
        if free_index.device.type != "cpu":
            free_index = free_index.cpu()
        free_index = free_index.to(dtype=torch.int32)
        super().free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        # Radix-cache node values are int64. Normalize to host-resident int32 so
        # torch.cat does not promote the release pool, keeping the free pages
        # int32 for subsequent decode assignments.
        if free_index.device.type != "cpu":
            free_index = free_index.cpu()
        free_index = free_index.to(dtype=torch.int32)
        super().free_segment(free_index, start_pos=start_pos)


def host_return_allocators(allocator) -> list[TpuPagedTokenToKVPoolAllocator]:
    """TPU allocators behind ``allocator`` that can return host-resident indices.

    Empty when any inner allocator would keep its device-return contract. An SWA
    allocator qualifies when both inner allocators are plain TPU allocators; its
    full-to-SWA mapping is host-resident on TPU.
    """
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    if type(allocator) is TpuPagedTokenToKVPoolAllocator:
        return [allocator]
    if type(allocator) is SWATokenToKVPoolAllocator:
        inner = [allocator.full_attn_allocator, allocator.swa_attn_allocator]
        if all(type(a) is TpuPagedTokenToKVPoolAllocator for a in inner) and (
            allocator.full_to_swa_index_mapping.device.type == "cpu"
        ):
            return inner
    return []


class TpuPageInterleavePoolAllocator(PageInterleavePoolAllocator):
    """Page-interleaved full-attention allocator of a TPU decode-context-parallel
    group: logical page ``l`` is rank ``l % N``'s local physical page ``l // N``
    (``PageInterleavePlacement`` at the physical page), each class being a stock
    TPU paged allocator with its own host-resident free list, KV free fence
    hold and double-free guard.

    Position page ``j`` of a request takes class ``j % N`` (rotation base 0):
    a page's parity is its position parity on every rank, which the per-rank
    attention metadata and the P/D page transfer rely on. Decode and
    page-unaligned extends continue a request inside its current page and
    open new pages by position, so no per-chain rotation base is kept.
    ``available_size`` is the min-class capacity floor of the generic class.
    """

    def __init__(
        self,
        size: int,
        physical_page_size: int,
        shard_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache,
        need_sort: bool,
    ):
        assert size % physical_page_size == 0, (
            f"per-rank size {size} is not a multiple of page_size {physical_page_size}"
        )
        # One stock pool per class, each over this rank's physical slots;
        # built before super().__init__, whose clear() resets them.
        self.class_pools = [
            TpuPagedTokenToKVPoolAllocator(
                size, physical_page_size, dtype, device, kvcache, need_sort
            )
            for _ in range(shard_size)
        ]
        super().__init__(
            size,
            physical_page_size,
            shard_size,
            dtype,
            device,
            kvcache,
            need_sort,
        )

    def clear(self):
        for pool in self.class_pools:
            pool.clear()
        self.free_group = None
        self.free_page_ids_group = []
        self.free_pages = None
        self.release_pages = None

    # ---- classed accounting over the class pools ------------------------------

    def class_free_page_counts(self) -> list[int]:
        return [pool.available_size() // self.page_size for pool in self.class_pools]

    def merge_and_sort_free(self):
        for pool in self.class_pools:
            pool.merge_and_sort_free()

    def get_all_free_pages(self) -> torch.Tensor:
        return torch.cat(
            [
                pool.get_all_free_pages().to(torch.int64) * self.shard_size + r
                for r, pool in enumerate(self.class_pools)
            ]
        )

    def debug_check_classes(self):
        for pool in self.class_pools:
            pool._debug_check_no_duplicate_pages()

    def _take_class_pages(self, classes: torch.Tensor) -> torch.Tensor | None:
        """Logical page ids for new pages of the given classes (host int64),
        or None when some class cannot supply its share. Every class is checked
        before any page is taken, so a refusal leaves the pools untouched."""
        n = self.shard_size
        needs = [int((classes == c).sum()) for c in range(n)]
        if any(
            needs[c] * self.page_size > self.class_pools[c].available_size()
            for c in range(n)
        ):
            return None
        logical = torch.empty(classes.numel(), dtype=torch.int64)
        for c in range(n):
            if needs[c] == 0:
                continue
            pages = self.class_pools[c].take_free_pages(needs[c])
            assert pages is not None, "a class pool refused pages it reported free"
            logical[classes == c] = pages.to(torch.int64) * n + c
        return logical

    # ---- alloc / free -----------------------------------------------------------

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
        out_indices_cpu: torch.Tensor = None,
        return_host: bool = False,
        rotation_bases=None,
    ):
        """Continue each request inside its current page from ``last_loc`` and
        open its new position pages ``ceil(prefix / ps) .. ceil(seq / ps) - 1``
        from class ``j % N``; prefixes need not be page-aligned."""
        if rotation_bases is not None and any(
            b not in (None, 0) for b in rotation_bases
        ):
            raise ValueError(
                "TpuPageInterleavePoolAllocator keeps rotation base 0 (page "
                f"parity = position parity), got rotation_bases={rotation_bases}"
            )
        ps, n = self.page_size, self.shard_size
        prefix = prefix_lens_cpu.to(torch.int64)
        seq = seq_lens_cpu.to(torch.int64)
        if self.debug_mode:
            assert torch.all((last_loc.cpu() + 1) % ps == prefix % ps)
            assert extend_num_tokens == int((seq - prefix).sum())
        first_new = (prefix + ps - 1) // ps
        counts = ((seq + ps - 1) // ps - first_new).clamp(min=0)
        total = int(counts.sum())
        if num_new_pages is not None:
            assert num_new_pages == total, (num_new_pages, total)
        starts = torch.cumsum(counts, 0) - counts
        req_of_page = torch.repeat_interleave(torch.arange(len(counts)), counts)
        position_pages = (
            first_new[req_of_page]
            + torch.arange(total, dtype=torch.int64)
            - starts[req_of_page]
        )
        assigned = self._take_class_pages(position_pages % n)
        if assigned is None:
            return None

        if out_indices_cpu is None:
            out_indices_cpu = torch.empty((extend_num_tokens,), dtype=torch.int32)
        # The stock fill: request i's k-th new page is assigned[starts[i] + k].
        alloc_extend_vectorized_cpu(
            prefix_lens_cpu, seq_lens_cpu, last_loc.cpu(), assigned, out_indices_cpu, ps
        )
        if self.debug_mode:
            assert len(torch.unique(out_indices_cpu)) == len(out_indices_cpu)
        if return_host:
            return out_indices_cpu
        return out_indices_cpu.to(self.device, non_blocking=True).int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        out_indices_cpu: torch.Tensor | None = None,
    ):
        """One new page for each request whose new token opens position page
        ``(seq_len - 1) // ps``, from class ``page % N``; the others continue
        at ``last_loc + 1``. ``out_indices_cpu`` (optional host buffer) receives
        the locations too and needs a host-resident ``last_loc``."""
        ps, n = self.page_size, self.shard_size
        seq = seq_lens_cpu.to(torch.int64)
        if out_indices_cpu is not None and last_loc.device.type != "cpu":
            raise ValueError("out_indices_cpu requires a host-resident last_loc")
        if self.debug_mode:
            assert torch.all((last_loc.cpu() + 2) % ps == seq % ps)
        need_new = seq % ps == 1
        opened = ((seq - 1) // ps)[need_new]
        assigned = self._take_class_pages(opened % n)
        if assigned is None:
            return None
        per_req_new_page = torch.zeros(seq.numel(), dtype=torch.int64)
        per_req_new_page[need_new] = assigned
        need_new_pages = need_new.to(torch.int64)
        if last_loc.device.type != "cpu":
            # Device formula on fixed-shape (bs,) inputs, as the stock pool does.
            per_req_new_page = per_req_new_page.to(self.device, non_blocking=True)
            need_new_pages = need_new_pages.to(self.device, non_blocking=True)
        out_indices = (last_loc + 1) * (1 - need_new_pages) + (
            per_req_new_page * ps * need_new_pages
        )
        if last_loc.device.type == "cpu":
            if out_indices_cpu is not None:
                out_indices_cpu.copy_(out_indices)
            out_indices = out_indices.to(self.device, non_blocking=True)
        return out_indices.int()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        # The tree quantum is the physical page, so every free covers whole
        # pages; split them by owner class, each pool defers inside its own
        # free group and holds them behind the KV free fence.
        self.free_page_ids(
            torch.unique(free_index.to("cpu", torch.int64) // self.page_size)
        )

    def free_page_ids(self, page_ids: torch.Tensor):
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to("cpu", torch.int64)
        owners = page_ids % self.shard_size
        for c, pool in enumerate(self.class_pools):
            mine = page_ids[owners == c]
            if mine.numel():
                pool.free_page_ids((mine // self.shard_size).to(torch.int32))

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        self.free(free_index)

    def free_group_begin(self):
        for pool in self.class_pools:
            pool.free_group_begin()

    def free_group_end(self):
        for pool in self.class_pools:
            pool.free_group_end()
