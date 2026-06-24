"""TPU paged memory allocator using pure PyTorch ops (no triton)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


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

    # ------------------------------------------------------------------ #
    # Fixed-capacity free-list (TPU shape stability)
    #
    # The base allocator stores ``free_pages`` as a *dynamic-size* tensor:
    # allocation slices it (``free_pages[n:]``) and free concatenates
    # (``torch.cat``), so the tensor is a different size every decode step. On
    # TPU torch_tpu compiles a new XLA program per distinct tensor size, so this
    # free-list churn re-materialized a few executables every step (pinned in
    # ``tt_jit_cat``/``tt_jit_mul`` on an s64[~free_count] buffer).
    #
    # When ``need_sort`` is False (the non-disaggregation path, i.e. our scope)
    # we keep ``free_pages`` at a CONSTANT shape ``[num_pages]`` with the valid
    # free pages compacted at the front ``[0:_num_free]``. Front-remove (alloc)
    # and front-add (free) are done with ``torch.roll`` + a fixed-size front
    # write, whose XLA programs are keyed only on the small, bounded per-step
    # page counts (0..~num_draft_tokens), not on the free-list size -> a tiny
    # fixed set of shapes that warmup covers, so steady-state decode compiles 0
    # new programs. ``_num_free`` is a host int (never baked into a device op as
    # an offset). When ``need_sort`` is True we defer to the base dynamic
    # behaviour (disaggregation only; out of scope; not the recompile path).
    # ------------------------------------------------------------------ #

    def clear(self):
        self._fixed = not self.need_sort
        if not self._fixed:
            super().clear()
            return
        # The padded slot 0 is reserved (dummy outputs); free pages are 1..num_pages.
        self._cap = self.num_pages
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self._num_free = self.num_pages
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.is_not_in_free_group = True
        self.free_group = []

    def _navail(self) -> int:
        """Logical free-page count (constant-shape path uses the counter)."""
        return self._num_free if self._fixed else len(self.free_pages)

    def _consume_front(self, n: int) -> None:
        """Remove the first ``n`` (already-read) free pages, fixed-shape.

        ``roll(-n)`` moves the kept tail ``[n:_num_free]`` to the front; the
        consumed ``n`` rotate to the buffer tail (beyond ``_num_free`` -> never
        re-allocated). Shape is constant ``[_cap]``; the program is keyed only on
        ``n`` (a small bounded count)."""
        if n:
            self.free_pages = torch.roll(self.free_pages, -n, 0)
        self._num_free -= n

    def _add_front(self, pages: torch.Tensor) -> None:
        """Prepend ``pages`` to the free list, fixed-shape (mirror of base
        ``free_pages = cat((pages, free_pages))`` for need_sort=False)."""
        m = pages.numel()
        if m == 0:
            return
        self.free_pages = torch.roll(self.free_pages, m, 0)
        self.free_pages[:m] = pages
        self._num_free += m

    def available_size(self):
        if not self._fixed:
            return super().available_size()
        return self._num_free * self.page_size

    def backup_state(self):
        if not self._fixed:
            return super().backup_state()
        return (self.free_pages.clone(), self._num_free)

    def restore_state(self, state):
        if not self._fixed:
            super().restore_state(state)
            return
        self.free_pages, self._num_free = state

    def merge_and_sort_free(self):
        # need_sort is False on the fixed path, so this is never needed; keep a
        # safe no-op so any guarded caller does not corrupt the buffer.
        if not self._fixed:
            super().merge_and_sort_free()

    def alloc(self, need_size: int):
        if not self._fixed:
            return super().alloc(need_size)
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"
        num_pages = need_size // self.page_size
        if num_pages > self._num_free:
            return None
        out_pages = self.free_pages[:num_pages]
        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)
        self._consume_front(num_pages)
        return out_indices

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
            num_new_pages_tensor = (
                (seq_lens + self.roundup) // self.page_size
                - (prefix_lens + self.roundup) // self.page_size
            ).sum()
            num_new_pages_item = num_new_pages_tensor.item()
        else:
            num_new_pages_item = num_new_pages

        if self.need_sort and num_new_pages_item > self._navail():
            self.merge_and_sort_free()

        if num_new_pages_item > self._navail():
            return None

        out_indices = torch.empty(
            (extend_num_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # The kernel reads the valid free pages from the front of free_pages
        # (indices 0..num_new_pages-1), which holds the valid prefix in both the
        # fixed and dynamic representations.
        alloc_extend_naive(
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.page_size,
            self.device,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        if self._fixed:
            self._consume_front(num_new_pages_item)
        else:
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

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )

        if num_new_pages > self._navail():
            self.merge_and_sort_free()

        if num_new_pages > self._navail():
            return None

        need_new_pages = (seq_lens % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = end_new_pages - need_new_pages
        if num_new_pages == 0:
            out_indices = last_loc + 1
        else:
            out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[
                start_new_pages
            ] * self.page_size * need_new_pages

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        if self._fixed:
            self._consume_front(int(num_new_pages))
        else:
            self.free_pages = self.free_pages[num_new_pages:]
        return out_indices.int()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            device = free_index.device
            free_page_indices = torch.unique(free_index.cpu() // self.page_size)
            free_page_indices = free_page_indices.to(device)
            if self.need_sort:
                # need_sort=True (disaggregation): base dynamic release path.
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                # need_sort=False (our scope): fixed-shape front-add.
                self._add_front(free_page_indices)
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            valid = (
                self.free_pages[: self._num_free] if self._fixed else self.free_pages
            )
            assert len(torch.unique(valid)) == len(valid)
