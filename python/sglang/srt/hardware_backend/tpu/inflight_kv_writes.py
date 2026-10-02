# SPDX-License-Identifier: Apache-2.0
"""KV pages a launched TPU forward may still write.

Under overlap scheduling the scheduler frees pages, and hands them to new work,
while the forward it launched last is still running. raiden's host-device DMAs
are not ordered against that forward, so a transfer onto one of its pages races
it. ``collect_forward_kv_writes`` snapshots those pages when the forward
launches, before any of them can be freed, and ``pages_overlap`` checks a
transfer's pages against the snapshot. ``prefill_early_send`` applies the same
idea to the PD prefill early send.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import msgspec
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class KVPages(msgspec.Struct, frozen=True):
    """Page ids, int64 on CPU, in the full KV pool and in the SWA pool."""

    full: torch.Tensor
    # None when the model has no SWA pool.
    swa: Optional[torch.Tensor] = None


def collect_forward_kv_writes(
    *,
    batch: ScheduleBatch,
    req_to_token: torch.Tensor,
    page_size: int,
    translate_full_to_swa: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> Optional[KVPages]:
    """Pages ``batch``'s forward may write, or None for forward modes not modelled
    (verify, split prefill, ...).

    Call right after the forward launches. A request writes token positions
    ``[start, kv_allocated_len)`` of its ``req_to_token`` row. For prefill and
    mixed batches ``start`` is the extend start. A decode has already counted
    its new token as committed. A speculative decode writes from the committed
    length on, and the host's committed length lags the device's.
    """
    mode = batch.forward_mode
    if batch.beam_tail is not None:
        # Beam members' rows follow batch.reqs and are written too.
        return None
    if not batch.reqs:
        starts, ends = [], []
    elif mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
        # Encoder KV is written below the extend start.
        if any(batch.encoder_lens_cpu or ()) or any(
            req.extend_range is None for req in batch.reqs
        ):
            return None
        starts = [req.extend_range.start for req in batch.reqs]
        ends = [
            max(req.extend_range.end, req.kv.kv_allocated_len) for req in batch.reqs
        ]
    elif mode == ForwardMode.DECODE:
        starts = [req.kv.kv_committed_len - 1 for req in batch.reqs]
        ends = [req.kv.kv_allocated_len for req in batch.reqs]
    else:
        return None
    slots = _gather_row_slots(
        req_to_token=req_to_token,
        rows=[req.kv.req_pool_idx for req in batch.reqs],
        starts=starts,
        ends=ends,
    )
    return KVPages(
        full=torch.unique(slots // page_size),
        swa=(
            None
            if translate_full_to_swa is None
            else torch.unique(translate_full_to_swa(slots) // page_size)
        ),
    )


def token_slots_to_pages(slots: torch.Tensor, page_size: int) -> torch.Tensor:
    return torch.unique(slots.cpu().to(torch.int64) // page_size)


def pages_overlap(load: Optional[KVPages], writes: Optional[KVPages]) -> bool:
    """Whether a load onto ``load`` may race a forward writing ``writes``.

    None on either side means the pages are unknown, which counts as overlap.
    """
    if load is None or writes is None:
        return True
    if torch.isin(load.full, writes.full).any():
        return True
    if load.swa is None or load.swa.numel() == 0:
        return False
    return writes.swa is None or bool(torch.isin(load.swa, writes.swa).any())


def _gather_row_slots(
    *,
    req_to_token: torch.Tensor,
    rows: Sequence[int],
    starts: Sequence[int],
    ends: Sequence[int],
) -> torch.Tensor:
    """KV slots ``req_to_token[rows[i], starts[i]:ends[i]]`` for every i, on CPU."""
    starts_t = torch.tensor(starts, dtype=torch.int64).clamp(min=0)
    lens = (torch.tensor(ends, dtype=torch.int64) - starts_t).clamp(min=0)
    total = int(lens.sum())
    if total == 0:
        return torch.empty(0, dtype=torch.int64)
    first = torch.cumsum(lens, dim=0) - lens
    positions = torch.repeat_interleave(starts_t - first, lens) + torch.arange(total)
    row_ids = torch.repeat_interleave(torch.tensor(rows, dtype=torch.int64), lens)
    device = req_to_token.device
    # TPU keeps req_to_token on the host (SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1).
    slots = req_to_token[row_ids.to(device), positions.to(device)]
    return slots.cpu().to(torch.int64)
