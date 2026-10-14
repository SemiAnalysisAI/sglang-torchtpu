# SPDX-License-Identifier: Apache-2.0
"""Static-shape state handoff and row compaction for the Kimi-K3 decode megakernel.

When a request finishes prefill on SGLang's paged MLA + KDA path and enters
decode, :func:`handoff_request_to_megakernel_row` populates its assigned row
``r`` in the four megakernel-owned decode state tensors:

1. ``conv_state``      ``[max_rows, 69, 3, 3, 3, 128]`` ``bfloat16``
2. ``recurrent_state`` ``[max_rows, 69, 3, 128, 128]``  ``float32``
3. ``key_cache``       ``[max_rows, 24, 3, context, 256]`` ``bfloat16``
4. ``value_cache``     ``[max_rows, 24, 3, context, 128]`` ``bfloat16``

When a request finishes and :meth:`SlotTable.release` compacts the highest live
row into the freed row, :func:`apply_row_move` copies that row across all four
state tensors in place.

Every device operation in both functions uses fixed-shape 1D index tensors and
``index_select`` / ``index_copy_`` (never dynamic slices ``[:seq_len]`` or
Python-int tensor views ``state[row]``), so XLA compiles a single static graph
with zero serving-time recompilations across arbitrary prompt lengths and slots.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .slots import RowMove


def extract_mamba_buffers(
    req_to_token_pool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(conv_buffers, recurrent_buffers)`` across the 69 KDA layers."""
    mamba_cache = req_to_token_pool.mamba_pool.mamba_cache
    return list(mamba_cache.conv[0]), list(mamba_cache.temporal)


def extract_mla_kv_buffers(token_to_kv_pool) -> list[torch.Tensor]:
    """Return the 24 per-layer MLA latent KV base buffers in MLA layer order."""
    full_pool = getattr(token_to_kv_pool, "full_kv_pool", token_to_kv_pool)
    return list(full_pool.kv_buffer)


def build_padded_token_locs_and_mask(
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    cached_len: int,
    context: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build static-shape ``(token_locs_dev [context], valid_mask_dev [1, 1, context, 1])``.

    Positions ``[0, cached_len)`` read the request's committed KV pool slots;
    positions ``[cached_len, context)`` read dummy slot ``0`` and are zeroed by
    ``valid_mask_dev``.
    """
    if cached_len < 0 or cached_len > context:
        raise ValueError(
            f"cached_len ({cached_len}) must be in [0, context ({context})]"
        )
    locs_cpu = torch.zeros((context,), dtype=torch.long)
    if cached_len > 0:
        row_tokens = req_to_token[req_pool_idx, :cached_len]
        if row_tokens.device.type != "cpu":
            row_tokens = row_tokens.cpu()
        locs_cpu[:cached_len] = row_tokens.to(torch.long)
    mask_cpu = (torch.arange(context, dtype=torch.long) < cached_len).view(
        1, 1, context, 1
    )
    return locs_cpu.to(device), mask_cpu.to(device)


def handoff_request_to_megakernel_row(
    states: Sequence[torch.Tensor],
    *,
    row_dev: torch.Tensor,
    mamba_slot_dev: torch.Tensor,
    token_locs_dev: torch.Tensor,
    valid_mask_dev: torch.Tensor,
    conv_buffers: Sequence[torch.Tensor],
    recurrent_buffers: Sequence[torch.Tensor],
    mla_kv_buffers: Sequence[torch.Tensor],
    m_kb: torch.Tensor,
) -> None:
    """Copy one request's prefill KDA + MLA caches into megakernel row ``row_dev``.

    Parameters
    ----------
    states:
        ``(conv_state, recurrent_state, key_cache, value_cache)`` megakernel
        state tensors on device, each with leading dimension ``max_rows``.
    row_dev:
        1D ``torch.long`` tensor of shape ``(1,)`` on device holding target row ``r``.
    mamba_slot_dev:
        1D ``torch.long`` tensor of shape ``(1,)`` on device holding the
        request's ``MambaPool`` slot index.
    token_locs_dev:
        1D ``torch.long`` tensor of shape ``(context,)`` on device holding the
        KV-pool token locations padded with ``0`` past ``cached_len``.
    valid_mask_dev:
        Boolean tensor of shape ``(1, 1, context, 1)`` on device, ``True`` for
        ``t < cached_len``.
    conv_buffers:
        69 per-layer KDA conv state tensors, each ``[pool_size, 3, 1, 1152]`` f32.
    recurrent_buffers:
        69 per-layer KDA recurrent state tensors, each ``[pool_size, 3, 128, 128]`` f32.
    mla_kv_buffers:
        24 per-layer MLA latent KV base buffers, each contiguous with trailing
        dimension ``>= 576`` (``[:512]`` normalized ``c_kv``, ``[512:576]`` ``k_pe``).
    m_kb:
        Megakernel ``m_kb`` weight family for this rank, shape ``[1, 24, 512, 768]``
        ``bfloat16``.
    """
    conv_state, recurrent_state, key_cache, value_cache = states
    kda_layers = len(conv_buffers)
    mla_layers = len(mla_kv_buffers)
    context = token_locs_dev.shape[0]
    device = conv_state.device

    # 1. KDA conv & recurrent state handoff (69 layers)
    conv_all = (
        torch.stack(
            [
                buf.index_select(0, mamba_slot_dev).reshape(3, 3, 3, 128)
                for buf in conv_buffers
            ],
            dim=0,
        )
        .to(torch.bfloat16)
        .unsqueeze(0)
    )
    rec_all = torch.stack(
        [
            buf.index_select(0, mamba_slot_dev).reshape(3, 128, 128)
            for buf in recurrent_buffers
        ],
        dim=0,
    ).unsqueeze(0)
    conv_state.index_copy_(0, row_dev, conv_all)
    recurrent_state.index_copy_(0, row_dev, rec_all)

    # 2. MLA latent -> expanded per-head K/V handoff (24 layers)
    # Index into the contiguous 2D view of each layer's base buffer BEFORE
    # slicing columns [:512] and [512:576], avoiding whole-pool non-contiguous
    # slice views on TpuFusedMLATokenToKVPool (where trailing dim is 640).
    latents = torch.stack(
        [
            buf.view(-1, buf.shape[-1]).index_select(0, token_locs_dev)
            for buf in mla_kv_buffers
        ],
        dim=0,
    )  # [24, context, D >= 576]
    c_kv = latents[:, :, :512]  # [24, context, 512]
    k_pe = latents[:, :, 512:576]  # [24, context, 64]

    w_kb = m_kb[0] if m_kb.ndim == 4 else m_kb  # [24, 512, 768]
    expanded = (
        torch.bmm(c_kv.float(), w_kb.float())
        .to(torch.bfloat16)
        .view(mla_layers, context, 3, 256)
        .permute(0, 2, 1, 3)
    )  # [24, 3, context, 256]

    k_nope = expanded[:, :, :, :128]  # [24, 3, context, 128]
    v_all = expanded[:, :, :, 128:]  # [24, 3, context, 128]
    k_rope = k_pe.unsqueeze(1).expand(mla_layers, 3, context, 64)
    k_zeros = torch.zeros(
        (mla_layers, 3, context, 64), dtype=torch.bfloat16, device=device
    )
    k_all = torch.cat((k_nope, k_rope, k_zeros), dim=-1)  # [24, 3, context, 256]

    k_masked = torch.where(valid_mask_dev, k_all, torch.zeros_like(k_all)).unsqueeze(0)
    v_masked = torch.where(valid_mask_dev, v_all, torch.zeros_like(v_all)).unsqueeze(0)

    key_cache.index_copy_(0, row_dev, k_masked)
    value_cache.index_copy_(0, row_dev, v_masked)


def apply_row_move(
    states: Sequence[torch.Tensor],
    move: RowMove | None,
    *,
    src_dev: torch.Tensor | None = None,
    dst_dev: torch.Tensor | None = None,
) -> None:
    """Copy ``move.source`` into ``move.destination`` across all 4 state tensors."""
    if move is None:
        return
    device = states[0].device
    if src_dev is None:
        src_dev = torch.tensor([move.source], dtype=torch.long, device=device)
    if dst_dev is None:
        dst_dev = torch.tensor([move.destination], dtype=torch.long, device=device)
    for state in states:
        row_data = state.index_select(0, src_dev)
        state.index_copy_(0, dst_dev, row_data)
