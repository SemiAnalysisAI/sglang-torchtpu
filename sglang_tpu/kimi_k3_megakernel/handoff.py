# SPDX-License-Identifier: Apache-2.0
"""Prefill-to-decode state handoff and row moves for the Kimi-K3 megakernel.

When a request enters decode, ``Handoff.copy_request`` fills its megakernel row
in the four state arrays (``program.state_shapes``) from SGLang's pools:

* KDA conv / recurrent state from the request's ``TpuMambaPool`` slot;
* MLA key / value expanded per head from the request's latent rows in the
  ``TpuFusedMLATokenToKVPool``: ``key = [latent @ m_kb]_nope | k_rope | 0``,
  ``value = [latent @ m_kb]_v``, zero past the cached length.

Every device op uses fixed-shape index tensors (``index_select`` /
``index_copy_``, never ``tensor[python_int]`` views or ``[:length]`` slices),
so each compiles once and serving never recompiles for a new prompt length or
row.
"""

from __future__ import annotations

import msgspec
import torch

from sglang_tpu.kv_pool import TpuFusedMLATokenToKVPool
from sglang_tpu.mamba_pool import TpuMambaPool

from .slots import RowMove

# Kimi-K3 per rank at TP32: 3 KDA heads of 128 (q|k|v conv channels 3 * 384),
# MLA latent 512 + rope 64, 3 MLA heads whose m_kb columns are k_nope 128 | v 128.
_KDA_HEADS = 3
_HEAD_DIM = 128
_CONV_TAPS = 3
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_MLA_HEADS = 3


class HandoffSources(msgspec.Struct, frozen=True):
    """The pool buffers a handoff reads, checked once against the megakernel layout."""

    req_to_token: torch.Tensor  # [req_pool_size, max_context] token locations
    mamba_slot_of_req: torch.Tensor  # [req_pool_size] int32
    conv_buffers: tuple[torch.Tensor, ...]  # per KDA layer [slots, 3, 1, 1152] f32
    recurrent_buffers: tuple[torch.Tensor, ...]  # per KDA layer [slots, 3, 128, 128]
    latent_buffers: tuple[torch.Tensor, ...]  # per MLA layer, paged, last dim 640

    @classmethod
    def from_pools(
        cls,
        *,
        req_to_token: torch.Tensor,
        mamba_slot_of_req: torch.Tensor,
        mamba_pool: TpuMambaPool,
        kv_pool: TpuFusedMLATokenToKVPool,
        kda_layers: int,
        mla_layers: int,
    ) -> HandoffSources:
        _check_mamba_pool(mamba_pool, kda_layers)
        _check_mla_pool(kv_pool, mla_layers)
        return cls(
            req_to_token=req_to_token,
            mamba_slot_of_req=mamba_slot_of_req,
            conv_buffers=tuple(mamba_pool.mamba_cache.conv[0]),
            recurrent_buffers=tuple(mamba_pool.mamba_cache.temporal),
            latent_buffers=tuple(kv_pool.kv_buffer),
        )


def _check_mamba_pool(pool: TpuMambaPool, kda_layers: int) -> None:
    if not isinstance(pool, TpuMambaPool):
        raise TypeError(
            f"the megakernel reads a TpuMambaPool, got {type(pool).__name__}"
        )
    # Speculative checkpoints (#1460) give every request several state rows;
    # the handoff reads one row per slot.
    if pool.checkpoint_width != 1:
        raise ValueError(
            f"the megakernel handoff needs one state row per slot, got "
            f"checkpoint_width {pool.checkpoint_width}"
        )
    conv, temporal = pool.mamba_cache.conv, pool.mamba_cache.temporal
    conv_row = (_CONV_TAPS, 1, 3 * _KDA_HEADS * _HEAD_DIM)
    recurrent_row = (_KDA_HEADS, _HEAD_DIM, _HEAD_DIM)
    if (
        len(conv) != 1
        or len(conv[0]) != kda_layers
        or len(temporal) != kda_layers
        or any(tuple(buf.shape[1:]) != conv_row for buf in conv[0])
        or any(tuple(buf.shape[1:]) != recurrent_row for buf in temporal)
    ):
        raise ValueError(
            f"Mamba pool layout does not match the megakernel's {kda_layers} KDA "
            f"layers of conv {conv_row} / recurrent {recurrent_row}"
        )


def _check_mla_pool(pool: TpuFusedMLATokenToKVPool, mla_layers: int) -> None:
    if not isinstance(pool, TpuFusedMLATokenToKVPool):
        raise TypeError(
            "the megakernel reads a TpuFusedMLATokenToKVPool (--attention-backend "
            f"pallas_tpu_mla), got {type(pool).__name__}"
        )
    if (
        pool.store_dtype != torch.bfloat16
        or pool.kv_lora_rank != _KV_LORA_RANK
        or pool.nope_dim != _KV_LORA_RANK
        or pool.qk_rope_head_dim != _ROPE_DIM
        or len(pool.kv_buffer) != mla_layers
    ):
        raise ValueError(
            f"MLA pool layout does not match the megakernel's {mla_layers} bf16 "
            f"layers of latent {_KV_LORA_RANK} + rope {_ROPE_DIM}"
        )


class Handoff:
    """Copies requests' prefill state into megakernel rows and moves rows."""

    def __init__(
        self,
        *,
        states: tuple[torch.Tensor, ...],
        sources: HandoffSources,
        m_kb: torch.Tensor,
    ):
        self.states = states
        self.sources = sources
        self.m_kb = m_kb[0]  # [mla, 512, 768]
        self.context = states[2].shape[3]
        self.device = states[0].device
        rows = states[0].shape[0]
        # One [1] index per row, built once: a fresh tensor per call is an H2D
        # copy, a Python-int view a recompile per row.
        self._row_index = tuple(
            torch.tensor([row], dtype=torch.long, device=self.device)
            for row in range(rows)
        )
        width = sources.req_to_token.shape[1]
        self._req_columns = min(width, self.context)
        self._positions = torch.arange(self.context, device=self.device)

    def copy_request(self, *, row: int, req_pool_idx: int, cached_len: int) -> None:
        """Fill ``row`` from request ``req_pool_idx``'s first ``cached_len`` tokens."""
        if not 0 <= cached_len <= self.context:
            raise ValueError(f"cached_len {cached_len} is outside [0, {self.context}]")
        req_index = torch.tensor(
            [req_pool_idx], dtype=torch.long, device=self.sources.req_to_token.device
        )
        valid = self._positions < torch.tensor(cached_len, device=self.device)
        self._copy_kda(row, req_index)
        self._copy_mla(row, req_index, valid)

    def _copy_kda(self, row: int, req_index: torch.Tensor) -> None:
        conv_state, recurrent_state = self.states[:2]
        slot = self.sources.mamba_slot_of_req.index_select(
            0, req_index.to(self.sources.mamba_slot_of_req.device)
        ).to(device=self.device, dtype=torch.long)
        # [3 taps, 1, (q|k|v) x 3 heads x 128] -> [taps, q/k/v, head, channel].
        conv = torch.stack(
            [
                buf.index_select(0, slot).reshape(_CONV_TAPS, 3, _KDA_HEADS, _HEAD_DIM)
                for buf in self.sources.conv_buffers
            ]
        )
        recurrent = torch.stack(
            [buf.index_select(0, slot)[0] for buf in self.sources.recurrent_buffers]
        )
        conv_state.index_copy_(0, self._row_index[row], conv.to(torch.bfloat16)[None])
        recurrent_state.index_copy_(0, self._row_index[row], recurrent[None])

    def _token_locations(self, req_index: torch.Tensor) -> torch.Tensor:
        """The request's token locations, padded or cut to ``[context]``."""
        locations = self.sources.req_to_token.index_select(0, req_index)[0]
        locations = locations[: self._req_columns].to(
            device=self.device, dtype=torch.long
        )
        if self._req_columns < self.context:
            locations = torch.nn.functional.pad(
                locations, (0, self.context - self._req_columns)
            )
        return locations

    def _copy_mla(self, row: int, req_index: torch.Tensor, valid: torch.Tensor) -> None:
        key_cache, value_cache = self.states[2:]
        # Positions past the cached length read location 0, the pool's padding
        # page, and are zeroed below.
        locations = torch.where(valid, self._token_locations(req_index), 0)
        # Gather whole 640-wide rows from each layer's base buffer, then slice:
        # slicing the pool first would materialize a strided copy of it.
        latents = torch.stack(
            [
                buf.view(-1, buf.shape[-1]).index_select(0, locations)
                for buf in self.sources.latent_buffers
            ]
        )
        latent = latents[:, :, :_KV_LORA_RANK]
        rope = latents[:, :, _KV_LORA_RANK : _KV_LORA_RANK + _ROPE_DIM]
        mla_layers = latents.shape[0]
        expanded = (
            torch.bmm(latent.float(), self.m_kb.float())
            .to(torch.bfloat16)
            .view(mla_layers, self.context, _MLA_HEADS, 2 * _HEAD_DIM)
            .permute(0, 2, 1, 3)
        )
        key = torch.cat(
            (
                expanded[..., :_HEAD_DIM],
                rope[:, None].expand(mla_layers, _MLA_HEADS, self.context, _ROPE_DIM),
                torch.zeros(
                    (mla_layers, _MLA_HEADS, self.context, _HEAD_DIM - _ROPE_DIM),
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
            ),
            dim=-1,
        )
        value = expanded[..., _HEAD_DIM:]
        mask = valid.view(1, 1, self.context, 1)
        key_cache.index_copy_(0, self._row_index[row], torch.where(mask, key, 0)[None])
        value_cache.index_copy_(
            0, self._row_index[row], torch.where(mask, value, 0)[None]
        )

    def move_row(self, move: RowMove) -> None:
        """Copy row ``move.source`` into ``move.destination`` in every state array."""
        source = self._row_index[move.source]
        destination = self._row_index[move.destination]
        for state in self.states:
            state.index_copy_(0, destination, state.index_select(0, source))
