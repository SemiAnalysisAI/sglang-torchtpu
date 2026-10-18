# SPDX-License-Identifier: Apache-2.0
"""Per-step driver of the Kimi-K3 decode megakernel.

The runner owns the megakernel's weights, its four per-row state arrays and
one compiled op per row tile. Before every decode step (eagerly, after the
compile policy pads the batch) it releases the rows of requests that left,
hands off requests that just entered decode, and stages the step's row
permutation; the compiled forward then runs one megakernel call.

A request is keyed by ``(req_pool_idx, req_generation)``. The generation is
bumped every time a request-pool slot is allocated, so a finished, aborted or
retracted request never matches a later occupant of its slot, and is released
at the next decode step it is absent from. A retracted request re-prefills
through SGLang and is handed off again under its new generation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool

from .handoff import Handoff, HandoffSources
from .program import layer_counts
from .slots import RequestKey, RowMove, SlotTable

# Batch rows in the request pool's slot 0 are padding: real requests are never
# allocated it, and the compile policy pads with it.
_PADDING_SLOT = 0


class StepInputs(NamedTuple):
    """One decode step's row mapping, as fresh device tensors.

    Built per step rather than written into persistent buffers, so the next
    step's staging can never overwrite tensors an in-flight step still reads.
    """

    input_to_row: torch.Tensor  # [B] long: the batch index feeding each row
    positions: torch.Tensor  # [B] int32: cached tokens per row, -1 for padding
    row_to_output: torch.Tensor  # [B] long: the row each batch index reads


class KimiK3MegakernelRunner:
    """Weights, row state, row bookkeeping and the compiled decode ops."""

    def __init__(
        self,
        *,
        weights: Sequence[torch.Tensor],
        states: tuple[torch.Tensor, ...],
        m_kb: torch.Tensor,
        decode_ops: dict[int, Callable[..., torch.Tensor]],
        rank: torch.Tensor,
        layers: int,
    ):
        self.weights = tuple(weights)
        self.states = states
        self.m_kb = m_kb
        self.decode_ops = decode_ops
        # int32 [1] holding this device's TP rank (program.py).
        self.rank = rank
        self.layers = layers
        self.max_rows = states[0].shape[0]
        self.context = states[2].shape[3]
        self.device = states[0].device
        self.slot_table = SlotTable(self.max_rows)
        self._handoff: Handoff | None = None
        self._req_generation: torch.Tensor | None = None

    def attach_pools(
        self,
        *,
        req_to_token_pool: HybridReqToTokenPool,
        token_to_kv_pool: HybridLinearKVPool,
    ) -> None:
        """Bind SGLang's pools, check their layout, and warm the eager copies."""
        if not isinstance(req_to_token_pool, HybridReqToTokenPool) or not isinstance(
            token_to_kv_pool, HybridLinearKVPool
        ):
            raise TypeError(
                "the Kimi-K3 megakernel reads hybrid KDA + MLA pools, got "
                f"{type(req_to_token_pool).__name__} / "
                f"{type(token_to_kv_pool).__name__}"
            )
        kda_layers, mla_layers = layer_counts(self.layers)
        sources = HandoffSources.from_pools(
            req_to_token=req_to_token_pool.req_to_token,
            mamba_slot_of_req=req_to_token_pool.req_index_to_mamba_index_mapping,
            mamba_pool=req_to_token_pool.mamba_pool,
            kv_pool=token_to_kv_pool.full_kv_pool,
            kda_layers=kda_layers,
            mla_layers=mla_layers,
        )
        # Read on every step; a device tensor would sync each time.
        if req_to_token_pool.req_generation.device.type != "cpu":
            raise RuntimeError("req_generation must be host-resident")
        self._req_generation = req_to_token_pool.req_generation
        self._handoff = Handoff(states=self.states, sources=sources, m_kb=self.m_kb)
        # Compile the handoff and row-move programs now, not on the first
        # request; row 0 holds no request yet.
        self._handoff.copy_request(row=0, req_pool_idx=_PADDING_SLOT, cached_len=0)
        self._handoff.move_row(RowMove(source=0, destination=0))

    def prepare_decode(
        self,
        *,
        req_pool_indices: Sequence[int],
        seq_lens: Sequence[int],
        padded_batch_size: int,
    ) -> StepInputs:
        """Sync rows with this decode batch and stage its row mapping.

        ``req_pool_indices`` covers the padded batch (padding rows hold the
        padding slot); ``seq_lens`` covers at least the live rows.
        """
        live = [
            (batch_index, req_pool_idx, seq_len)
            for batch_index, (req_pool_idx, seq_len) in enumerate(
                zip(req_pool_indices, seq_lens)
            )
            if req_pool_idx != _PADDING_SLOT
        ]
        self._check_capacity(live)
        if padded_batch_size not in self.decode_ops:
            raise RuntimeError(
                f"decode batch padded to {padded_batch_size} rows, but the "
                f"megakernel compiled tiles {sorted(self.decode_ops)}; "
                "--tpu-compile-bs must list exactly those tiles"
            )
        keys = self._request_keys([req_pool_idx for _, req_pool_idx, _ in live])
        self._release_absent(set(keys))
        for key, (_, req_pool_idx, seq_len) in zip(keys, live):
            if key not in self.slot_table:
                row = self.slot_table.assign(key)
                self._handoff.copy_request(
                    row=row, req_pool_idx=req_pool_idx, cached_len=seq_len - 1
                )
        rows = self.slot_table.rows_for(keys)
        return self._stage(live, rows, padded_batch_size)

    def check_not_decoding(self, *, req_pool_indices: Sequence[int]) -> None:
        """Raise if a non-decode batch carries a request the megakernel decodes.

        SGLang would run it over pools frozen at that request's handoff.
        """
        indices = [idx for idx in req_pool_indices if idx != _PADDING_SLOT]
        owned = [key for key in self._request_keys(indices) if key in self.slot_table]
        if owned:
            raise RuntimeError(
                f"requests {owned} decode in the Kimi-K3 megakernel but reached a "
                "non-decode batch, whose SGLang caches are stale for them"
            )

    def forward_decode(self, input_ids: torch.Tensor, step: StepInputs) -> torch.Tensor:
        """One megakernel step (traced into the compiled forward); [B, vocab] logits."""
        tokens = input_ids.to(torch.int32).index_select(0, step.input_to_row)
        logits = self.decode_ops[input_ids.shape[0]](
            *self.weights, *self.states, self.rank, tokens, step.positions
        )
        return logits.index_select(0, step.row_to_output)

    def _check_capacity(self, live: list[tuple[int, int, int]]) -> None:
        if len(live) > self.max_rows:
            raise RuntimeError(
                f"decode batch of {len(live)} requests, but the megakernel holds "
                f"{self.max_rows} rows (--max-running-requests)"
            )
        longest = max((seq_len for _, _, seq_len in live), default=0)
        if longest > self.context:
            raise RuntimeError(
                f"decode position {longest - 1} is past the megakernel's "
                f"{self.context}-token rows (--context-length)"
            )

    def _request_keys(self, req_pool_indices: list[int]) -> list[RequestKey]:
        generations = self._req_generation[req_pool_indices].tolist()
        return list(zip(req_pool_indices, generations))

    def _release_absent(self, live_keys: set[RequestKey]) -> None:
        for key in self.slot_table.owners():
            if key not in live_keys:
                move = self.slot_table.release(key)
                if move is not None:
                    self._handoff.move_row(move)

    def _stage(
        self, live: list[tuple[int, int, int]], rows: list[int], padded: int
    ) -> StepInputs:
        input_to_row = torch.arange(padded, dtype=torch.long)
        positions = torch.full((padded,), -1, dtype=torch.int32)
        row_to_output = torch.arange(padded, dtype=torch.long)
        for (batch_index, _, seq_len), row in zip(live, rows):
            input_to_row[row] = batch_index
            positions[row] = seq_len - 1
            row_to_output[batch_index] = row
        return StepInputs(
            input_to_row=input_to_row.to(self.device),
            positions=positions.to(self.device),
            row_to_output=row_to_output.to(self.device),
        )
