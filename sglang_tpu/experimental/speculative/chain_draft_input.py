"""Draft forward input of the chain worker: verify-shaped rows plus the hidden the draft reads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jax import tree_util

from sglang.srt.speculative.dflash_info import DFlashVerifyInput


# A dataclass because it extends core's DFlashVerifyInput and registers as a jax pytree.
@dataclass
class ChainDraftForwardInput(DFlashVerifyInput):
    """Per-row draft conditioning, plus the KV length RPA reads during the fused draft.

    Attributes:
      hidden_states: [rows, D] target aux features or the draft's own hidden.
      rpa_draft_kv_lens: [pbs] int32 absolute KV length of the current fused draft
        step; None outside the fused draft, where the attention metadata holds it.
    """

    hidden_states: torch.Tensor | None = None
    rpa_draft_kv_lens: torch.Tensor | None = None


# A subclass is its own pytree type; register it like DFlashVerifyInput so
# direct_compile traces hidden_states as a graph input.
tree_util.register_dataclass(ChainDraftForwardInput)


def chain_draft_forward_input(
    *, hidden_states: torch.Tensor, width: int
) -> ChainDraftForwardInput:
    device = hidden_states.device
    return ChainDraftForwardInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=width,
        custom_mask=None,
        hidden_states=hidden_states,
    )
