# SPDX-License-Identifier: Apache-2.0
"""Select checkpoint tensor names before RunAI submits payload reads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

_EXPERT = re.compile(
    r"^layers\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w[123]\.(weight|scale)$"
)


@dataclass(frozen=True)
class ExpertReadPlan:
    tensor_names: frozenset[str]
    shard_names: frozenset[str]
    total_tensors: int
    local_expert_tensors: int


def plan_expert_reads(
    weight_map: Mapping[str, str],
    *,
    num_layers: int,
    num_experts: int,
    ep_size: int,
    ep_rank: int,
) -> ExpertReadPlan:
    """Keep local routed experts and all replicated/host-table parameters.

    This is a payload filter, not a filter on the already-materialized iterator.
    Host Engram tensors must stay: the current shared table loader has every TP
    rank fill its own row range of each checkpoint tensor. MTP is not loaded by
    the target model. Unknown names are retained rather than silently dropped.
    """
    if num_layers <= 0 or num_experts <= 0:
        raise ValueError("Layer and expert counts must be positive")
    if ep_size <= 0 or num_experts % ep_size or not 0 <= ep_rank < ep_size:
        raise ValueError("Expert count must divide EP size and rank must be in range")
    local_count = num_experts // ep_size
    lo, hi = ep_rank * local_count, (ep_rank + 1) * local_count
    selected = set()
    expert_count = local_expert_count = 0
    for name in weight_map:
        if name.startswith("mtp."):
            continue
        match = _EXPERT.fullmatch(name)
        if match is not None:
            layer, expert = int(match[1]), int(match[2])
            if layer >= num_layers or expert >= num_experts:
                raise ValueError(f"Checkpoint expert outside configured model: {name}")
            expert_count += 1
            if not lo <= expert < hi:
                continue
            local_expert_count += 1
        selected.add(name)
    # Fail closed on an incompatible naming scheme, partial index or wrong
    # config instead of reporting a speedup caused by silently missing weights.
    expected = num_layers * num_experts * 6
    if expert_count != expected:
        raise ValueError(
            f"Expected {expected} routed expert tensors, found {expert_count}"
        )
    return ExpertReadPlan(
        frozenset(selected),
        frozenset(weight_map[name] for name in selected),
        len(weight_map),
        local_expert_count,
    )
