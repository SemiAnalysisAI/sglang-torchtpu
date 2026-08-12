# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Canonical rank initialization shared by scheduler process launchers."""

from dataclasses import dataclass, replace
from typing import List, Optional

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.patch_torch import initialize_torch_tpu_worker


@dataclass(frozen=True)
class SchedulerProcessConfig:
    port_args: PortArgs
    tp_rank: int
    pp_rank: int
    attn_cp_rank: int
    moe_dp_rank: int
    moe_ep_rank: int
    dp_rank: Optional[int]
    display_tp_rank: int
    display_dp_rank: Optional[int]
    display_moe_ep_rank: int


def compute_scheduler_parallel_ranks(
    server_args: ServerArgs, tp_rank: int, dp_rank: Optional[int]
) -> tuple[int, int, int, Optional[int]]:
    """Compute scheduler-local ranks from one canonical TP coordinate."""
    attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
    attn_tp_size = server_args.tp_size // attn_dp_size // server_args.attn_cp_size
    attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
    moe_dp_rank = tp_rank // (server_args.tp_size // server_args.moe_dp_size)
    moe_ep_rank = (
        tp_rank
        % (server_args.tp_size // server_args.moe_dp_size)
        // (server_args.tp_size // server_args.moe_dp_size // server_args.ep_size)
    )
    if server_args.enable_dp_attention:
        dp_rank = tp_rank // (attn_tp_size * server_args.attn_cp_size)
    return attn_cp_rank, moe_dp_rank, moe_ep_rank, dp_rank


def initialize_scheduler_process_config(
    server_args: ServerArgs,
    port_args: PortArgs,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    worker_ports: Optional[List[int]] = None,
) -> SchedulerProcessConfig:
    """Resolve physical rank identity before deriving any scheduler state."""
    tp_rank, pp_rank = initialize_torch_tpu_worker(server_args, tp_rank, pp_rank)
    provisional_dp_rank = dp_rank
    attn_cp_rank, moe_dp_rank, moe_ep_rank, dp_rank = (
        compute_scheduler_parallel_ranks(server_args, tp_rank, dp_rank)
    )

    if server_args.enable_dp_attention and dp_rank != provisional_dp_rank:
        current_endpoint = NetworkAddress.parse(
            port_args.scheduler_input_ipc_name.removeprefix("tcp://")
        )
        if worker_ports is not None:
            scheduler_input_port = worker_ports[dp_rank]
        else:
            # The Rust-server path derives one consecutive scheduler port per
            # DP rank instead of broadcasting pre-bound worker ports.
            if provisional_dp_rank is None:
                raise RuntimeError(
                    "DP-attention rank alignment requires either the provisional "
                    "DP rank or the worker-port table."
                )
            scheduler_input_port = (
                current_endpoint.port + dp_rank - provisional_dp_rank
            )
        port_args = replace(
            port_args,
            scheduler_input_ipc_name=NetworkAddress(
                current_endpoint.host, scheduler_input_port
            ).to_tcp(),
        )

    offset = server_args.ep_join_rank_offset
    return SchedulerProcessConfig(
        port_args=port_args,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        moe_ep_rank=moe_ep_rank,
        dp_rank=dp_rank,
        display_tp_rank=tp_rank + offset,
        display_dp_rank=dp_rank + offset if dp_rank is not None else None,
        display_moe_ep_rank=moe_ep_rank + offset,
    )
