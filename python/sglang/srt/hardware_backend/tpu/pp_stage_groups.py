# SPDX-License-Identifier: Apache-2.0
"""Process groups of a pipeline stage that spans several devices.

By default torch_tpu compiles the collectives of a process subgroup into
programs for every device of the slice, which all subgroups launch together.
Pipeline stages run their forwards at different times, so the collectives
inside one stage (the context-parallel K/V gather, the last-token and MoE
data-parallel gathers) would wait for the devices of the other stages. A
standalone subgroup compiles them for its own devices, which launch them
alone. The stage hand-off stays one collective over the whole slice.
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


def _stage_groups(parallel: Any) -> list[Any]:
    """The distinct model-parallel groups of more than one device inside this
    rank's pipeline stage, in the same order on every rank."""
    groups: list[Any] = []
    for group in (
        parallel.tp_group,
        parallel.attn_tp_group,
        parallel.attn_cp_group,
        parallel.moe_dp_group,
        parallel.moe_ep_group,
        parallel.moe_tp_group,
    ):
        if group.world_size > 1 and all(group is not seen for seen in groups):
            groups.append(group)
    return groups


def _load_make_standalone() -> Callable[[Any], None]:
    try:
        from torch_tpu._internal.distributed.spmd_util import make_standalone
    except ImportError as e:
        raise RuntimeError(
            "Pipeline stages of more than one device (--pp-size > 1 with "
            "--tp-size > 1) need a torch_tpu with standalone process subgroups "
            "(google-pytorch/torch_tpu#4387); the installed torch_tpu has no "
            "torch_tpu._internal.distributed.spmd_util.make_standalone."
        ) from e
    return make_standalone


def make_stage_groups_standalone(parallel: Any) -> None:
    """Let every pipeline stage of several devices launch its collectives
    without the other stages; must run before the first collective of the
    groups. Nothing to do for one-device stages or a single stage."""
    if parallel.pp_size == 1 or parallel.tp_size == 1:
        return
    make_standalone = _load_make_standalone()
    groups = _stage_groups(parallel)
    for group in groups:
        # Returns once every member of the group has called it.
        make_standalone(group.device_group)
    logger.info(
        "pipeline stage %d: standalone process groups %s",
        parallel.pp_rank,
        [(group.unique_name, group.ranks) for group in groups],
    )
