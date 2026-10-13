"""Exercise expert movement with a caller-owned CPU process group."""

import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.eplb import expert_location_updater
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _move_experts(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        cpu_group = dist.new_group(backend="gloo")
        old_placement = [0, 1, 2, 3]
        new_placement = [2, 0, 3, 1]
        for group in (None, cpu_group):
            tensors = [
                torch.tensor(old_placement[rank * 2 : rank * 2 + 2], dtype=dtype)
                .unsqueeze(1)
                .repeat(1, 3)
                for dtype in (torch.float32, torch.uint8)
            ]
            with patch.object(
                expert_location_updater, "P2POp", wraps=dist.P2POp
            ) as make_op:
                expert_location_updater.update_expert_weights_single_layer(
                    routed_experts_weights=tensors,
                    temp_buffers=[torch.empty_like(tensor) for tensor in tensors],
                    old_physical_to_logical_map=old_placement,
                    new_physical_to_logical_map=new_placement,
                    num_local_physical_experts=2,
                    num_gpu_per_node=1,
                    rank=rank,
                    world_size=2,
                    group=group,
                )
            assert {call.kwargs["op"] for call in make_op.call_args_list} == {
                dist.isend,
                dist.irecv,
            }
            assert all(call.kwargs["group"] is group for call in make_op.call_args_list)
            for tensor in tensors:
                expected = (
                    torch.tensor(
                        new_placement[rank * 2 : rank * 2 + 2], dtype=tensor.dtype
                    )
                    .unsqueeze(1)
                    .expand_as(tensor)
                )
                torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TestExpertLocationUpdaterGroup(unittest.TestCase):
    def test_default_and_explicit_group(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_move_experts, args=(f"{directory}/rendezvous",), nprocs=2)


if __name__ == "__main__":
    unittest.main()
