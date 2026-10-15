"""Two-level request broadcast with shared-memory and inline tensor inputs.

Four gloo ranks stand in for two nodes of two ranks each. Rank 0 receives one
request whose features the tokenizer already put in /dev/shm and one whose
features arrive inline (the tokenizer's fallback). After the scheduler broadcast
every rank must hold SHM handles for the first request
and inline tensors for the fallback. After ``_finalize_shm_features`` every rank
must see the exact pixels while every segment is unlinked. Also exercise empty
polls, entirely inline inputs, and a remote node that cannot allocate SHM.
"""

import datetime
import os
import pickle
import unittest
from array import array
from multiprocessing import shared_memory
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    MMInputsProcessError,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.managers.mm_utils import (  # noqa: E402
    ShmPointerMMData,
    wrap_shm_features,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler_components.node_fanout import (  # noqa: E402
    node_fanout_enabled,
)
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

WORLD, NNODES, NUM_IMAGES = 4, 2, 7


def _request(rid, features) -> TokenizedEmbeddingReqInput:
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text="",
        input_ids=array("q", [1]),
        mm_inputs=(
            MultimodalProcessorOutput(
                mm_items=[
                    MultimodalDataItem(modality=Modality.IMAGE, feature=f, hash=100 + i)
                    for i, f in enumerate(features)
                ]
            )
            if features
            else None
        ),
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def _receiver() -> SchedulerRequestReceiver:
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(is_multimodal=True),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


def _run_rank(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=WORLD,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        receiver = _receiver()
        parallel = SimpleNamespace(
            attn_dp_enabled=False,
            tp_size=WORLD,
            pp_size=1,
            nnodes=NNODES,
            tp_group=SimpleNamespace(
                rank=rank,
                rank_in_group=rank,
                ranks=list(range(WORLD)),
                cpu_group=torch.distributed.group.WORLD,
            ),
        )
        with (
            patch(
                "sglang.srt.managers.scheduler_components.node_fanout.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.node_fanout.get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.mm_utils.get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
            patch.dict(os.environ, {"SGLANG_NODE_FANOUT_REQ_BROADCAST": "1"}),
        ):
            _exercise_receiver(rank, receiver)
    finally:
        torch.distributed.destroy_process_group()


def _exercise_receiver(rank, receiver):
    assert receiver._broadcast_reqs_across_ranks([] if rank == 0 else None) == []
    for base in (30, 90):
        _check_mixed_requests(rank, receiver, base)

    # No SHM on any node: this must not enter a materialization collective.
    # This tensor is large enough to use broadcast_pyobj's separate buffer path.
    reqs = (
        [_request("inline-only", [torch.full((640, 640, 3), 42, dtype=torch.uint8)])]
        if rank == 0
        else None
    )
    with patch(
        "sglang.srt.managers.scheduler_components.node_fanout._allocate_segment"
    ) as allocate:
        reqs = receiver._broadcast_reqs_across_ranks(reqs)
        receiver._finalize_shm_features(reqs)
        allocate.assert_not_called()
    assert torch.all(reqs[0].mm_inputs.mm_items[0].feature == 42)

    # Remote allocation failure must reject the request on every rank.
    reqs = None
    if rank == 0:
        req = _request("remote-shm-full", [torch.ones(8, dtype=torch.uint8)])
        wrap_shm_features(req, force=True)
        reqs = pickle.loads(pickle.dumps([req]))
    with patch(
        "sglang.srt.managers.scheduler_components.node_fanout._allocate_segment",
        return_value=None,
    ):
        reqs = receiver._broadcast_reqs_across_ranks(reqs)
    names = [item.feature.shm_name for item in reqs[0].mm_inputs.mm_items]
    receiver._finalize_shm_features(reqs)
    assert isinstance(reqs[0].mm_inputs, MMInputsProcessError)
    torch.distributed.barrier()
    _assert_unlinked(names)

    # The opt-out continues to use the flat broadcast.
    with patch.dict(os.environ, {"SGLANG_NODE_FANOUT_REQ_BROADCAST": "0"}):
        reqs = receiver._broadcast_reqs_across_ranks(
            [_request("text-only", None)] if rank == 0 else None
        )
    assert reqs[0].rid == "text-only"


def _check_mixed_requests(rank, receiver, base):
    reqs = None
    if rank == 0:
        pixels = [
            torch.full((321 + i, 323, 3), base + i, dtype=torch.uint8)
            for i in range(NUM_IMAGES)
        ]
        # The tokenizer's /dev/shm handles, as rank 0 unpickles them.
        shm_req = _request(f"vlm-{base}", pixels)
        wrap_shm_features(shm_req, force=True)
        shm_req = pickle.loads(pickle.dumps(shm_req))
        inline = [t.clone() + 100 for t in pixels]
        reqs = [
            shm_req,
            _request(f"inline-{base}", inline),
            _request(f"text-{base}", None),
        ]
    reqs = receiver._broadcast_reqs_across_ranks(reqs)
    assert [r.rid for r in reqs] == [
        f"vlm-{base}",
        f"inline-{base}",
        f"text-{base}",
    ]
    assert reqs[2].mm_inputs is None
    items = reqs[0].mm_inputs.mm_items
    assert all(isinstance(it.feature, ShmPointerMMData) for it in items)
    assert all(torch.is_tensor(it.feature) for it in reqs[1].mm_inputs.mm_items)
    names = [it.feature.shm_name for it in items]
    if rank >= WORLD // NNODES:
        # Forwarded tensors land in one segment on the second node.
        forwarded = reqs[0].mm_inputs.mm_items
        assert len({it.feature.shm_name for it in forwarded}) == 1
        assert all(it.feature.offset % 4096 == 0 for it in forwarded)
    receiver._finalize_shm_features(reqs)
    for shift, req in ((0, reqs[0]), (100, reqs[1])):
        for i, it in enumerate(req.mm_inputs.mm_items):
            assert torch.is_tensor(it.feature)
            assert torch.all(it.feature == base + i + shift)
            assert it.hash == 100 + i
    # Check rank isolation for a tensor at a nonzero packed-segment offset.
    last = reqs[0].mm_inputs.mm_items[-1].feature
    writer_rank = WORLD // NNODES
    if rank == writer_rank:
        last.zero_()
    torch.distributed.barrier()
    assert torch.all(last == (0 if rank == writer_rank else base + NUM_IMAGES - 1))
    _assert_unlinked(names)


def _assert_unlinked(names):
    for name in set(names):
        try:
            shared_memory.SharedMemory(name=name).close()
        except FileNotFoundError:
            continue
        raise AssertionError(f"segment {name} was not unlinked")


class TestNodeFanoutBroadcast(unittest.TestCase):
    def test_transport_eligibility(self):
        cases = [
            ({}, False, True),
            ({"nnodes": 1}, False, False),
            ({"tp_size": 1}, False, False),
            ({"pp_size": 2}, False, False),
            ({"tp_size": 3}, False, False),
            ({"attn_dp_enabled": True}, False, False),
            ({}, True, False),
        ]
        for overrides, skip_tokenizer, expected in cases:
            with self.subTest(overrides=overrides, skip_tokenizer=skip_tokenizer):
                config = dict(nnodes=2, tp_size=4, pp_size=1, attn_dp_enabled=False)
                config.update(overrides)
                with (
                    patch.dict(os.environ, {"SGLANG_NODE_FANOUT_REQ_BROADCAST": "1"}),
                    patch(
                        "sglang.srt.managers.scheduler_components.node_fanout.get_parallel",
                        return_value=SimpleNamespace(**config),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.node_fanout.get_serving",
                        return_value=SimpleNamespace(
                            skip_tokenizer_init=skip_tokenizer
                        ),
                    ),
                ):
                    self.assertEqual(node_fanout_enabled(), expected)

    def test_pixels_reach_every_rank_through_node_shm(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _run_rank, args=(str(Path(directory) / "gloo-init"),), nprocs=WORLD
            )

    def test_handles_are_small_on_the_wire(self):
        req = _request("vlm", [torch.zeros((640, 640, 3), dtype=torch.uint8)])
        from sglang.srt.managers.mm_utils import wrap_shm_features

        wrap_shm_features(req, force=True)
        try:
            self.assertLess(len(pickle.dumps(req)), 8192)
        finally:
            for it in req.mm_inputs.mm_items:
                it.feature.close_and_unlink()


if __name__ == "__main__":
    unittest.main()
