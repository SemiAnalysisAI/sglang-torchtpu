"""Each n > 1 sub-request, the prefix request included, gets its own bootstrap room.

The rooms are pinned exactly: a prefill and a decode on different builds must still
derive the same rooms from one body.
"""

import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
)
from sglang.srt.observability.trace import TraceNullContext  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# The room a PD router draws for one HTTP request.
ROUTER_ROOM = 1000
MAX_NEW_TOKENS = 32
PD_MODES = (DisaggregationMode.PREFILL, DisaggregationMode.DECODE)


def _body(n, input_ids=(11, 12, 13), bootstrap_room=ROUTER_ROOM):
    """What a PD router sends to both the prefill and the decode."""
    body = {
        "input_ids": list(input_ids),
        "sampling_params": {"n": n, "max_new_tokens": MAX_NEW_TOKENS},
    }
    if bootstrap_room is not None:
        host, port = "127.0.0.1", 8998
        if isinstance(bootstrap_room, list):
            host, port = [host] * len(bootstrap_room), [port] * len(bootstrap_room)
        body.update(
            bootstrap_host=host, bootstrap_port=port, bootstrap_room=bootstrap_room
        )
    return body


class _RoomTraceContext(TraceNullContext):
    """A disabled trace context that keeps the room it was opened with."""

    def __init__(self, bootstrap_room):
        self.bootstrap_room = bootstrap_room


def _fan_out(body, mode):
    """Run TokenizerManager's n > 1 fan-out on ``body`` in the ``mode`` role.

    Returns (bootstrap_room, max_new_tokens) per dispatched sub-request, in
    dispatch order: the prefix requests, then the samples of each prompt. Also
    returns the room of the trace context that each one carries to the scheduler.
    """
    dispatched = []
    traced = []

    def init_trace_ctx(self, rid, bootstrap_room, external_trace_header=None):
        self.trace_ctx = _RoomTraceContext(bootstrap_room)

    with (
        published_topology(disaggregation_mode=mode.value),
        patch.object(APIServerReqTimeStats, "init_trace_ctx", init_trace_ctx),
    ):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.init_disaggregation(start_pd_bootstrap_service=False)
        tm.rid_to_state = {}
        tm.enable_trace = True
        tm.preferred_sampling_params = None
        tm.sampling_params_class = SamplingParams
        tm.tokenizer = None
        tm.model_config = SimpleNamespace(vocab_size=1000)

        async def tokenize(obj):
            return tm._create_tokenized_object(obj, None, obj.input_ids)

        async def send(tokenized_obj):
            dispatched.append(
                (
                    tokenized_obj.bootstrap_room,
                    tokenized_obj.sampling_params.max_new_tokens,
                )
            )
            traced.append(tokenized_obj.time_stats.trace_ctx.bootstrap_room)

        async def wait(obj, request=None):
            yield {"meta_info": {"id": obj.rid}, "text": ""}

        tm._tokenize_one_request = tokenize
        tm._send_one_request = send
        tm._wait_one_response = wait

        obj = GenerateReqInput(**copy.deepcopy(body))
        obj.normalize_batch_and_arguments()
        tm._init_req_state(obj)
        request_states = {rid: tm.rid_to_state[rid] for rid in obj.rid}

        async def drive():
            async for _ in tm._handle_batch_request(obj, None, request_states):
                pass

        asyncio.run(drive())
    return dispatched, traced


class TestParallelSamplingBootstrapRooms(CustomTestCase):
    def assert_fan_out(self, body, modes, prefix_rooms, sample_rooms):
        expected = [(room, 0) for room in prefix_rooms]
        expected += [(room, MAX_NEW_TOKENS) for room in sample_rooms]
        for mode in modes:
            with self.subTest(mode=mode.value):
                dispatched, traced = _fan_out(body, mode)
                self.assertEqual(dispatched, expected)
                self.assertEqual(traced, [room for room, _ in expected])

    def test_prefix_request_and_each_sample_get_their_own_room(self):
        # Normalization reserves ROUTER_ROOM + k for sample k.
        self.assert_fan_out(
            _body(n=2),
            PD_MODES,
            prefix_rooms=[ROUTER_ROOM + 2],
            sample_rooms=[ROUTER_ROOM, ROUTER_ROOM + 1],
        )

    def test_batch_prefix_requests_and_samples_get_their_own_rooms(self):
        r = ROUTER_ROOM
        cases = (
            # One room for the batch, which normalization expands to r + i for
            # prompt i.
            (r, [r + 6, r + 7], [r, r + 2, r + 4, r + 1, r + 3, r + 5]),
            # One room per prompt, as the test router sends for a batch. 0 is a
            # valid room.
            ([0, 200], [6, 206], [0, 2, 4, 200, 202, 204]),
        )
        for bootstrap_room, prefix_rooms, sample_rooms in cases:
            with self.subTest(bootstrap_room=bootstrap_room):
                body = _body(
                    n=3, input_ids=[[11, 12], [21, 22]], bootstrap_room=bootstrap_room
                )
                self.assert_fan_out(body, PD_MODES, prefix_rooms, sample_rooms)

    def test_gateway_room_list_for_one_prompt_uses_its_first_room(self):
        # sgl-model-gateway sends n rooms for one chat prompt with n > 1.
        self.assert_fan_out(
            _body(n=3, bootstrap_room=[100, 900, 500]),
            PD_MODES,
            prefix_rooms=[103],
            sample_rooms=[100, 101, 102],
        )

    def test_requests_without_a_room_keep_none(self):
        self.assert_fan_out(
            _body(n=2, bootstrap_room=None),
            (DisaggregationMode.NULL,),
            prefix_rooms=[None],
            sample_rooms=[None, None],
        )


if __name__ == "__main__":
    unittest.main()
