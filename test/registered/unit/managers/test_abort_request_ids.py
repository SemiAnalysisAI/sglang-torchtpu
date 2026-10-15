"""Cancellation must not match unrelated request IDs that share a prefix."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.constrained.grammar_manager import GrammarManager  # noqa: E402
from sglang.srt.disaggregation.encoder.receiver import (  # noqa: E402
    MMReceiverBase,
    WaitingMMRequestStatus,
)
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.dllm.mixin.scheduler import DllmManager  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq, GenerateReqInput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)
        self.disagg_kv_sender = Mock()
        self.to_finish = None
        self.grammar = None
        self.set_finish_with_abort = Mock()

    def finished(self):
        return False


def _scheduler():
    sched = Scheduler.__new__(Scheduler)
    sched.chunked_req = None
    sched._pending_chunked_abort_req = None
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.mm_receiver = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = None
    sched.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
    sched.disagg_prefill_inflight_queue = []
    sched.disagg_decode_prealloc_queue = SimpleNamespace(queue=[], retracted_queue=[])
    sched.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
    sched._release_aborted_request = Mock()
    sched.release_aborted_prefill_waiting_req = Mock()
    sched.beam_coordinator = Mock()
    sched.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())
    sched.tree_cache = Mock()
    sched.collect_inflight_reqs = Mock(return_value=[])
    return sched


class TestAbortRequestIds(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_scheduler_queues_match_exact_ids_and_abort_all(self):
        for queue in (
            "waiting",
            "chunked",
            "running",
            "prefill_bootstrap",
            "prefill_transfer",
            "decode_prealloc",
            "decode_transfer",
            "decode_retracted",
        ):
            for abort_all in (False, True):
                with self.subTest(queue=queue, abort_all=abort_all):
                    reqs = [
                        _Req(rid) for rid in ("req-2", "req-20", "req-200", "req-2_0")
                    ]
                    sched = _scheduler()
                    wrappers = []
                    if queue == "waiting":
                        sched.waiting_queue = list(reqs)
                    elif queue == "chunked":
                        # The chunked slot is not the aborted request.
                        sched.chunked_req = reqs[1]
                    elif queue == "running":
                        sched.collect_inflight_reqs.return_value = reqs
                    elif queue.startswith("prefill"):
                        sched.disaggregation_mode = DisaggregationMode.PREFILL
                        if queue == "prefill_bootstrap":
                            sched.disagg_prefill_bootstrap_queue.queue = reqs
                        else:
                            sched.disagg_prefill_inflight_queue = reqs
                    else:
                        sched.disaggregation_mode = DisaggregationMode.DECODE
                        wrappers = [
                            SimpleNamespace(
                                req=req, kv_receiver=Mock(), host_staged=False
                            )
                            for req in reqs
                        ]
                        if queue == "decode_prealloc":
                            sched.disagg_decode_prealloc_queue.queue = wrappers
                        elif queue == "decode_transfer":
                            sched.disagg_decode_transfer_queue.queue = wrappers
                        else:
                            sched.disagg_decode_prealloc_queue.retracted_queue = list(
                                reqs
                            )
                    with (
                        patch(
                            "sglang.srt.managers.scheduler._make_abort_req",
                            side_effect=lambda req, **kw: AbortReq(rid=req.rid),
                        ),
                        patch("sglang.srt.managers.scheduler.discard_kv_cache_backup"),
                    ):
                        sched.abort_request(AbortReq(rid="req-2", abort_all=abort_all))
                    expected = reqs if abort_all else reqs[:1]
                    if queue == "waiting":
                        self.assertEqual(
                            sched.waiting_queue, [] if abort_all else reqs[1:]
                        )
                    elif queue == "chunked":
                        self.assertIs(
                            sched._pending_chunked_abort_req,
                            reqs[1] if abort_all else None,
                        )
                    elif queue == "running":
                        self.assertEqual(
                            [req for req in reqs if req.to_finish is not None], expected
                        )
                    elif queue.startswith("prefill"):
                        self.assertEqual(
                            [req for req in reqs if req.disagg_kv_sender.abort.called],
                            expected,
                        )
                    elif queue == "decode_retracted":
                        self.assertEqual(
                            sched.disagg_decode_prealloc_queue.retracted_queue,
                            [] if abort_all else reqs[1:],
                        )
                    else:
                        self.assertEqual(
                            [
                                wrapper.req
                                for wrapper in wrappers
                                if wrapper.kv_receiver.abort.called
                            ],
                            expected,
                        )

    def test_auxiliary_queues_match_exact_ids_and_abort_all(self):
        for abort_all in (False, True):
            with self.subTest(abort_all=abort_all):
                reqs = [_Req(rid) for rid in ("req-2", "req-20", "req-2_0")]
                expected = reqs if abort_all else reqs[:1]
                abort = AbortReq(rid="req-2", abort_all=abort_all)
                grammar = GrammarManager.__new__(GrammarManager)
                grammar.grammar_queue = reqs
                grammar.abort_requests(abort)
                self.assertEqual(
                    [req for req in reqs if req.set_finish_with_abort.called], expected
                )
                dllm = DllmManager.__new__(DllmManager)
                dllm.waiting_queue = list(reqs)
                dllm.staging_queue = list(reqs)
                self.assertEqual(dllm.pop_aborted_reqs(abort_all, "req-2"), expected)
                self.assertEqual(dllm.waiting_queue, [] if abort_all else reqs[1:])
                self.assertEqual(dllm.staging_queue, [] if abort_all else reqs[1:])
                waiting = [
                    SimpleNamespace(
                        rid=req.rid,
                        status=WaitingMMRequestStatus.PENDING,
                        _fail_and_release=Mock(),
                        release_resources=Mock(),
                    )
                    for req in reqs
                ]
                MMReceiverBase.abort_waiting_requests(
                    SimpleNamespace(waiting_list=waiting), abort
                )
                self.assertEqual(
                    [req.rid for req in waiting if req.release_resources.called],
                    [req.rid for req in expected],
                )

    def test_batch_disconnect_enumerates_exact_member_ids(self):
        obj = GenerateReqInput(input_ids=[[1], [2]], rid="batch", stream=True)
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.rid, ["batch_0", "batch_1"])
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.rid_to_state = {
            rid: SimpleNamespace(abort_sent=False)
            for rid in [*obj.rid, "batch_10", "batch_0_extra"]
        }
        tm._dispatch_to_scheduler = Mock()
        tm.enable_metrics = False
        with patch(
            "sglang.srt.managers.tokenizer_manager.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            asyncio.run(tm.create_abort_task(obj)())
        self.assertEqual(
            [call.args[0].rid for call in tm._dispatch_to_scheduler.call_args_list],
            obj.rid,
        )
        self.assertFalse(tm.rid_to_state["batch_10"].abort_sent)
        self.assertFalse(tm.rid_to_state["batch_0_extra"].abort_sent)


if __name__ == "__main__":
    unittest.main()
