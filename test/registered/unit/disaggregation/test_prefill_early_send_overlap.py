"""Overlap-mode early send must not offer in-flight pages to a reads-on-send
transport (issue #984).

``maybe_send_cached_prefix_chunk`` runs before a batch's forward is launched.
For a chunked request its prefix already includes the previous chunk, whose
forward may still be running under overlap scheduling. CUDA guards this with a
completion event; elsewhere a sender that reads offered pages at once (raiden
CHUNK) must not get the early send, or it copies stale KV.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import prefill as prefill_mod
from sglang.srt.disaggregation.base.conn import BaseKVSender
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE = 128
EARLY_SEND_ENV = "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX"


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, *, enable_overlap):
        self.enable_overlap = enable_overlap
        self.enable_staging = False
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=PAGE)
        self.forward_stream = Mock()
        self.send_kv_chunk = Mock()


def _chunked_req(*, reads_on_send, prefix_tokens=4096):
    """A request whose first chunk ran: its prefix is that chunk's pages."""
    return SimpleNamespace(
        rid="r",
        pending_bootstrap=False,
        prefix_indices=list(range(prefix_tokens)),
        host_hit_length=0,
        start_send_idx=0,
        early_send_prefix_end=None,
        disagg_kv_sender=SimpleNamespace(reads_on_send=reads_on_send),
    )


@patch.dict(os.environ, {EARLY_SEND_ENV: "true"})
class TestEarlySendUnderOverlap(unittest.TestCase):
    def test_reads_on_send_sender_gets_no_early_send_under_overlap(self):
        sched = _Scheduler(enable_overlap=True)
        req = _chunked_req(reads_on_send=True)
        with patch.object(prefill_mod, "_is_cuda", False):
            sched.maybe_send_cached_prefix_chunk(req)
        sched.send_kv_chunk.assert_not_called()
        # Nothing was recorded as sent, so the post-copy_done send in
        # process_batch_result_disagg_prefill still covers these pages.
        self.assertEqual(req.start_send_idx, 0)

    def test_other_senders_keep_early_send_under_overlap(self):
        sched = _Scheduler(enable_overlap=True)
        req = _chunked_req(reads_on_send=False)
        with patch.object(prefill_mod, "_is_cuda", False):
            sched.maybe_send_cached_prefix_chunk(req)
        sched.send_kv_chunk.assert_called_once_with(req, last_chunk=False, end_idx=4096)

    def test_reads_on_send_sender_keeps_early_send_without_overlap(self):
        # Without overlap the previous batch is fully processed before the
        # next run_batch, so its pages are written when they are offered.
        sched = _Scheduler(enable_overlap=False)
        req = _chunked_req(reads_on_send=True)
        with patch.object(prefill_mod, "_is_cuda", False):
            sched.maybe_send_cached_prefix_chunk(req)
        sched.send_kv_chunk.assert_called_once_with(req, last_chunk=False, end_idx=4096)

    def test_cuda_keeps_event_guarded_early_send(self):
        sched = _Scheduler(enable_overlap=True)
        req = _chunked_req(reads_on_send=True)
        event = Mock()
        with (
            patch.object(prefill_mod, "_is_cuda", True),
            patch.object(prefill_mod.torch.cuda, "Event", return_value=event),
        ):
            sched.maybe_send_cached_prefix_chunk(req)
        event.record.assert_called_once_with(sched.forward_stream)
        self.assertIs(req.disagg_kv_sender._early_send_wait_event, event)
        sched.send_kv_chunk.assert_called_once_with(req, last_chunk=False, end_idx=4096)


class TestReadsOnSendCapability(unittest.TestCase):
    def test_default_is_false(self):
        self.assertFalse(BaseKVSender.reads_on_send)

    def test_raiden_sender_reads_on_send(self):
        from sglang.srt.disaggregation.raiden.sender import RaidenKVSender

        self.assertTrue(RaidenKVSender.reads_on_send)


if __name__ == "__main__":
    unittest.main()
