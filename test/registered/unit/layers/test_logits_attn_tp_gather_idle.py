"""LogitsProcessor._gather_attn_tp_logits with an empty (idle DP rank) input.

Under DP attention a rank with no requests still runs a forward so collectives
stay in lockstep, and its unpadded logits have zero rows. XLA rejects an
all-gather whose operand is zero-length along the gather dim, so the gather has
to be skipped rather than issued -- on TPU this crashed every idle rank as soon
as a dp_size>1 server took its first request (a single health-check request
leaves dp_size-1 groups idle, so it fails immediately, not rarely).
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

VOCAB_SIZE = 32
ATTN_TP_SIZE = 4


def _processor():
    proc = object.__new__(LogitsProcessor)
    proc.vocab_size = VOCAB_SIZE
    proc.attn_tp_size = ATTN_TP_SIZE
    return proc


class TestGatherAttnTpLogitsIdle(CustomTestCase):
    def test_empty_logits_skip_the_all_gather(self):
        proc = _processor()
        logits = torch.empty((0, VOCAB_SIZE // ATTN_TP_SIZE))

        with patch(
            "sglang.srt.layers.logits_processor.attn_tp_all_gather_into_tensor"
        ) as gather, patch(
            "sglang.srt.layers.logits_processor.attn_tp_all_gather"
        ) as gather_list:
            out = proc._gather_attn_tp_logits(logits)

        gather.assert_not_called()
        gather_list.assert_not_called()
        # Shape must still be the full vocab so downstream consumers see a
        # well-formed empty result rather than a shard width.
        self.assertEqual(out.shape, (0, VOCAB_SIZE))
        self.assertEqual(out.dtype, logits.dtype)

    def test_empty_logits_skip_even_when_vocab_is_indivisible(self):
        # The indivisible branch uses tensor_split + attn_tp_all_gather, which
        # has the same zero-length problem; the guard must precede both.
        proc = _processor()
        proc.vocab_size = VOCAB_SIZE + 1
        logits = torch.empty((0, 8))

        with patch(
            "sglang.srt.layers.logits_processor.attn_tp_all_gather_into_tensor"
        ) as gather, patch(
            "sglang.srt.layers.logits_processor.attn_tp_all_gather"
        ) as gather_list:
            out = proc._gather_attn_tp_logits(logits)

        gather.assert_not_called()
        gather_list.assert_not_called()
        self.assertEqual(out.shape, (0, VOCAB_SIZE + 1))

    def test_nonempty_logits_still_gather(self):
        proc = _processor()
        shard = VOCAB_SIZE // ATTN_TP_SIZE
        logits = torch.zeros((3, shard))

        with patch(
            "sglang.srt.layers.logits_processor.attn_tp_all_gather_into_tensor"
        ) as gather:
            out = proc._gather_attn_tp_logits(logits)

        gather.assert_called_once()
        destination = gather.call_args.args[0]
        self.assertEqual(destination.shape, (ATTN_TP_SIZE, 3, shard))
        self.assertEqual(out.shape, (3, VOCAB_SIZE))


if __name__ == "__main__":
    unittest.main()
