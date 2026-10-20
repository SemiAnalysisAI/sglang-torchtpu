# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for shared chain acceptance."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.experimental.speculative.spec_types import (
    BucketState,
    DraftResult,
    SpecState,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChainComponents(unittest.TestCase):
    def test_greedy_verify_commits_live_rows_and_preserves_bucket_rows(self):
        worker = object.__new__(TpuBlockSpecWorker)
        worker.num_verify_rows = 3
        worker._accept_threshold_single = worker._accept_threshold_acc = 1.0
        worker._simulate_acc_len = 0
        worker._tp_sync = SimpleNamespace(sync=Mock())
        winners = torch.tensor([[1, 2, 3], [3, 2, 1]])
        logits = torch.nn.functional.one_hot(winners, 5).float().reshape(6, 5)
        worker._forward_target_verify = Mock(
            return_value=LogitsProcessorOutput(next_token_logits=logits)
        )
        state = SpecState(
            torch.tensor([0, 0]), torch.tensor([10, 0]), torch.tensor([2, 0])
        )
        bucket = BucketState(
            torch.tensor([2]), state, torch.tensor([True, False]), None, None
        )
        draft = DraftResult(
            torch.tensor([[1, 2], [3, 2]]),
            None,
            torch.tensor([2, 2], dtype=torch.int32),
        )
        out, accept = worker.verify(None, bucket, None, draft, None)
        self.assertIs(out.next_token_logits, logits)
        self.assertEqual(accept.accept_tokens.tolist(), [[1, 2, 3], [3, 2, 1]])
        self.assertEqual(accept.accept_lens.tolist(), [3, 0])
        self.assertEqual(accept.next_seq_lens.tolist(), [13, 0])
        self.assertEqual(worker._tp_sync.sync.call_count, 2)


if __name__ == "__main__":
    unittest.main()
