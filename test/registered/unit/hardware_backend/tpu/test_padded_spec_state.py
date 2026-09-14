"""Physical continuation must preserve real rows and keep dummy KV rows inert."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.padded_spec_state import (
    VerifyContinuation,
    install_verify_continuation_tpu,
    prepare_extend_continuation_tpu,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPaddedSpecState(CustomTestCase):
    def test_tpu_equal_size_padding_preserves_owned_storage(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        tensor = torch.arange(12).reshape(4, 3)
        with patch("sglang.srt.model_executor.forward_batch_info._is_tpu", True):
            same = ForwardBatch._pad_tensor_to_size(None, tensor, 4)
            larger = ForwardBatch._pad_tensor_to_size(None, tensor, 8)
        self.assertIs(same, tensor)
        torch.testing.assert_close(larger[:4], tensor)
        self.assertEqual(larger[4:].count_nonzero().item(), 0)

    def test_dummy_sampling_acceptance_cannot_create_kv_length(self):
        # Sampling can accept every dummy token. Those rows must still have
        # zero length/count when the next attention operation sees them.
        for logical_bs in (1, 3, 7, 12, 15):
            for count in (1, 2, 4):
                with self.subTest(logical_bs=logical_bs, count=count):
                    prefix = torch.zeros(16, dtype=torch.int64)
                    prefix[:logical_bs] = torch.arange(1, logical_bs + 1)
                    accepts = torch.full((16,), count, dtype=torch.int32)
                    tokens = torch.arange(64, dtype=torch.int32)
                    result, counts, correct, lengths, hidden = (
                        prepare_extend_continuation_tpu(
                            tokens, accepts, prefix, 4, torch.ones(64, 3)
                        )
                    )
                    torch.testing.assert_close(
                        result[: logical_bs * 4],
                        tokens[: logical_bs * 4].to(torch.int64),
                    )
                    self.assertEqual(result[logical_bs * 4 :].count_nonzero().item(), 0)
                    self.assertEqual(hidden[logical_bs * 4 :].count_nonzero().item(), 0)
                    torch.testing.assert_close(
                        counts[:logical_bs], accepts[:logical_bs]
                    )
                    torch.testing.assert_close(
                        correct[:logical_bs], accepts[:logical_bs] - 1
                    )
                    torch.testing.assert_close(
                        lengths[:logical_bs], prefix[:logical_bs] + 4
                    )
                    for tensor in (counts, correct, lengths):
                        self.assertEqual(tensor[logical_bs:].count_nonzero().item(), 0)

    def test_physical_hidden_storage_survives_logical_result_rebinding(self):
        hidden = torch.arange(64 * 3).reshape(64, 3)
        state = VerifyContinuation(
            logical_batch_size=12,
            physical_batch_size=16,
            hidden_states=hidden,
            predict=torch.arange(64, dtype=torch.int32),
            accept_lens=torch.ones(16, dtype=torch.int32),
            seq_lens=torch.tensor([10] * 12 + [0] * 4),
            bonus_tokens=torch.zeros(16, dtype=torch.int32),
        )
        logical_result = SimpleNamespace(hidden_states=hidden[:48])
        forward = SimpleNamespace(spec_info=SimpleNamespace())
        install_verify_continuation_tpu(forward, state, 4)
        logical_result.hidden_states = torch.empty(0)
        torch.testing.assert_close(forward.spec_info.hidden_states[:48], hidden[:48])
        self.assertEqual(forward.spec_info.hidden_states[48:].count_nonzero().item(), 0)
        self.assertEqual(forward.input_ids.shape, (64,))
        self.assertEqual(forward.seq_lens.shape, (12,))
        self.assertEqual(forward.spec_info.num_accept_tokens.shape, (16,))
        self.assertEqual(forward.spec_info.num_accept_tokens[12:].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
