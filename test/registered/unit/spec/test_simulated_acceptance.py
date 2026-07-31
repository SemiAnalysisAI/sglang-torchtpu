import unittest

import torch
from sglang.srt.speculative.spec_utils import generate_simulated_accept_index
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSimulatedAcceptance(unittest.TestCase):
    def test_accept_index_uses_input_device(self):
        accept_index = torch.tensor(
            [[5, -1, -1, -1], [10, -1, -1, -1]], dtype=torch.int32
        )
        predict = torch.zeros(16, dtype=torch.int64)
        num_correct_drafts = torch.zeros(2, dtype=torch.int32)

        simulated = generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predict,
            num_correct_drafts=num_correct_drafts,
            candidates=None,
            target_predict=None,
            bs=2,
            spec_steps=3,
            simulate_acc_len=3,
            simulate_acc_method="match-expected",
            simulate_acc_token_mode="fixed",
        )

        self.assertEqual(simulated.device, accept_index.device)
        torch.testing.assert_close(
            simulated,
            torch.tensor([[5, 6, 7, -1], [10, 11, 12, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            num_correct_drafts, torch.tensor([2, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(predict, torch.full_like(predict, 100))


if __name__ == "__main__":
    unittest.main()
