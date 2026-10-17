"""CPU contracts for the shared speculative-decoding functions in spec_common."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.spec_common import adjust_verify_logits
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAdjustVerifyLogits(CustomTestCase):
    def test_per_request_terms_cover_its_verify_rows_and_skip_bucket_padding(self):
        bs, draft_token_num, vocab, padding = 2, 3, 4, 2
        logits = torch.linspace(-3.0, 3.0, (bs * draft_token_num + padding) * vocab)
        logits = logits.view(-1, vocab)
        original = logits.clone()
        additive = torch.tensor([[0.5, -0.5, 1.0, 0.0], [2.0, 0.0, -1.0, 0.5]])
        scaling = torch.tensor([[2.0, 1.0, 4.0, 1.0], [1.0, 2.0, 1.0, 8.0]])
        bias = torch.tensor([[0.0, 1.0, 0.0, -1.0], [3.0, 0.0, 0.0, 0.0]])

        adjust_verify_logits(
            next_token_logits=logits,
            sampling_info=SimpleNamespace(
                acc_additive_penalties=additive,
                acc_scaling_penalties=scaling,
                logit_bias=bias,
            ),
            draft_token_num=draft_token_num,
        )

        for row in range(bs * draft_token_num):
            req = row // draft_token_num
            shifted = original[row] + additive[req]
            # Scaling penalties shrink positive logits and push negative ones further.
            scaled = torch.where(
                shifted < 0, shifted * scaling[req], shifted / scaling[req]
            )
            torch.testing.assert_close(logits[row], scaled + bias[req])
        torch.testing.assert_close(
            logits[bs * draft_token_num :], original[bs * draft_token_num :]
        )


if __name__ == "__main__":
    unittest.main()
