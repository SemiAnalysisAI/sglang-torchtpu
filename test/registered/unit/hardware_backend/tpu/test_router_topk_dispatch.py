"""CPU checks of how core top-k hands decode routing to the TPU router kernel."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.moe import router_topk_tpu

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOKENS, EXPERTS, K = 1, 256, 8


def _inputs(bias_rows=None):
    logits = torch.randn(TOKENS, EXPERTS)
    shape = (EXPERTS,) if bias_rows is None else (bias_rows, EXPERTS)
    return logits, torch.randn(shape)


class TestRouterTopkDispatch(CustomTestCase):
    def setUp(self):
        self.kernel_out = (
            torch.ones(TOKENS, K),
            torch.zeros(TOKENS, K, dtype=torch.int32),
        )
        self.op = lambda *args: self.kernel_out

    def _grouped(self, logits, bias, num_expert_group=1, topk_group=1, shared=0):
        return topk.biased_grouped_topk_gpu(
            logits,
            logits,
            bias,
            K,
            True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_fused_shared_experts=shared,
        )

    def test_single_group_takes_the_kernel(self):
        logits, bias = _inputs()
        with (
            patch.object(topk, "_is_tpu", True),
            patch.object(router_topk_tpu, "_OP", [self.op]),
            patch.object(router_topk_tpu, "_OP_READY", True),
        ):
            out = self._grouped(logits, bias)
        self.assertIs(out, self.kernel_out)

    def test_ineligible_inputs_keep_the_sort(self):
        cases = {
            "several groups": dict(num_expert_group=8, topk_group=4),
            "fused shared expert": dict(shared=1),
            "per-token bias": dict(bias_rows=TOKENS),
            "op not registered": dict(ready=False),
        }
        for name, case in cases.items():
            with self.subTest(name):
                logits, bias = _inputs(case.pop("bias_rows", None))
                ready = case.pop("ready", True)
                with (
                    patch.object(topk, "_is_tpu", True),
                    patch.object(router_topk_tpu, "_OP", [self.op]),
                    patch.object(router_topk_tpu, "_OP_READY", ready),
                    patch.object(
                        topk, "biased_grouped_topk_impl", return_value="sorted"
                    ),
                ):
                    out = self._grouped(logits, bias, **case)
                self.assertEqual(out, "sorted")

    def test_too_many_tokens_keep_the_sort(self):
        logits = torch.randn(router_topk_tpu.MAX_TOKENS + 1, EXPERTS)
        bias = torch.randn(EXPERTS)
        with (
            patch.object(router_topk_tpu, "_OP", [self.op]),
            patch.object(router_topk_tpu, "_OP_READY", True),
        ):
            out = router_topk_tpu.router_topk(logits, bias, K, True, 0, None, False)
        self.assertIsNone(out)

    def test_routed_scaling_reaches_the_kernel_only_on_output(self):
        logits, bias = _inputs()
        seen = []
        op = lambda *args: seen.append(args[-1]) or self.kernel_out
        with (
            patch.object(router_topk_tpu, "_OP", [op]),
            patch.object(router_topk_tpu, "_OP_READY", True),
        ):
            router_topk_tpu.router_topk(logits, bias, K, True, 0, 2.5, True)
            router_topk_tpu.router_topk(logits, bias, K, True, 0, 2.5, False)
        self.assertEqual(seen, [2.5, 1.0])


if __name__ == "__main__":
    unittest.main()
