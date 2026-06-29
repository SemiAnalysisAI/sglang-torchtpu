"""Unit tests for the native MoE TPU-layout / no_combine additions.

Covers:
- ``_linear_maybe_transposed_weight``: picks matmul vs F.linear by layout.
- ``moe_forward_native`` with ``no_combine``: returns per-expert
  ``[tokens, topk, hidden]`` and combining it with topk_weights equals the
  default combined path.

All CPU-runnable (no TPU hardware needed).
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.fused_moe_native import (
    _linear_maybe_transposed_weight,
    moe_forward_native,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Layer(torch.nn.Module):
    """Minimal stand-in for a FusedMoE layer for moe_forward_native."""

    def __init__(self, w13_weight, w2_weight, num_experts):
        super().__init__()
        self.w13_weight = w13_weight
        self.w2_weight = w2_weight
        self.num_experts = num_experts


class TestLinearMaybeTransposedWeight(CustomTestCase):
    def test_out_in_layout_uses_f_linear(self):
        x = torch.randn(3, 4)
        # standard (out, in) layout: out=5, in=4
        w = torch.randn(5, 4)
        got = _linear_maybe_transposed_weight(x, w)
        torch.testing.assert_close(got, F.linear(x, w))
        self.assertEqual(got.shape, (3, 5))

    def test_in_out_layout_uses_matmul(self):
        x = torch.randn(3, 4)
        # transposed (in, out) layout: in=4, out=5
        w = torch.randn(4, 5)
        got = _linear_maybe_transposed_weight(x, w)
        torch.testing.assert_close(got, x @ w)
        self.assertEqual(got.shape, (3, 5))


def _topk_output(topk_weights, topk_ids):
    # StandardTopKOutput is a (weights, ids, router_logits)-style tuple.
    return (topk_weights, topk_ids, None)


class TestNativeMoeNoCombine(CustomTestCase):
    def setUp(self):
        # Activation modules (SiluAndMul) read the global server args at
        # construction time.
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )

    def _setup(self):
        torch.manual_seed(0)
        num_experts, hidden, inter, tokens, topk = 4, 8, 6, 4, 2
        # (out, in) layout for the default native path.
        w13 = torch.randn(num_experts, 2 * inter, hidden)
        w2 = torch.randn(num_experts, hidden, inter)
        layer = _Layer(w13, w2, num_experts)
        x = torch.randn(tokens, hidden)
        # Deterministic routing where every expert is used and each token's
        # top-k picks are distinct (avoids the num_tokens==0 / duplicate-expert
        # edges in the reference moe_forward_native loop).
        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64
        )
        topk_weights = torch.rand(tokens, topk)
        return layer, x, topk_ids, topk_weights, hidden, topk

    def test_no_combine_returns_per_expert_and_recombines(self):
        layer, x, topk_ids, topk_weights, hidden, topk = self._setup()

        combined_cfg = MoeRunnerConfig(activation="silu", no_combine=False)
        nocombine_cfg = MoeRunnerConfig(activation="silu", no_combine=True)

        combined = moe_forward_native(
            layer, x, _topk_output(topk_weights, topk_ids), combined_cfg
        )
        per_expert = moe_forward_native(
            layer, x, _topk_output(topk_weights, topk_ids), nocombine_cfg
        )

        self.assertEqual(per_expert.shape, (x.shape[0], topk, hidden))
        self.assertEqual(combined.shape, (x.shape[0], hidden))

        # Re-combine the per-expert output the way llama4x's post-norm combine
        # would; it must reproduce the default combined path.
        recombined = (
            per_expert.type(topk_weights.dtype)
            .mul(topk_weights.unsqueeze(-1))
            .sum(dim=1)
            .type(per_expert.dtype)
        )
        torch.testing.assert_close(recombined, combined, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
