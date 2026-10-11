# SPDX-License-Identifier: Apache-2.0
"""The rank-based router kernel must select the XLA path's experts with its weights."""

import unittest

import torch

from sglang.srt.layers.moe import topk as topk_mod
from sglang.test.ci.ci_register import register_tpu_ci

try:
    import torch_tpu  # noqa: F401

    from sglang.srt.hardware_backend.tpu import router_topk
except ImportError:
    torch_tpu = None

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

# GLM-5.3-Flash: 288 experts, top-8, one group, routed scaling 2.5.
EXPERTS, TOPK, SCALE = 288, 8, 2.5


@unittest.skipUnless(torch_tpu is not None, "Requires torch_tpu")
class TestRouterTopkSmallM(unittest.TestCase):
    def route(self, logits, bias, apply_scale):
        return topk_mod.biased_grouped_topk_impl(
            logits,
            logits,
            bias,
            topk=TOPK,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=SCALE,
            apply_routed_scaling_factor_on_output=apply_scale,
        )

    def by_expert(self, weights, ids):
        # Slot order is unspecified (torch.topk runs unsorted); the combine
        # sums over slots, so compare per expert.
        order = ids.argsort(dim=-1)
        return ids.gather(-1, order), weights.gather(-1, order)

    def test_kernel_matches_xla_path(self):
        router_topk.register_router_topk_op()
        op = router_topk._OP
        gen = torch.Generator().manual_seed(11)
        # One decode token, and an 8-token speculative verify window.
        for trial in range(20):
            tokens = 1 if trial % 2 == 0 else 8
            for apply_scale in (False, True):
                with self.subTest(trial=trial, tokens=tokens, apply_scale=apply_scale):
                    logits = (torch.randn(tokens, EXPERTS, generator=gen) * 2).to("tpu")
                    bias = (torch.randn(EXPERTS, generator=gen) * 0.1).to("tpu")
                    try:
                        router_topk._OP = None
                        w_ref, ids_ref = self.route(logits, bias, apply_scale)
                    finally:
                        router_topk._OP = op
                    w, ids = self.route(logits, bias, apply_scale)
                    self.assertEqual((w.dtype, ids.dtype), (torch.float32, torch.int32))
                    ids_ref, w_ref = self.by_expert(w_ref.cpu(), ids_ref.cpu().long())
                    ids, w = self.by_expert(w.cpu(), ids.cpu().long())
                    self.assertTrue(torch.equal(ids, ids_ref))
                    torch.testing.assert_close(w, w_ref, rtol=2e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
