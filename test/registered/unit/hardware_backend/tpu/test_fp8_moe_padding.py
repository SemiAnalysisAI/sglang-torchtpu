"""FP8 EP padding must be ignored without changing valid expert IDs/weights."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.fp8_moe import apply_tpu_fp8_moe
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFp8MoePadding(CustomTestCase):
    def test_padding_sentinel_is_opt_in_and_ep_only(self):
        cases = [(False, 16, 0), (True, 1, 0), (True, 16, 0), (True, 16, 7)]
        for enabled, ep_size, ep_rank in cases:
            for no_combine in (False, True):
                with self.subTest(
                    enabled=enabled,
                    ep_size=ep_size,
                    rank=ep_rank,
                    no_combine=no_combine,
                ):
                    x = torch.ones(2, 8)
                    ids = torch.tensor(
                        [[0, 1, 2, 3, 4, 5, 6, 7], [-1] * 8], dtype=torch.int32
                    )
                    weights = torch.full((2, 8), 0.125)
                    ids_before, weights_before = ids.clone(), weights.clone()
                    layer = SimpleNamespace(
                        num_local_experts=16,
                        num_fused_shared_experts=0,
                        w13_weight=torch.empty(16, 8, 16),
                        w2_weight=torch.empty(16, 8, 8),
                        w13_weight_scale=torch.ones(16, 1, 1, 16),
                        w2_weight_scale=torch.ones(16, 1, 1, 8),
                        moe_runner_config=SimpleNamespace(swiglu_limit=None),
                    )
                    captured = {}

                    def kernel(**kwargs):
                        captured.update(kwargs)
                        return torch.zeros_like(x)

                    with (
                        patch.dict(
                            sys.modules,
                            {
                                "sglang.srt.layers.moe.pallas.fused_moe_op": (
                                    SimpleNamespace(fused_moe_call=kernel)
                                ),
                            },
                        ),
                        patch(
                            "sglang.srt.hardware_backend.tpu.fp8_moe.get_parallel",
                            return_value=SimpleNamespace(
                                moe_ep_size=ep_size, moe_ep_rank=ep_rank
                            ),
                        ),
                        patch(
                            "sglang.srt.hardware_backend.tpu.fp8_moe.envs."
                            "SGLANG_TPU_FP8_MOE_SKIP_PAD_EXPERT.get",
                            return_value=enabled,
                        ),
                    ):
                        apply_tpu_fp8_moe(
                            layer=layer,
                            dispatch_output=SimpleNamespace(
                                hidden_states=x, topk_output=(weights, ids, None)
                            ),
                            activation="silu",
                            block_quant=False,
                            apply_router_weight_on_input=False,
                            no_combine=no_combine,
                        )
                    skipping = enabled and ep_size > 1
                    self.assertEqual(captured["skip_pad_experts"], skipping)
                    self.assertEqual(captured["no_combine"], no_combine)
                    self.assertEqual(captured["expert_group_offset"], ep_rank * 16)
                    torch.testing.assert_close(captured["topk_ids"][0], ids[0])
                    torch.testing.assert_close(captured["topk_weights"][0], weights[0])
                    self.assertTrue(torch.all(captured["topk_weights"][1] == 0))
                    self.assertTrue(
                        torch.all(captured["topk_ids"][1] == (256 if skipping else 0))
                    )
                    torch.testing.assert_close(ids, ids_before)
                    torch.testing.assert_close(weights, weights_before)


if __name__ == "__main__":
    unittest.main()
