"""Unit tests for TPU unquantized MoE routing helpers."""

import unittest

import torch

from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnquantTpuMoe(CustomTestCase):
    def test_translates_global_ids_and_masks_non_local_slots(self):
        # 8 experts over EP size 4 -> 2 local experts per rank; rank 1 owns
        # global ids 2 and 3.
        topk_weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32)
        topk_ids = torch.tensor([[3, 6], [0, 2]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=2, moe_ep_rank=1
        )

        torch.testing.assert_close(
            masked_weights,
            torch.tensor([[0.25, 0.0], [0.0, 0.40]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            masked_ids,
            torch.tensor([[1, -1], [-1, 0]], dtype=torch.int32),
        )
        # Inputs are not mutated.
        torch.testing.assert_close(
            topk_weights,
            torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            topk_ids,
            torch.tensor([[3, 6], [0, 2]], dtype=torch.int32),
        )

    def test_rank0_full_local_range_is_identity(self):
        # EP rank 0 with all ids inside the local range: ids pass through and
        # no weight is zeroed.
        topk_weights = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
        topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=4, moe_ep_rank=0
        )

        torch.testing.assert_close(masked_weights, topk_weights)
        torch.testing.assert_close(masked_ids, topk_ids)

    def test_last_rank_boundary(self):
        # Ids one below and one at the start of the last rank's range.
        topk_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        topk_ids = torch.tensor([[5, 6]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=2, moe_ep_rank=3
        )

        torch.testing.assert_close(
            masked_weights, torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        )
        torch.testing.assert_close(
            masked_ids, torch.tensor([[-1, 0]], dtype=torch.int32)
        )

    def test_padded_slots_stay_dropped(self):
        # TopK marks padded tokens -1; they stay -1 with zero weight on rank 0.
        topk_weights = torch.tensor([[0.0, 0.0], [0.3, 0.7]], dtype=torch.float32)
        topk_ids = torch.tensor([[-1, -1], [1, 0]], dtype=torch.int32)

        masked_weights, masked_ids = mask_tpu_moe_non_local_experts(
            topk_weights, topk_ids, num_local_experts=2, moe_ep_rank=0
        )

        torch.testing.assert_close(masked_weights, topk_weights)
        torch.testing.assert_close(
            masked_ids, torch.tensor([[-1, -1], [1, 0]], dtype=torch.int32)
        )

    def test_resolve_tpu_moe_ep_routing_shifts_ids_when_rank_is_traced(self):
        from unittest.mock import patch

        from sglang.srt.layers.moe.tpu_moe_utils import resolve_tpu_moe_ep_routing

        topk_weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32)
        topk_ids = torch.tensor([[3, 6], [0, 2]], dtype=torch.int32)

        # Outside rank-lifted direct_compile: global IDs and offset pass through.
        with patch(
            "sglang.srt.layers.moe.tpu_moe_utils.current_platform.get_traced_rank",
            return_value=None,
        ):
            w, ids, global_e, offset = resolve_tpu_moe_ep_routing(
                topk_weights,
                topk_ids,
                num_local_experts=2,
                moe_ep_size=4,
                moe_ep_rank=1,
            )
        torch.testing.assert_close(w, topk_weights)
        torch.testing.assert_close(ids, topk_ids)
        self.assertEqual((global_e, offset), (8, 2))

        # Inside rank-lifted direct_compile: IDs shift by traced rank tensor and offset=0.
        with patch(
            "sglang.srt.layers.moe.tpu_moe_utils.current_platform.get_traced_rank",
            return_value=torch.tensor([1], dtype=torch.int32),
        ):
            w, ids, global_e, offset = resolve_tpu_moe_ep_routing(
                topk_weights,
                topk_ids,
                num_local_experts=2,
                moe_ep_size=4,
                moe_ep_rank=1,
                default_global_num_experts=2,
            )
        torch.testing.assert_close(
            w, torch.tensor([[0.25, 0.0], [0.0, 0.40]], dtype=torch.float32)
        )
        torch.testing.assert_close(
            ids, torch.tensor([[1, -1], [-1, 0]], dtype=torch.int32)
        )
        self.assertEqual((global_e, offset), (2, 0))

    def test_quantized_tpu_moe_backends_forward_situ_params(self):
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import Mock, patch

        bridge_name = "sglang_tpu.moe.pallas.fused_moe_op"
        bridge = ModuleType(bridge_name)
        fake_call = bridge.fused_moe_call = Mock(
            side_effect=lambda **kw: torch.zeros_like(kw["hidden_states"])
        )
        layer = SimpleNamespace(
            num_local_experts=2,
            moe_ep_rank=0,
            moe_ep_size=1,
            num_fused_shared_experts=0,
            weight_block_size=None,
            w13_weight=torch.zeros(2, 4, 8),
            w2_weight=torch.zeros(2, 4, 4),
            w13_weight_packed=torch.zeros(2, 1, 8, dtype=torch.int32),
            w2_weight_packed=torch.zeros(2, 1, 4, dtype=torch.int32),
            w13_weight_scale=torch.ones(2, 1, 1, 8),
            w2_weight_scale=torch.ones(2, 1, 1, 4),
        )
        dispatch = SimpleNamespace(
            hidden_states=torch.zeros(3, 4, dtype=torch.bfloat16),
            topk_output=(
                torch.ones(3, 2, dtype=torch.float32),
                torch.zeros(3, 2, dtype=torch.int32),
                None,
            ),
        )
        parallel = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
        with patch.dict(sys.modules, {bridge_name: bridge}):
            from sglang_tpu.fp4_moe import apply_tpu_nvfp4_moe
            from sglang_tpu.fp8_moe import apply_tpu_fp8_moe
            from sglang_tpu.w4a16_moe import apply_tpu_w4a16_moe

            with (
                patch("sglang_tpu.fp4_moe.fused_moe_call", fake_call),
                patch("sglang_tpu.fp8_moe.get_parallel", return_value=parallel),
                patch("sglang_tpu.w4a16_moe.get_parallel", return_value=parallel),
            ):
                for fn, extra in (
                    (apply_tpu_fp8_moe, {"block_quant": False}),
                    (apply_tpu_nvfp4_moe, {"use_fused_rs": False}),
                    (apply_tpu_w4a16_moe, {}),
                ):
                    fake_call.reset_mock()
                    fn(
                        layer=layer,
                        dispatch_output=dispatch,
                        activation="situ",
                        apply_router_weight_on_input=False,
                        gemm1_alpha=4.0,
                        gemm1_clamp_limit=25.0,
                        **extra,
                    )
                    self.assertEqual(fake_call.call_args.kwargs["activation"], "situ")
                    self.assertEqual(fake_call.call_args.kwargs["gemm1_alpha"], 4.0)
                    self.assertEqual(
                        fake_call.call_args.kwargs["gemm1_clamp_limit"], 25.0
                    )


if __name__ == "__main__":
    unittest.main()
