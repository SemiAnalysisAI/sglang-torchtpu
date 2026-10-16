"""Unit tests for the TPU MXFP4 MoE weight preparation (CPU only)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types
import unittest

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.mxfp4_tpu_util import (
    _EXPERTS_PER_CHUNK,
    prepare_tpu_mxfp4_moe_weights,
    process_mxfp4_moe_weights_bf16,
)
from sglang.test.test_utils import CustomTestCase

HIDDEN, INTER = 64, 96


def _fake_layer(
    num_experts: int,
    seed: int = 0,
    activation: str = "swigluoai",
    hidden: int = HIDDEN,
) -> types.SimpleNamespace:
    g = torch.Generator().manual_seed(seed)
    u8 = lambda *shape: torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    # e8m0 scales near 1.0 keep the dequantized values finite.
    e8 = lambda *shape: torch.randint(120, 134, shape, dtype=torch.uint8, generator=g)
    layer = types.SimpleNamespace()
    layer.w13_weight = Parameter(
        u8(num_experts, 2 * INTER, hidden // 2), requires_grad=False
    )
    layer.w13_weight_scale = Parameter(
        e8(num_experts, 2 * INTER, hidden // 32), requires_grad=False
    )
    layer.w13_weight_bias = Parameter(
        torch.randn(num_experts, 2 * INTER, generator=g, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.w2_weight = Parameter(
        u8(num_experts, hidden, INTER // 2), requires_grad=False
    )
    layer.w2_weight_scale = Parameter(
        e8(num_experts, hidden, INTER // 32), requires_grad=False
    )
    layer.w2_weight_bias = Parameter(
        torch.randn(num_experts, hidden, generator=g, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.moe_runner_config = types.SimpleNamespace(activation=activation)
    return layer


class TestChunkedExpertPreparation(CustomTestCase):
    def test_experts_prepared_in_groups_match_the_whole_tensor(self):
        """The experts are dequantized a group at a time; with a group count
        that does not divide the expert count, the assembled weights and
        biases must equal the single-tensor result expert for expert."""
        num_experts = 2 * _EXPERTS_PER_CHUNK + 8
        layer = _fake_layer(num_experts)
        whole = process_mxfp4_moe_weights_bf16(
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
            layer.w13_weight_bias.data,
            layer.w2_weight.data,
            layer.w2_weight_scale.data,
            layer.w2_weight_bias.data,
            w13_interleaved=True,
        )
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        got = (
            layer.w13_weight,
            layer.w13_weight_bias,
            layer.w2_weight,
            layer.w2_weight_bias,
        )
        for name, a, b in zip(("w13", "w13_bias", "w2", "w2_bias"), whole, got):
            self.assertEqual(tuple(a.shape), tuple(b.shape), name)
            self.assertEqual(a.dtype, b.dtype, name)
            self.assertTrue(torch.equal(a, b), name)
        self.assertIsNone(layer.w13_weight_scale)
        self.assertIsNone(layer.w2_weight_scale)


class TestPackedPreparation(CustomTestCase):
    def test_non_interleaved_aligned_keeps_packed_fp4(self):
        """Kimi-K3-style checkpoints keep packed FP4 in gmm_v2's (E, K, N) layout."""
        layer = _fake_layer(num_experts=4, activation="situ", hidden=128)
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=False)
        n_pad = 128  # INTER=96 pads each gate/up half to 128
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(tuple(layer.w13_weight.shape), (4, 128 // 2, 2 * n_pad))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape), (4, 128 // 32, 1, 2 * n_pad)
        )
        self.assertEqual(tuple(layer.w2_weight.shape), (4, n_pad // 2, 128))
        self.assertEqual(tuple(layer.w2_weight_scale.shape), (4, n_pad // 32, 1, 128))
        self.assertIsNone(layer.w13_weight_bias)
        self.assertIsNone(layer.w2_weight_bias)

    def test_misaligned_hidden_falls_back_to_bf16(self):
        """The packed path cannot pad K, so a misaligned hidden size dequantizes."""
        layer = _fake_layer(num_experts=4, activation="situ")
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
        self.assertIsNone(layer.w13_weight_scale)
        self.assertIsNone(layer.w2_weight_scale)

    def test_preserves_parameter_identity_and_frees_replaced_storages(self):
        """Weight conversion keeps Parameter identity/attrs and frees old buffers."""
        for hidden, with_bias in ((128, False), (HIDDEN, True)):
            with self.subTest(hidden=hidden, with_bias=with_bias):
                layer = _fake_layer(num_experts=4, activation="situ", hidden=hidden)
                w13_param = layer.w13_weight
                w2_param = layer.w2_weight
                w13_param.weight_loader = "sentinel_w13"
                w2_param.weight_loader = "sentinel_w2"
                old_storages = [
                    layer.w13_weight.data.untyped_storage(),
                    layer.w13_weight_scale.data.untyped_storage(),
                    layer.w13_weight_bias.data.untyped_storage(),
                    layer.w2_weight.data.untyped_storage(),
                    layer.w2_weight_scale.data.untyped_storage(),
                    layer.w2_weight_bias.data.untyped_storage(),
                ]

                prepare_tpu_mxfp4_moe_weights(layer, with_bias=with_bias)

                self.assertIs(layer.w13_weight, w13_param)
                self.assertIs(layer.w2_weight, w2_param)
                self.assertEqual(layer.w13_weight.weight_loader, "sentinel_w13")
                self.assertEqual(layer.w2_weight.weight_loader, "sentinel_w2")
                for storage in old_storages:
                    self.assertEqual(storage.nbytes(), 0)

    def test_honors_explicit_gate_up_interleaved_with_silu_activation(self):
        """Standard FusedMoE layers with gate_up_interleaved=True deinterleave w13."""
        layer = _fake_layer(num_experts=2, activation="silu", hidden=HIDDEN)
        layer.moe_runner_config.gate_up_interleaved = True
        expected = process_mxfp4_moe_weights_bf16(
            layer.w13_weight.data.clone(),
            layer.w13_weight_scale.data.clone(),
            layer.w13_weight_bias.data.clone(),
            layer.w2_weight.data.clone(),
            layer.w2_weight_scale.data.clone(),
            layer.w2_weight_bias.data.clone(),
            w13_interleaved=True,
        )
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        torch.testing.assert_close(layer.w13_weight.data, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(
            layer.w13_weight_bias.data, expected[1], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
