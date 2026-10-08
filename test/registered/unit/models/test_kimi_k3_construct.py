"""Kimi-K3 construction and post-load on TPU."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _tiny_config() -> KimiLinearConfig:
    return KimiLinearConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=2048,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        moe_intermediate_size=256,
        num_experts_per_token=2,
        num_experts=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        routed_expert_hidden_size=256,
        latent_moe_use_norm=True,
        mla_use_output_gate=True,
        attn_res_block_size=2,
        linear_attn_config={
            "kda_layers": [0, 1, 2],
            "full_attn_layers": [3],
            "short_conv_kernel_size": 4,
            "head_dim": 64,
            "num_heads": 8,
            "num_key_value_heads": 8,
            "q_lora_rank": None,
            "use_full_rank_gate": True,
        },
    )


@unittest.skipIf(torch.cuda.is_available(), "builds on CPU with the TPU flag set")
class TestKimiK3ConstructOnTpu(CustomTestCase):
    def test_post_load_binds_kernel_layout_and_rejects_weight_updates(self):
        """The KDA kernel reads a [K, 1, dim] copy of the conv weight, so online
        updates, which could skip the post-load rebind, are rejected."""
        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        kdas = [
            layer.self_attn
            for layer in model.model.layers
            if isinstance(layer.self_attn, KimiK3DeltaAttention)
        ]
        with torch.no_grad():
            for kda in kdas:
                weight = kda.qkv_conv1d.weight  # [dim, 1, K]
                weight.copy_(torch.randn(weight.shape))
        model.post_load_weights()
        for kda in kdas:
            want = kda.qkv_conv1d.weight.permute(2, 1, 0)
            self.assertTrue(kda.attn.conv_weights.is_contiguous())
            self.assertTrue(torch.equal(kda.attn.conv_weights, want))
        self.assertIn(
            "Kimi-K3 on TPU", _unsupported_derived_weight_cache_error(model) or ""
        )

    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)
        tpu_flag = patch.object(kimi_k3, "_is_tpu", True)
        tpu_flag.start()
        cls.addClassCleanup(tpu_flag.stop)

    def test_model_builds_without_side_streams(self):
        """No torch.cuda.Stream is created on TPU."""
        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        self.assertIsNone(model.model.alt_streams)

    def test_post_load_keeps_unfused_weights(self):
        """CUDA-only merged layouts stay unset on TPU."""
        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        model.post_load_weights()
        layers = model.model.layers
        moes = [l.mlp for l in layers if isinstance(l.mlp, KimiK3MoE)]
        kdas = [
            l.self_attn for l in layers if isinstance(l.self_attn, KimiK3DeltaAttention)
        ]
        self.assertTrue(moes and kdas)
        for moe in moes:
            self.assertIsNone(moe._front_w)
        for kda in kdas:
            self.assertIsNone(kda._bfa_w)
            self.assertFalse(kda._kda_fused_decode_ready)


if __name__ == "__main__":
    unittest.main()
