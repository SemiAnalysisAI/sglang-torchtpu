"""Kimi-K3 construction and post-load on TPU."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear, RowParallelLinear
from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
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

    @torch.no_grad()
    def test_kn_weight_opt_in_and_post_load_order(self):
        """Decoder linears opt in except q_b_proj; post-load run after the
        KN bind (sharded-state loader order) derives the same MLA weights."""
        torch.manual_seed(0)
        ref = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        for p in ref.parameters():
            p.normal_()
        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        model.load_state_dict(ref.state_dict())
        linears = {
            name: m
            for name, m in model.model.named_modules()
            if isinstance(getattr(m, "quant_method", None), UnquantizedLinearMethod)
        }
        self.assertTrue(linears)
        for name, m in linears.items():
            self.assertEqual(m.allow_kn_weight, not name.endswith("q_b_proj"), name)
        self.assertFalse(hasattr(model.lm_head, "allow_kn_weight"))

        with (
            patch.object(unquant, "_is_tpu", True),
            patch.object(unquant, "_is_cpu", False),
            envs.SGLANG_TPU_ENABLE_LINEAR_KN_WEIGHT.override(True),
        ):
            for m in linears.values():
                m.quant_method.process_weights_after_loading(m)
        model.post_load_weights()
        ref.post_load_weights()
        bound = 0
        for name, m in linears.items():
            # 3-D conv weights, q_b_proj and sub-lane outputs (the one-row
            # attn-res score projections) stay [out, in]-contiguous.
            expect_kn = (
                m.allow_kn_weight
                and m.weight.dim() == 2
                and m.weight.shape[0] >= unquant._TPU_KN_MIN_OUT_FEATURES
            )
            weight_kn = getattr(m, "weight_kn", None)
            self.assertEqual(weight_kn is not None, expect_kn, name)
            if expect_kn:
                # A view would be re-materialized on every TPU graph call.
                self.assertIsNone(weight_kn._base, name)
                bound += 1
        self.assertGreater(bound, 0)
        self.assertIsNone(
            getattr(model.model.layers[0].self_attention_res_proj, "weight_kn", None)
        )
        mla = model.model.layers[3].self_attn
        ref_mla = ref.model.layers[3].self_attn
        torch.testing.assert_close(mla.w_kc, ref_mla.w_kc, rtol=0, atol=0)
        torch.testing.assert_close(mla.w_vc, ref_mla.w_vc, rtol=0, atol=0)

    def test_latent_up_and_merged_gemv_gated_off(self):
        """TP1 (or flags off) keeps the replicated up-proj and unmerged GEMVs."""
        for flags in (False, True):
            with (
                envs.SGLANG_TPU_ENABLE_K3_SHARD_LATENT_UP.override(flags),
                envs.SGLANG_TPU_ENABLE_K3_MERGED_GEMV.override(flags),
            ):
                model = KimiK3LinearForCausalLM(
                    config=_tiny_config(), quant_config=None
                )
                model.post_load_weights()
            for moe in _moes(model):
                self.assertFalse(moe._shard_latent_up)
                self.assertIsInstance(moe.routed_expert_up_proj, ReplicatedLinear)
                self.assertIsNone(moe._tpu_front_kn)

    def test_latent_up_shard_matches_row_parallel_loader(self):
        """The static latent slice is the column block RowParallelLinear loads."""
        w = torch.randn(32, 16)
        for rank in range(4):
            proj = RowParallelLinear(
                16, 32, bias=False, reduce_results=False, tp_rank=rank, tp_size=4
            )
            proj.weight_loader(proj.weight, w)
            shard = 16 // 4
            torch.testing.assert_close(
                proj.weight.data, w[:, rank * shard : (rank + 1) * shard]
            )

    @torch.no_grad()
    def test_shard_and_merged_tail_match_reference(self):
        """At TP1 the sharded tail and the merged GEMVs reproduce the
        replicated up-proj + separate shared-expert output."""
        torch.manual_seed(0)
        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        moe = _moes(model)[0].to(torch.bfloat16)
        for p in moe.parameters():
            p.normal_(std=0.05)
        _stub_routing(moe)
        x = torch.randn(4, 512, dtype=torch.bfloat16)
        ref = moe(x)
        moe._shard_latent_up = True
        moe._latent_up_start, moe._latent_up_shard = 0, moe.moe_hidden_size
        torch.testing.assert_close(moe(x), ref, rtol=0, atol=0)
        with envs.SGLANG_TPU_ENABLE_K3_MERGED_GEMV.override(True):
            moe._bind_tpu_merged_gemvs()
        self.assertIsNotNone(moe._tpu_front_kn)
        self.assertEqual(
            moe.routed_expert_down_proj.weight.data_ptr(), moe._tpu_front_kn.data_ptr()
        )
        # Tail accumulates [latent | shared] in one GEMV: one bf16 rounding.
        torch.testing.assert_close(moe(x), ref, rtol=2e-2, atol=2e-2)


class _Experts(nn.Module):
    def forward(self, x, topk_output):
        return torch.tanh(x.float()).to(x.dtype)


class _Null(nn.Module):
    def forward(self, *args):
        return None


def _moes(model):
    return [l.mlp for l in model.model.layers if isinstance(l.mlp, KimiK3MoE)]


def _stub_routing(moe):
    """Routed experts as a pointwise map: the test covers the latent glue."""
    moe.experts, moe.gate, moe.topk = _Experts(), _Null(), _Null()


if __name__ == "__main__":
    unittest.main()
