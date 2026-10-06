"""Kimi-K3 latent MoE tail on TPU: KN weights, sharded up-proj tail and merged
GEMVs reproduce the reference output (routed experts stubbed, TP1)."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.environ import envs
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM, KimiK3MoE
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")


def _config() -> KimiLinearConfig:
    return KimiLinearConfig(
        vocab_size=1024,
        hidden_size=1024,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        moe_intermediate_size=512,
        num_experts_per_token=2,
        num_experts=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        routed_expert_hidden_size=512,
        latent_moe_use_norm=True,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "short_conv_kernel_size": 4,
            "head_dim": 64,
            "num_heads": 8,
            "num_key_value_heads": 8,
        },
    )


def _check(out, ref, tol):
    """One bf16 ulp for KN/shard (operand order only); report bitwise status."""
    out = out.cpu()
    print("max abs diff", (out.float() - ref.float()).abs().max().item())
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


class _Experts(nn.Module):
    def forward(self, x, topk_output):
        return torch.tanh(x.float()).to(x.dtype)


class _Null(nn.Module):
    def forward(self, *args):
        return None


@unittest.skipUnless(
    hasattr(torch, "accelerator") and torch.accelerator.is_available(), "requires TPU"
)
class TestKimiK3LatentGemvTpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the PJRT client first: once torch.distributed is up, torch_tpu
        # expects a torchrun-style multi-host environment.
        torch.zeros(1, device="tpu")
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)

    @torch.no_grad()
    def test_tail_variants_match_reference(self):
        torch.manual_seed(0)
        with patch.object(kimi_k3, "_is_tpu", True):
            model = KimiK3LinearForCausalLM(config=_config(), quant_config=None)
        moe = next(l.mlp for l in model.model.layers if isinstance(l.mlp, KimiK3MoE))
        for p in moe.parameters():
            p.normal_(std=0.05)
        moe.experts, moe.gate, moe.topk = _Experts(), _Null(), _Null()
        moe = moe.to("tpu", torch.bfloat16)
        x = torch.randn(8, 1024, dtype=torch.bfloat16).to("tpu")
        ref = moe(x).cpu()

        with envs.SGLANG_TPU_ENABLE_LINEAR_KN_WEIGHT.override(True):
            for m in moe.modules():
                if getattr(m, "allow_kn_weight", False):
                    m.quant_method.process_weights_after_loading(m)
        self.assertIsNotNone(moe.routed_expert_up_proj.weight_kn)
        _check(moe(x), ref, 8e-3)

        moe._shard_latent_up = True
        moe._latent_up_start, moe._latent_up_shard = 0, moe.moe_hidden_size
        _check(moe(x), ref, 8e-3)

        with envs.SGLANG_TPU_ENABLE_K3_MERGED_GEMV.override(True):
            moe._bind_tpu_merged_gemvs()
        self.assertIsNotNone(moe._tpu_front_kn)
        self.assertIsNone(moe.routed_expert_up_proj.weight_kn)
        # The merged tail rounds [latent | shared] once instead of twice.
        _check(moe(x), ref, 2e-2)


if __name__ == "__main__":
    unittest.main()
