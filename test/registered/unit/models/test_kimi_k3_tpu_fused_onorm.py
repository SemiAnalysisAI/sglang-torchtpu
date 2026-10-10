"""Kimi-K3 TPU Pallas KDA o_norm (SGLANG_TPU_ENABLE_K3_FUSED_ONORM): post-load
gating and forward dispatch on real KimiK3DeltaAttention modules, with the
kernel replaced by its contract written in torch.

Kernel numerics: test/registered/tpu/unit/test_kimi_k3_gated_rms_norm.py.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.kernels.ops.attention.fla import fused_norm_gate
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention, KimiK3LinearForCausalLM
from sglang.srt.runtime_context import get_context
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

_KERNEL_MODULE = "sglang_tpu.gated_rms_norm"
_HIDDEN, _EPS = 256, 1e-5


def _config(num_heads=2, head_dim=128, use_full_rank_gate=True):
    return KimiLinearConfig(
        vocab_size=1024,
        hidden_size=_HIDDEN,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=2048,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        moe_intermediate_size=128,
        num_experts_per_token=2,
        num_experts=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=True,
        mla_use_output_gate=True,
        attn_res_block_size=2,
        linear_attn_config={
            # 1-based: layer 0 is KDA, layer 1 MLA.
            "kda_layers": [1],
            "full_attn_layers": [2],
            "short_conv_kernel_size": 4,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "num_key_value_heads": num_heads,
            "q_lora_rank": None,
            "use_full_rank_gate": use_full_rank_gate,
        },
    )


def _kernel_contract(x, g, weight, eps, g_block=0):
    """The Pallas op in torch: x [T, H, D], gate = column block g_block of g,
    same op order as FusedRMSNormGated.forward_native."""
    head_dim = weight.shape[0]
    x_f = x.float()
    width = x_f.shape[1] * head_dim
    gate = g[:, g_block * width : (g_block + 1) * width].float()
    y = x_f * torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    y = y * weight.float()
    y = y * torch.sigmoid(gate.unflatten(-1, (-1, head_dim)))
    return y.to(x.dtype).flatten(-2)


class _StubQuant:
    """Any quant config (K3 checkpoints carry one for the MoE): it turns off
    the fused low-rank projection, so a low-rank layer has no split_sizes."""

    def get_quant_method(self, layer, prefix=""):
        return UnquantizedLinearMethod()


class _FakeKDA(nn.Module):
    """Stands in for RadixLinearAttention: returns a fixed [1, T, H, D] output."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, forward_batch, mixed_qkv, a, b):
        return self.out


class _PassThrough(nn.Module):
    def forward(self, x):
        return x, None


@unittest.skipIf(torch.cuda.is_available(), "builds on CPU with the TPU flag set")
class TestK3TpuFusedONorm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)
        for target, name, value in (
            (kimi_k3, "_is_tpu", True),
            # TPU runs FusedRMSNormGated.forward_native: the flag-off reference.
            (fused_norm_gate, "_is_tpu", True),
            (fused_norm_gate, "_use_cpu", False),
        ):
            flag = patch.object(target, name, value)
            flag.start()
            cls.addClassCleanup(flag.stop)

    def setUp(self):
        self.kernel = types.ModuleType(_KERNEL_MODULE)
        self.kernel.get_gated_rms_norm_op = MagicMock()
        self.kernel.gated_rms_norm_tpu = MagicMock(side_effect=_kernel_contract)
        stub = patch.dict(sys.modules, {_KERNEL_MODULE: self.kernel})
        stub.start()
        self.addCleanup(stub.stop)

    def _layer(self, quant_config=None, **config):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = KimiK3DeltaAttention(
                layer_idx=0,
                hidden_size=_HIDDEN,
                config=_config(**config),
                quant_config=quant_config,
                rms_norm_eps=_EPS,
            )
        finally:
            torch.set_default_dtype(default_dtype)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for param in layer.parameters():
                param.copy_(0.05 * torch.randn(param.shape, generator=gen))
            layer.o_norm.weight.add_(1.0)
        return layer

    def _prepare(self, layer, flag):
        with envs.SGLANG_TPU_ENABLE_K3_FUSED_ONORM.override(flag):
            layer._prepare_tpu_onorm()
        return layer._tpu_onorm_g_block

    def test_flag_off_leaves_it_unbound(self):
        self.assertIsNone(self._prepare(self._layer(), False))
        self.kernel.get_gated_rms_norm_op.assert_not_called()

    def test_flag_on_registers_op_and_finds_g_block(self):
        layer = self._layer()
        self.assertEqual(self._prepare(layer, True), 3)
        self.kernel.get_gated_rms_norm_op.assert_called_once_with()

    def test_unsupported_layers_keep_the_xla_chain(self):
        # Kimi-K3-0.40B has head_dim 32; the kernel needs 128-lane heads.
        self.assertIsNone(self._prepare(self._layer(num_heads=8, head_dim=32), True))
        # Without the full-rank gate there is no fused [q, k, v, g] output.
        self.assertIsNone(self._prepare(self._layer(use_full_rank_gate=False), True))
        # The kernel implements the sigmoid gate only.
        layer = self._layer()
        layer.o_norm.activation = "silu"
        self.assertIsNone(self._prepare(layer, True))
        self.kernel.get_gated_rms_norm_op.assert_not_called()

    def test_quantized_low_rank_layer_without_a_split(self):
        layer = self._layer(quant_config=_StubQuant(), use_full_rank_gate=False)
        self.assertFalse(hasattr(layer, "split_sizes"))
        for flag in (False, True):
            with self.subTest(flag=flag):
                self.assertIsNone(self._prepare(layer, flag))
        self.kernel.get_gated_rms_norm_op.assert_not_called()

    def test_flag_is_ignored_off_tpu(self):
        model = KimiK3LinearForCausalLM(config=_config(), quant_config=None)
        with (
            patch.object(kimi_k3, "_is_tpu", False),
            envs.SGLANG_TPU_ENABLE_K3_FUSED_ONORM.override(True),
        ):
            model.post_load_weights()
        for layer in model.model.layers:
            if isinstance(layer.self_attn, KimiK3DeltaAttention):
                self.assertIsNone(layer.self_attn._tpu_onorm_g_block)
        self.kernel.get_gated_rms_norm_op.assert_not_called()

    def test_post_load_prepares_every_kda_layer(self):
        model = KimiK3LinearForCausalLM(config=_config(), quant_config=None)
        kdas = [
            layer.self_attn
            for layer in model.model.layers
            if isinstance(layer.self_attn, KimiK3DeltaAttention)
        ]
        self.assertTrue(kdas)
        for flag, expected in ((False, None), (True, 3)):
            with self.subTest(flag=flag):
                with envs.SGLANG_TPU_ENABLE_K3_FUSED_ONORM.override(flag):
                    model.post_load_weights()
                for kda in kdas:
                    self.assertEqual(kda._tpu_onorm_g_block, expected)

    def _forward(self, layer, rows, mode):
        gen = torch.Generator().manual_seed(rows)
        hidden = torch.randn(rows, _HIDDEN, generator=gen).bfloat16()
        core = torch.randn(1, rows, 2, 128, generator=gen).bfloat16()
        layer.attn = _FakeKDA(core)
        layer.o_proj = _PassThrough()
        batch = SimpleNamespace(forward_mode=mode)
        with torch.no_grad():
            out = layer(hidden, None, batch, None)
            qkvg = layer.fused_qkvg_proj(hidden)[0]
        return out, qkvg

    def test_forward_matches_xla_chain(self):
        """Batches up to 64 tokens take the kernel in decode and extend, with g
        read from the whole [q, k, v, g] output and the live norm weight; 65
        tokens keep the XLA chain. Outputs equal the flag-off path."""
        reference = self._layer()
        fused = self._layer()
        self._prepare(fused, True)
        cases = [(ForwardMode.DECODE, rows) for rows in (1, 4, 64, 65)]
        cases += [(ForwardMode.EXTEND, rows) for rows in (1, 64, 65)]
        for mode, rows in cases:
            with self.subTest(mode=mode.name, rows=rows):
                self.kernel.gated_rms_norm_tpu.reset_mock()
                ref, _ = self._forward(reference, rows, mode)
                out, qkvg = self._forward(fused, rows, mode)
                self.assertEqual(tuple(out.shape), (rows, 2 * 128))
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
                if rows > 64:
                    self.kernel.gated_rms_norm_tpu.assert_not_called()
                    continue
                self.kernel.gated_rms_norm_tpu.assert_called_once()
                (x, g, weight, eps), kwargs = self.kernel.gated_rms_norm_tpu.call_args
                self.assertEqual(tuple(x.shape), (rows, 2, 128))
                self.assertTrue(torch.equal(g, qkvg))
                self.assertIs(weight, fused.o_norm.weight)
                self.assertEqual((eps, kwargs), (_EPS, {"g_block": 3}))


if __name__ == "__main__":
    unittest.main()
