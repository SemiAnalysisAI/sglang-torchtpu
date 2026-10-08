"""Kimi-K3 attention-residual TPU decode kernel: init registration and post-load caches."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.environ import envs
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_KERNEL_MODULE = "sglang.srt.hardware_backend.tpu.kimi_attn_residual"


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


def _attn_res_points(model):
    """(score_proj, score_norm, out_norm) of every AttnResidual.forward call site."""
    points = []
    for layer in model.model.layers:
        if isinstance(layer, PPMissingLayer) or not layer.use_attn_residuals:
            continue
        points.append(
            (
                layer.self_attention_res_proj,
                layer.self_attention_res_norm,
                layer.input_layernorm,
            )
        )
        points.append(
            (layer.mlp_res_proj, layer.mlp_res_norm, layer.post_attention_layernorm)
        )
    points.append(
        (
            model.model.output_attn_res_proj,
            model.model.output_attn_res_norm,
            model.model.norm,
        )
    )
    return points


@unittest.skipIf(torch.cuda.is_available(), "builds on CPU with the TPU flag set")
class TestKimiK3AttnResTpuKernelSetup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)
        tpu_flag = patch.object(kimi_k3, "_is_tpu", True)
        tpu_flag.start()
        cls.addClassCleanup(tpu_flag.stop)

    def _build(self, flag: bool):
        """Construct and post-load with a stand-in kernel module (the real one
        needs JAX)."""
        kernel = types.ModuleType(_KERNEL_MODULE)
        kernel.register_kimi_attn_res_decode_ops = MagicMock()
        kernel.get_norm_weight_f32 = MagicMock()
        with (
            patch.dict(sys.modules, {_KERNEL_MODULE: kernel}),
            envs.SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL.override(flag),
        ):
            model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
            model.post_load_weights()
        return model, kernel

    def test_model_init_registers_decode_ops(self):
        """Both decode ops exist before the dynamo trace when the kernel is on."""
        for flag in (True, False):
            with self.subTest(flag=flag):
                _, kernel = self._build(flag=flag)
                register = kernel.register_kimi_attn_res_decode_ops
                if flag:
                    register.assert_called_once_with()
                else:
                    register.assert_not_called()

    def test_post_load_builds_the_kernel_weights(self):
        """post_load_weights builds both cw dtypes at every aggregation point and,
        with the kernel on, that point's fp32 out-norm weight, all before the
        trace."""
        for flag in (True, False):
            with self.subTest(flag=flag):
                model, kernel = self._build(flag=flag)
                points = _attn_res_points(model)
                self.assertEqual(len(points), 2 * 4 + 1)
                for proj, _, _ in points:
                    cache = proj._attn_res_cw_cache
                    self.assertEqual(set(cache), {torch.float32, torch.bfloat16})
                # Modules compare by identity: each point's own out norm.
                built = [c.args for c in kernel.get_norm_weight_f32.call_args_list]
                expected = [(out_norm,) for _, _, out_norm in points] if flag else []
                self.assertEqual(built, expected)
                # The copies are load-time only: online updates are rejected.
                reason = _unsupported_derived_weight_cache_error(model)
                if flag:
                    self.assertIn("SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL", reason)
                else:
                    self.assertIsNone(reason)

    def test_flag_is_ignored_off_tpu(self):
        # Off TPU the model also builds CUDA alt streams, which a CPU host lacks.
        with (
            patch.object(kimi_k3, "_is_tpu", False),
            patch("torch.cuda.Stream", MagicMock),
        ):
            model, kernel = self._build(flag=True)
        kernel.register_kimi_attn_res_decode_ops.assert_not_called()
        kernel.get_norm_weight_f32.assert_not_called()
        self.assertIsNone(_unsupported_derived_weight_cache_error(model))


if __name__ == "__main__":
    unittest.main()
