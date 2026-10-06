"""CPU coverage of GLM direct-tracer state discovery and opt-in registration."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.compilation.support_torch_compile import get_torch_compile_policy
from sglang.srt.hardware_backend.tpu.compilation.glm_moe_dsa_compile_policy import (
    GlmMoeDsaTpuCompilePolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer(*, moe=False):
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.w_kc = torch.ones(2, 3, 4)
    layer.self_attn.w_vc = torch.ones(2, 4, 3)
    layer.self_attn.w_scale = 1.0
    layer.self_attn.w_scale_k = None
    layer.self_attn.w_scale_v = None
    layer.mlp = torch.nn.Module()
    if moe:
        layer.mlp.gate = torch.nn.Module()
        layer.mlp.gate.e_score_correction_bias = torch.nn.Parameter(torch.ones(8))
        layer.mlp.topk = torch.nn.Module()
        layer.mlp.topk.topk_config = SimpleNamespace(
            correction_bias=layer.mlp.gate.e_score_correction_bias
        )
    return layer


def _model(layers):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(layers)
    return model


def _policy(model):
    return GlmMoeDsaTpuCompilePolicy(
        model_runner=SimpleNamespace(model=model), owner=model, tracer="direct"
    )


def _resolve(model, path):
    for name in path.split("."):
        model = getattr(model, name)
    return model


class TestGlmMoeDsaTpuCompilePolicy(unittest.TestCase):
    def test_dense_and_moe_state_includes_helper_parameter_alias(self):
        model = _model([_layer(), _layer(moe=True)])
        paths = _policy(model).get_model_extra_tensor_attrs()
        self.assertEqual(
            set(paths),
            {
                "model.layers.0.self_attn.w_kc",
                "model.layers.0.self_attn.w_vc",
                "model.layers.1.self_attn.w_kc",
                "model.layers.1.self_attn.w_vc",
                "model.layers.1.mlp.topk.topk_config.correction_bias",
            },
        )
        alias_path = "model.layers.1.mlp.topk.topk_config.correction_bias"
        self.assertIs(
            _resolve(model, alias_path),
            dict(model.named_parameters())[
                "model.layers.1.mlp.gate.e_score_correction_bias"
            ],
        )
        self.assertTrue(all(isinstance(_resolve(model, p), torch.Tensor) for p in paths))
        self.assertEqual(len(paths), len(set(paths)))

    def test_pp_placeholders_preserve_global_layer_paths(self):
        model = _model(
            [
                torch.nn.Identity(),
                torch.nn.Identity(),
                _layer(moe=True),
                _layer(),
                torch.nn.Identity(),
            ]
        )
        model.model.start_layer = 2
        model.model.end_layer = 4
        paths = _policy(model).get_model_extra_tensor_attrs()
        self.assertEqual({p.split(".")[2] for p in paths}, {"2", "3"})
        self.assertIn("model.layers.2.mlp.topk.topk_config.correction_bias", paths)
        self.assertEqual(len(paths), 5)

    def test_registered_absorb_tensor_is_not_added_twice(self):
        layer = _layer()
        value = layer.self_attn.w_kc
        del layer.self_attn.w_kc
        layer.self_attn.register_buffer("w_kc", value, persistent=False)
        model = _model([layer])
        self.assertEqual(
            _policy(model).get_model_extra_tensor_attrs(),
            ("model.layers.0.self_attn.w_vc",),
        )

    def test_state_is_discovered_after_loading_and_supports_optional_scales(self):
        layer = _layer()
        layer.self_attn.w_kc = None
        model = _model([layer])
        policy = _policy(model)
        self.assertNotIn(
            "model.layers.0.self_attn.w_kc", policy.get_model_extra_tensor_attrs()
        )
        layer.self_attn.w_kc = torch.ones(2, 3, 4)
        layer.self_attn.w_scale_k = torch.ones(2, 1)
        layer.self_attn.w_kc_qrep = torch.ones(4, 3, 4)
        paths = policy.get_model_extra_tensor_attrs()
        self.assertIn("model.layers.0.self_attn.w_kc", paths)
        self.assertIn("model.layers.0.self_attn.w_scale_k", paths)
        self.assertIn("model.layers.0.self_attn.w_kc_qrep", paths)
        self.assertNotIn("model.layers.0.self_attn.w_scale", paths)
        self.assertNotIn("model.layers.0.self_attn.w_scale_v", paths)


class TestGlmDirectCompileOptIn(unittest.TestCase):
    def _construct(self, *, tpu, tracer):
        # Execute the actual GLM wrapper constructor while replacing only its
        # heavyweight DeepSeek parent. This verifies the production opt-in
        # condition without allocating model weights or initializing devices.
        path = Path(__file__).resolve().parents[5] / "python/sglang/srt/models/glm4_moe.py"
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "GlmMoeDsaForCausalLM"
        )

        class Parent(torch.nn.Module):
            def __init__(self, config, quant_config=None, prefix=""):
                super().__init__()

        namespace = {
            "DeepseekV2ForCausalLM": Parent,
            "get_server_args": lambda: SimpleNamespace(tpu_compile_tracer=tracer),
            "is_tpu": lambda: tpu,
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["GlmMoeDsaForCausalLM"](SimpleNamespace())

    def test_only_direct_tpu_instance_is_annotated(self):
        model = self._construct(tpu=True, tracer="direct")
        self.assertIs(get_torch_compile_policy(model), GlmMoeDsaTpuCompilePolicy)

    def test_dynamo_keeps_existing_fallback(self):
        self.assertIsNone(
            get_torch_compile_policy(self._construct(tpu=True, tracer="dynamo"))
        )

    def test_non_tpu_has_no_tpu_annotation(self):
        self.assertIsNone(
            get_torch_compile_policy(self._construct(tpu=False, tracer="direct"))
        )


if __name__ == "__main__":
    unittest.main()
