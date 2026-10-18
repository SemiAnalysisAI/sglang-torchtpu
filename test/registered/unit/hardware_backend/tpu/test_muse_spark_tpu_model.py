# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Muse Spark on TPU: checkpoint config parsing, the mapping
of checkpoint tensors onto the model, routing and the post-expert-norm combine."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import get_hybrid_layer_ids
from sglang.srt.configs.muse_spark import MuseSparkConfig
from sglang.srt.layers.quantization.modelopt_quant import (
    TpuModelOptCompatibleFp4Config,
)
from sglang.srt.layers.rotary_embedding import base as rotary_base
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.hf_transformers.config import HfModelConfigParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.muse_spark import (
    MuseSparkForConditionalGeneration,
    MuseSparkMoE,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_LM = "model.language_model"
_TEXT_CONFIG = {
    "hidden_size": 128,
    "head_dim": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "layer_types": ["sliding_attention", "full_attention"],
    "layer_rope_theta": [10000.0, 0],
    "mlp_layer_types": ["sparse", "sparse"],
    "sliding_window": 16,
    "moe_hidden_size": 64,
    "moe_intermediate_size": 64,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 256,
    "max_position_embeddings": 256,
}
# The released export keeps the first layer's experts in BF16.
_QUANT_CONFIG = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "group_size": 16,
    "modules_to_not_convert": ["lm_head", f"{_LM}.layers.0.mlp.experts"],
}


def _checkpoint() -> dict:
    """Random tensors under the checkpoint's names, shaped like the HF export."""
    c = _TEXT_CONFIG
    e, h, i = c["num_local_experts"], c["moe_hidden_size"], c["moe_intermediate_size"]
    q_width = c["num_attention_heads"] * c["head_dim"]
    kv_width = c["num_key_value_heads"] * c["head_dim"]
    hidden = c["hidden_size"]
    t = {
        f"{_LM}.embed_tokens.weight": torch.randn(c["vocab_size"], hidden),
        f"{_LM}.norm.weight": torch.randn(hidden),
        "lm_head.weight": torch.randn(c["vocab_size"], hidden),
        # Vision and audio towers are not part of the text model.
        "model.vision_tower.layers.0.attn.q_proj.weight": torch.randn(8, 8),
        "model.audio_tower.proj.weight": torch.randn(8, 8),
    }
    for layer in range(c["num_hidden_layers"]):
        p = f"{_LM}.layers.{layer}"
        for name, rows in (
            ("q", q_width),
            ("k", kv_width),
            ("v", kv_width),
            ("gate", q_width),
        ):
            t[f"{p}.self_attn.{name}_proj.weight"] = torch.randn(rows, hidden)
        t[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, q_width)
        for name in (
            "input_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            t[f"{p}.{name}.weight"] = torch.randn(hidden)
        for name in ("post_attention_residual_gate", "post_feedforward_residual_gate"):
            t[f"{p}.{name}.gate"] = torch.randn(hidden)
        t[f"{p}.mlp.gate.weight"] = torch.randn(e, hidden, dtype=torch.float32)
        t[f"{p}.mlp.gate.e_score_correction_bias"] = torch.randn(e)
        t[f"{p}.mlp.pre_expert_proj.weight"] = torch.randn(h, hidden)
        t[f"{p}.mlp.post_expert_proj.weight"] = torch.randn(hidden, h)
        t[f"{p}.mlp.pre_expert_norm.weight"] = torch.randn(h)
        t[f"{p}.mlp.experts.post_expert_norm.weight"] = torch.randn(h)
        x = f"{p}.mlp.experts"
        if layer == 0:
            t[f"{x}.gate_up_proj"] = torch.randn(e, 2 * i, h)
            t[f"{x}.down_proj"] = torch.randn(e, h, i)
            continue
        t[f"{x}.gate_up_proj"] = torch.randint(
            0, 256, (e, 2 * i, h // 2), dtype=torch.uint8
        )
        t[f"{x}.down_proj"] = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)
        t[f"{x}.gate_up_proj_weight_scale"] = torch.rand(e, 2 * i, h // 16).to(
            torch.float8_e4m3fn
        )
        t[f"{x}.down_proj_weight_scale"] = torch.rand(e, h, i // 16).to(
            torch.float8_e4m3fn
        )
        t[f"{x}.gate_up_proj_weight_scale_2"] = torch.rand(e, 2)
        t[f"{x}.down_proj_weight_scale_2"] = torch.rand(e)
        t[f"{x}.gate_up_proj_input_scale"] = torch.rand(e, 2)
        t[f"{x}.down_proj_input_scale"] = torch.rand(e)
    return t


class TestMuseSparkTpuModel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)
        # RoPE construction off TPU looks for a vLLM kernel; build it as on TPU.
        tpu_rope = patch.object(rotary_base, "_is_tpu", True)
        tpu_rope.start()
        cls.addClassCleanup(tpu_rope.stop)

    def _build(self):
        config = MuseSparkConfig(text_config=dict(_TEXT_CONFIG))
        # Set by ModelConfig from --language-model-only.
        config.language_model_only = True
        quant_config = TpuModelOptCompatibleFp4Config.from_config(_QUANT_CONFIG)
        return MuseSparkForConditionalGeneration(config, quant_config=quant_config)

    def test_checkpoint_config_parses_with_layer_schedule_and_exclusions(self):
        """config.json needs the ported config class (the pinned transformers
        predates muse_spark), the hybrid split, and the exclude list read from
        ``modules_to_not_convert``."""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "muse_spark",
                        "architectures": ["MuseSparkForConditionalGeneration"],
                        "text_config": _TEXT_CONFIG,
                        "quantization_config": _QUANT_CONFIG,
                    }
                )
            )
            config = HfModelConfigParser().parse(directory, trust_remote_code=False)
        self.assertEqual(config.text_config.layer_types, _TEXT_CONFIG["layer_types"])
        self.assertEqual(
            get_hybrid_layer_ids(config.architectures, config.text_config),
            ([0], [1]),
        )
        quant_config = TpuModelOptCompatibleFp4Config.from_config(_QUANT_CONFIG)
        self.assertTrue(quant_config.is_layer_excluded(f"{_LM}.layers.0.mlp.experts"))
        self.assertFalse(quant_config.is_layer_excluded(f"{_LM}.layers.1.mlp.experts"))

    def test_load_weights_maps_checkpoint_tensors(self):
        model = self._build()
        checkpoint = _checkpoint()
        model.load_weights(iter(checkpoint.items()))
        params = dict(model.named_parameters())
        i = _TEXT_CONFIG["moe_intermediate_size"]

        attn = f"{_LM}.layers.1.self_attn"
        torch.testing.assert_close(
            params[f"{attn}.qkv_proj.weight"].float(),
            torch.cat([checkpoint[f"{attn}.{n}_proj.weight"] for n in "qkv"])
            .to(params[f"{attn}.qkv_proj.weight"].dtype)
            .float(),
        )
        router = params[f"{_LM}.layers.1.mlp.gate.weight"]
        self.assertEqual(router.dtype, torch.float32)
        torch.testing.assert_close(
            router, checkpoint[f"{_LM}.layers.1.mlp.gate.weight"]
        )

        for layer, dtype in ((0, None), (1, torch.uint8)):
            x = f"{_LM}.layers.{layer}.mlp.experts"
            w13 = params[f"{x}.w13_weight"]
            if dtype is not None:
                self.assertEqual(w13.dtype, dtype)
            # [gate, up] halves of expert 3 land in the same order.
            expected = checkpoint[f"{x}.gate_up_proj"][3].to(w13.dtype)
            self.assertTrue(torch.equal(w13[3][:i], expected[:i]))
            self.assertTrue(torch.equal(w13[3][i:], expected[i:]))
        x = f"{_LM}.layers.1.mlp.experts"
        torch.testing.assert_close(
            params[f"{x}.w13_weight_scale_2"],
            checkpoint[f"{x}.gate_up_proj_weight_scale_2"],
        )
        torch.testing.assert_close(
            params[f"{x}.w2_weight_scale_2"],
            checkpoint[f"{x}.down_proj_weight_scale_2"],
        )
        self.assertTrue(
            torch.equal(
                params[f"{x}.w13_weight_scale"].view(torch.uint8),
                checkpoint[f"{x}.gate_up_proj_weight_scale"].view(torch.uint8),
            )
        )

        del checkpoint[f"{_LM}.layers.1.pre_feedforward_layernorm.weight"]
        with self.assertRaisesRegex(ValueError, "missing 1 tensors"):
            self._build().load_weights(iter(checkpoint.items()))

    def test_router_selects_by_biased_score_and_weights_by_raw_score(self):
        moe = self._build().model.language_model.layers[1].mlp
        router = moe.gate
        with torch.no_grad():
            router.weight.zero_()
            router.weight[:, 0] = torch.tensor([4.0, 2.0, 0.0, -2.0])
            # Expert 3 has the lowest raw score but the bias pulls it into the top 2.
            router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        hidden = torch.zeros(1, _TEXT_CONFIG["hidden_size"])
        hidden[0, 0] = 1.0
        topk = router(hidden)
        self.assertEqual(sorted(topk.topk_ids[0].tolist()), [0, 3])
        raw = torch.sigmoid(torch.tensor([4.0, -2.0]) * router.logit_multiplier)
        torch.testing.assert_close(
            topk.topk_weights[0].sort().values, (raw / raw.sum()).sort().values
        )

    def test_combine_normalizes_each_expert_output_before_weighting(self):
        moe: MuseSparkMoE = self._build().model.language_model.layers[0].mlp
        h = _TEXT_CONFIG["moe_hidden_size"]
        per_slot = torch.randn(3, 2, h).bfloat16()
        per_slot[1, 1] = 0  # A slot owned by another expert-parallel rank.
        weights = torch.rand(3, 2)
        out = moe._combine(per_slot, topk_weights=weights)

        norm = moe.experts.post_expert_norm
        expected = torch.zeros(3, h)
        for t in range(3):
            for s in range(2):
                expected[t] += weights[t, s] * norm(per_slot[t, s]).float()
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
        with self.assertRaisesRegex(RuntimeError, "per-expert outputs"):
            moe._combine(per_slot.sum(1), topk_weights=weights)


if __name__ == "__main__":
    unittest.main()
