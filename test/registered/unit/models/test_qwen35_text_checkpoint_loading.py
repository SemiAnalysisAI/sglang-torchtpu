"""The text entry can consume native text or official multimodal weights lazily."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.qwen3_5_text import Qwen3_5ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("prefix", ["model.", "model.language_model."])
def test_text_weight_namespace_and_lazy_consumption(prefix):
    consumed = []

    class Body(nn.Module):
        def load_weights(self, weights):
            loaded = set()
            for name, weight in weights:
                assert name in (
                    "embed_tokens.weight",
                    "layers.0.mlp.experts.gate_up_proj",
                )
                consumed.append(name)
                loaded.add(name)
            return loaded

    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    model.model = Body()
    model.lm_head = nn.Linear(2, 3, bias=False)

    def weights():
        yield prefix + "embed_tokens.weight", torch.ones(3, 2)
        assert consumed == ["embed_tokens.weight"]
        yield "model.visual.patch_embed.weight", torch.ones(1)
        yield "mtp.layers.0.mlp.gate.weight", torch.ones(1)
        yield prefix + "layers.0.mlp.experts.gate_up_proj", torch.ones(1)
        yield "lm_head.weight", torch.full((3, 2), 7.0)

    loaded = model.load_weights(weights())
    assert loaded == {
        "model.embed_tokens.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "lm_head.weight",
    }
    torch.testing.assert_close(model.lm_head.weight, torch.full((3, 2), 7.0))
