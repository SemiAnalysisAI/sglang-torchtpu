"""The text entry can consume native text or official multimodal weights lazily."""

from types import SimpleNamespace
from unittest.mock import patch

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


@pytest.mark.parametrize("is_tpu", [False, True])
@pytest.mark.parametrize("has_mrope_positions", [False, True])
def test_text_forward_preserves_positions_for_idle_dp_batch(
    has_mrope_positions, is_tpu
):
    positions = torch.arange(2)
    mrope_positions = positions.expand(3, -1) if has_mrope_positions else None
    expected = mrope_positions if has_mrope_positions and not is_tpu else positions

    class Body(nn.Module):
        def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
            assert positions is expected
            return torch.ones(2, 4)

    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    nn.Module.__init__(model)
    model.model = Body()
    model.is_mrope_enabled = True
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.lm_head = nn.Identity()
    model.logits_processor = lambda ids, hidden, head, batch, aux: hidden
    with patch("sglang.srt.models.qwen3_5_text.current_platform") as platform:
        platform.is_tpu.return_value = is_tpu
        result = model.forward(
            torch.ones(2, dtype=torch.int64),
            positions,
            SimpleNamespace(mrope_positions=mrope_positions),
        )
    torch.testing.assert_close(result, torch.ones(2, 4))


@pytest.mark.parametrize("is_tpu", [False, True])
def test_text_compile_boundary_includes_logits_without_recompiling_backbone(is_tpu):
    from sglang.srt.compilation.support_torch_compile import (
        NoopCompilePolicy,
        get_torch_compile_policy,
        support_torch_compile,
    )
    from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
        DecoderTpuCompilePolicy,
        Qwen3VLDecoderTpuCompilePolicy,
    )

    class Body(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    support_torch_compile(Body, policy=Qwen3VLDecoderTpuCompilePolicy)
    config = SimpleNamespace(tie_word_embeddings=False, vocab_size=8, hidden_size=4)
    module = "sglang.srt.models.qwen3_5_text"
    with (
        patch.object(Qwen3_5ForCausalLM, "body_cls", Body),
        patch(
            f"{module}.get_pp_group",
            return_value=SimpleNamespace(is_last_rank=True, world_size=1),
        ),
        patch(
            f"{module}.get_parallel",
            return_value=SimpleNamespace(enable_dp_lm_head=True),
        ),
        patch(f"{module}.ParallelLMHead", return_value=nn.Identity()),
        patch(f"{module}.LogitsProcessor", return_value=nn.Identity()),
        patch(f"{module}.current_platform") as platform,
    ):
        platform.is_tpu.return_value = is_tpu
        model = Qwen3_5ForCausalLM(config)

    assert get_torch_compile_policy(model) is (
        DecoderTpuCompilePolicy if is_tpu else None
    )
    assert get_torch_compile_policy(model.model) is (
        NoopCompilePolicy if is_tpu else Qwen3VLDecoderTpuCompilePolicy
    )
    # The shared VL class retains its annotation for other wrappers.
    assert get_torch_compile_policy(Body) is Qwen3VLDecoderTpuCompilePolicy
