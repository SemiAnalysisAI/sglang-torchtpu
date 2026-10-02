"""Code that runs inside a torch.compile'd DiT forward must trace as one graph."""

import torch

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.layers.layernorm import FP32LayerNorm
from sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope import (
    OneDRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.platforms import current_platform


def _compile(fn):
    torch._dynamo.reset()
    return torch.compile(fn, backend="eager", fullgraph=True)


def test_fp32_layernorm_traces_without_a_break():
    norm = FP32LayerNorm(16, eps=1e-5).to(torch.bfloat16)
    inputs = torch.randn(4, 16, dtype=torch.bfloat16)

    with torch.no_grad():
        # An eager call first fills the fp32 parameter cache.
        expected = norm(inputs)
        torch.testing.assert_close(_compile(norm)(inputs), expected)


def test_req_delegates_sampling_params_without_a_break():
    req = Req(sampling_params=SamplingParams(num_frames=5))

    def frames(x):
        return x * req.num_frames

    assert _compile(frames)(torch.ones(2)).tolist() == [5.0, 5.0]


def test_model_config_extra_attrs_resolve_without_a_break():
    config = LTX2Config()
    config.arch_config.ltx_variant = "ltx_2"

    def scale(x):
        if getattr(config, "ltx_variant", "ltx_2") == "ltx_2_3":
            return x * 2
        return x + getattr(config, "missing_attr", 1)

    assert _compile(scale)(torch.ones(2)).tolist() == [2.0, 2.0]


def test_1d_rope_traces_without_a_break():
    rope = OneDRotaryEmbedding(dim=8)
    pos = torch.arange(6)
    want = rope(pos)
    got = _compile(rope)(pos)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w)


def test_platform_amp_query_traces_without_a_break():
    def cast(x):
        return x if current_platform.is_amp_supported() else x.float()

    assert _compile(cast)(torch.ones(2)).tolist() == [1.0, 1.0]
