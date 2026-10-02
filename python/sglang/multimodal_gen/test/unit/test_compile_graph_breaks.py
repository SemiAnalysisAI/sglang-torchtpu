"""Code that runs inside a torch.compile'd DiT forward must trace as one graph."""

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.layers.layernorm import FP32LayerNorm
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


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
