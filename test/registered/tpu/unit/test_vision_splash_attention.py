"""On-device coverage of the opt-in vision Splash path and runtime masks."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")


def _run_splash_attention() -> None:
    import torch
    import torch_tpu  # noqa: F401

    from sglang_tpu.compile_cache import (
        add_pallas_kernel_sources_to_compile_cache_key,
    )
    from sglang_tpu.env_override import apply_tpu_env_overrides

    apply_tpu_env_overrides()
    add_pallas_kernel_sources_to_compile_cache_key()

    from sglang.srt.layers.attention.vision import VisionTpuAttention

    torch.manual_seed(1729)
    batch, heads, sequence, dim = 2, 2, 1008, 96
    q, k, v = [
        torch.randn(batch, heads, sequence, dim).to(torch.bfloat16) for _ in range(3)
    ]
    inputs = [
        x.transpose(1, 2).reshape(batch * sequence, heads, dim).to("tpu")
        for x in (q, k, v)
    ]
    attention = VisionTpuAttention(dim, heads, heads)
    compiled = torch.compile(attention, backend="tpu", fullgraph=True, dynamic=False)

    # A real 24x42 patch grid: 768 left-tile patches and 240 right-tile
    # patches, interleaved by row. The wrapper supplies 16 isolated pad rows.
    sparse = (torch.arange(42) // 32).repeat(24).to(torch.int32)
    dense = torch.zeros(sequence, dtype=torch.int32)
    masks = [torch.stack((dense, sparse)), torch.stack((sparse, dense))]
    graphs_after_warmup = None
    first = None
    for index in (0, 1, 0):
        ids = masks[index]
        actual = compiled(*inputs, bsz=batch, segment_ids=ids.to("tpu")).cpu()
        actual = actual.reshape(batch, sequence, heads, dim).transpose(1, 2).float()
        scores = (q.float() @ k.float().transpose(-1, -2)) * dim**-0.5
        same_segment = ids[:, None, :, None] == ids[:, None, None, :]
        expected = scores.masked_fill(~same_segment, -torch.inf).softmax(-1) @ v.float()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        assert (actual - expected).norm() / expected.norm() < 0.01

        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        if graphs_after_warmup is None:
            graphs_after_warmup = graphs
            first = actual
        else:
            assert graphs == graphs_after_warmup, "mask contents recompiled attention"
            if index == 0:
                torch.testing.assert_close(actual, first, atol=0, rtol=0)


class TestVisionSplashAttention(unittest.TestCase):
    def test_splash_preserves_batched_sparse_masks_and_reuses_compilation(self):
        environment = os.environ.copy()
        environment.update(
            SGLANG_PLATFORM="tpu",
            SGLANG_TPU_VISION_ATTENTION_BACKEND="splash",
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--splash-worker"],
            check=True,
            env=environment,
            timeout=300,
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["--splash-worker"]:
        _run_splash_attention()
    else:
        unittest.main()
