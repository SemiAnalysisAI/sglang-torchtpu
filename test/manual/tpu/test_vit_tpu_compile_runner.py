"""CPU-runnable unit tests for the ViTTpuCompileRunner (initial version).

Covers the pure-Python logic introduced with the runner:
* ``_cu_seqlens_to_segment_ids`` — per-patch segment ids from cu_seqlens.
* ``ViTTpuCompileRunner.run`` input handling — unsqueeze x to ``[seq, 1, embed]``,
  build a ``[1, seq]`` segment_ids from cu_seqlens, and apply ``output_indices``.

The ``torch.compile(backend='tpu')`` path is stubbed (``_compiled_fn`` is
pre-set so it is never built), so no TPU is required.
"""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    ViTTpuCompileRunner,
    _cu_seqlens_to_segment_ids,
)
from sglang.test.test_utils import CustomTestCase


class _Block(torch.nn.Module):
    def forward(
        self,
        x,
        cu_seqlens=None,
        rotary_pos_emb_cos=None,
        rotary_pos_emb_sin=None,
        segment_ids=None,
    ):
        return x


class _FakeViT:
    spatial_merge_size = 2

    def __init__(self):
        self.blocks = [_Block()]
        self.deepstack_visual_indexes = []
        self.deepstack_merger_list = None


def _runner(seq):
    r = ViTTpuCompileRunner(_FakeViT())
    # Pin a single exact bucket == seq so this stays on the no-pad path.
    r._patch_buckets = [seq]
    return r


class TestCuSeqlensToSegmentIds(CustomTestCase):
    def test_multi_segment(self):
        cu = torch.tensor([0, 3, 7], dtype=torch.int32)
        seg = _cu_seqlens_to_segment_ids(cu, 7, torch.device("cpu"))
        self.assertEqual(seg.shape, (1, 7))
        self.assertEqual(seg.tolist(), [[0, 0, 0, 1, 1, 1, 1]])

    def test_single_segment(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        seg = _cu_seqlens_to_segment_ids(cu, 5, torch.device("cpu"))
        self.assertEqual(seg.tolist(), [[0, 0, 0, 0, 0]])


class TestRunBasic(CustomTestCase):
    def test_builds_segment_ids_and_unsqueezes(self):
        seq, embed = 64, 8
        r = _runner(seq)
        seen = {}

        def stub(x, cu, cos, sin, seg):
            seen["x"] = tuple(x.shape)
            seen["seg"] = tuple(seg.shape)
            return torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(1).repeat(1, embed)

        r._compiled_fn = stub
        out = r.run(
            torch.zeros(seq, embed),
            torch.tensor([0, seq], dtype=torch.int32),
            rotary_pos_emb_cos=torch.zeros(seq, 4),
            rotary_pos_emb_sin=torch.zeros(seq, 4),
        )
        self.assertEqual(seen["x"], (seq, 1, embed))  # x unsqueezed to [seq,1,embed]
        self.assertEqual(seen["seg"], (1, seq))  # per-patch segment ids
        self.assertEqual(out.shape[0], seq)

    def test_output_indices_gather(self):
        seq, embed = 64, 8
        r = _runner(seq)
        r._compiled_fn = lambda x, cu, cos, sin, seg: (
            torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(1).repeat(1, embed)
        )
        idx = torch.tensor([0, 2, 5], dtype=torch.int64)
        out = r.run(
            torch.zeros(seq, embed),
            torch.tensor([0, seq], dtype=torch.int32),
            rotary_pos_emb_cos=torch.zeros(seq, 4),
            rotary_pos_emb_sin=torch.zeros(seq, 4),
            output_indices=idx,
        )
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out[:, 0].tolist(), [0.0, 2.0, 5.0])  # rows gathered


if __name__ == "__main__":
    unittest.main()
