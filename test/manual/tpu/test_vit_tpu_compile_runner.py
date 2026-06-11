"""CPU-runnable unit tests for ViTTpuCompileRunner pure-Python logic.

Covers the parts of ``vit_tpu_compile_runner.py`` that are plain PyTorch/Python
and need no JAX / TPU device:

* ``_compute_patch_counts`` — square patch-count bucket generation.
* ``_cu_seqlens_to_segment_ids`` — per-patch segment ids from cu_seqlens.
* ``ViTTpuCompileRunner._bucket_for`` — nearest precompiled bucket / None when
  the patch count exceeds the largest bucket.
* ``ViTTpuCompileRunner.run`` pad-to-bucket behaviour — x/cos/sin are padded up
  to the bucket, a dedicated padding attention segment is appended, and the
  merged output is sliced back to the real (unpadded) row count.

The ``torch.compile(backend='tpu')`` path is stubbed (``_compiled_fn`` is
pre-set to a recording function so ``_ensure_compiled`` returns early), so no
TPU is required. Kernel numerics live with the flash-op / kernel tests.
"""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    ViTTpuCompileRunner,
    _compute_patch_counts,
    _cu_seqlens_to_segment_ids,
)
from sglang.test.test_utils import CustomTestCase


class _Block(torch.nn.Module):
    # Signature must expose ``segment_ids`` so the runner takes the compiled
    # (non-fallback) path.
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


def _make_runner():
    return ViTTpuCompileRunner(_FakeViT())


class TestComputePatchCounts(CustomTestCase):
    def test_square_buckets(self):
        counts = _compute_patch_counts(2, min_patches=196, max_patches=1600)
        self.assertEqual(counts[0], 196)  # 14^2
        self.assertEqual(counts[-1], 1600)  # 40^2
        for c in counts:
            r = round(c**0.5)
            self.assertEqual(r * r, c)  # perfect square
            self.assertEqual(r % 2, 0)  # grid is a multiple of merge_size

    def test_respects_bounds(self):
        counts = _compute_patch_counts(2, min_patches=300, max_patches=1100)
        self.assertTrue(all(300 <= c <= 1100 for c in counts))
        self.assertEqual(counts[0], 324)  # 18^2 is the first >= 300 even-grid sq


class TestCuSeqlensToSegmentIds(CustomTestCase):
    def test_assignment(self):
        cu = torch.tensor([0, 3, 7], dtype=torch.int32)
        seg = _cu_seqlens_to_segment_ids(cu, 7, torch.device("cpu"))
        self.assertEqual(seg.shape, (1, 7))
        self.assertEqual(seg.tolist(), [[0, 0, 0, 1, 1, 1, 1]])

    def test_single_segment(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        seg = _cu_seqlens_to_segment_ids(cu, 5, torch.device("cpu"))
        self.assertEqual(seg.tolist(), [[0, 0, 0, 0, 0]])


class TestBucketFor(CustomTestCase):
    def test_nearest_and_oversize(self):
        r = _make_runner()
        r._patch_buckets = [196, 256, 1024, 2304]
        self.assertEqual(r._bucket_for(196), 196)  # exact
        self.assertEqual(r._bucket_for(200), 256)  # rounds up
        self.assertEqual(r._bucket_for(1025), 2304)
        self.assertIsNone(r._bucket_for(9999))  # exceeds max -> fallback


class TestRunPadding(CustomTestCase):
    def test_pads_to_bucket_and_slices_back(self):
        r = _make_runner()
        r._patch_buckets = [256]
        embed, rotary_dim = 8, 4
        merge_sq = _FakeViT.spatial_merge_size**2  # 4

        seen = {}

        def stub(x, cu, cos, sin, seg):
            seen["x"] = tuple(x.shape)
            seen["cu"] = tuple(cu.shape)
            seen["cos"] = tuple(cos.shape)
            seen["sin"] = tuple(sin.shape)
            seen["seg"] = tuple(seg.shape)
            # Merger reduces rows by merge_size**2.
            return torch.zeros(x.shape[0] // merge_sq, embed)

        r._compiled_fn = stub  # makes _ensure_compiled() a no-op

        seq = 100
        x = torch.zeros(seq, embed)
        cu = torch.tensor([0, 40, seq], dtype=torch.int32)
        cos = torch.zeros(seq, rotary_dim)
        sin = torch.zeros(seq, rotary_dim)

        out = r.run(x, cu, rotary_pos_emb_cos=cos, rotary_pos_emb_sin=sin)

        # Inputs were padded up to the 256 bucket...
        self.assertEqual(seen["x"], (256, 1, embed))
        self.assertEqual(seen["cos"], (256, rotary_dim))
        self.assertEqual(seen["sin"], (256, rotary_dim))
        # ...with a per-patch segment_ids of the padded length.
        self.assertEqual(seen["seg"], (1, 256))
        # Runtime cu_seqlens may have many segments, but the compiled graph sees
        # a stable length-2 tensor and uses segment_ids for the real mask.
        self.assertEqual(seen["cu"], (2,))
        # ...and the merged output sliced back to the real row count.
        self.assertEqual(out.shape[0], seq // merge_sq)

    def test_exact_bucket_no_pad(self):
        r = _make_runner()
        r._patch_buckets = [256]
        embed, merge_sq = 8, 4
        seen = {}

        def stub(x, cu, cos, sin, seg):
            seen["x"] = tuple(x.shape)
            return torch.zeros(x.shape[0] // merge_sq, embed)

        r._compiled_fn = stub
        seq = 256
        out = r.run(
            torch.zeros(seq, embed),
            torch.tensor([0, seq], dtype=torch.int32),
            rotary_pos_emb_cos=torch.zeros(seq, 4),
            rotary_pos_emb_sin=torch.zeros(seq, 4),
        )
        self.assertEqual(seen["x"], (256, 1, embed))  # already a bucket
        self.assertEqual(out.shape[0], seq // merge_sq)


if __name__ == "__main__":
    unittest.main()
