"""Equivalence tests for the Qwen3-VL TPU host-side ViT prologue.

forward_with_tpu_graph moves the pre-encoder work (bilinear pos-embed
interpolation, the merger-window reorder, rotary cos/sin) onto the host and
pads it to a patch bucket, so the device only ever sees static shapes. The
rewrite is meant to be a pure relocation: the real rows must stay bit-exact
against the eager implementations it replaces, and the padded rows must be
inert. These tests pin that down without needing a TPU or model weights.

    python -m pytest test/registered/unit/models/test_qwen3_vl_tpu_host_prologue.py -v
"""

import unittest
from functools import partial
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.utils import RotaryPosMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NUM_POS = 2304  # Qwen3-VL num_position_embeddings -> 48x48 grid
HIDDEN = 64
MERGE = 2
ROTARY_DIM = 8

GRID_CASES = {
    "single": [[1, 16, 16]],
    "single_large": [[1, 64, 98]],
    "multi_mixed": [[1, 16, 24], [1, 32, 12], [1, 8, 40]],
    "video": [[4, 16, 20]],
    "video_plus_image": [[3, 12, 16], [1, 20, 28], [2, 8, 8]],
    "many": [[1, 24, 24]] * 8,
}


def _vision_model():
    try:
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
    except Exception as exc:  # heavy optional deps (flashinfer, ...) unavailable
        raise unittest.SkipTest(f"cannot import Qwen3VLMoeVisionModel: {exc}")
    return Qwen3VLMoeVisionModel


def _stub(model_cls, *, dtype, align_corners):
    stub = SimpleNamespace(
        num_grid_per_side=int(NUM_POS**0.5),
        spatial_merge_size=MERGE,
        num_position_embeddings=NUM_POS,
        align_corners=align_corners,
        pos_embed=nn.Embedding(NUM_POS, HIDDEN).to(dtype=dtype),
        dtype=dtype,
        device=torch.device("cpu"),
        merge_window_order=RotaryPosMixin.merge_window_order,
        rot_pos_ids=RotaryPosMixin.rot_pos_ids,
    )
    stub._get_interpolation_indices = partial(
        model_cls._get_interpolation_indices, stub
    )
    stub._calculate_indices_and_weights = partial(
        model_cls._calculate_indices_and_weights, stub
    )
    stub._get_position_embedding = partial(model_cls._get_position_embedding, stub)

    cache = torch.randn(256, 2 * ROTARY_DIM, dtype=dtype)
    stub.rotary_pos_emb = SimpleNamespace(
        cos_sin_cache=cache,
        get_cos_sin=lambda seqlen: cache[:seqlen].chunk(2, dim=-1),
    )
    return stub


class TestQwen3VLTpuHostPrologue(CustomTestCase):
    def test_host_pos_embed_indices_match_eager_interpolation(self):
        """Folding the merger reorder into the gather index must not move values.

        _pos_embed_indices_cpu replaces the device-side split / per-image
        view+permute / cat of _get_position_embedding with one flat gather, so a
        wrong permutation would silently scramble position embeddings.
        """
        model_cls = _vision_model()
        for dtype in (torch.bfloat16, torch.float32):
            for align_corners in (False, True):
                stub = _stub(model_cls, dtype=dtype, align_corners=align_corners)
                for name, grid in GRID_CASES.items():
                    label = f"{name}/{dtype}/align_corners={align_corners}"
                    reference = model_cls.fast_pos_embed_interpolate(
                        stub, torch.tensor(grid)
                    )
                    real_len = reference.shape[0]
                    bucket_len = real_len + 3 * MERGE**2

                    idx, weights = model_cls._pos_embed_indices_cpu(
                        stub, grid, bucket_len=bucket_len
                    )
                    self.assertEqual(idx.shape, (4, bucket_len), label)
                    gathered = stub.pos_embed(idx.view(-1)).view(4, bucket_len, -1)
                    out = (gathered * weights.unsqueeze(-1)).sum(dim=0)

                    self.assertTrue(
                        torch.equal(out[:real_len], reference),
                        f"{label}: real rows not bit-exact, max|diff|="
                        f"{(out[:real_len].float() - reference.float()).abs().max():.3e}",
                    )
                    self.assertTrue(
                        torch.equal(out[real_len:], torch.zeros_like(out[real_len:])),
                        f"{label}: padded rows are not zero",
                    )


if __name__ == "__main__":
    unittest.main()
