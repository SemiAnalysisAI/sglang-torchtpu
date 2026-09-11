import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding import rope_variant
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make(rotary_dim, head_size, neox, dtype):
    # Build with the TPU flag on so the tables are registered, on CPU tensors.
    with patch.object(rope_variant, "_is_tpu", True):
        return DeepseekScalingRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=256,
            base=10000,
            is_neox_style=neox,
            scaling_factor=4.0,
            dtype=dtype,
            device="cpu",
        )


class TestDeepseekRopeTpuTables(CustomTestCase):
    def test_tpu_path_is_bit_identical_to_native(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for neox in (False, True):
                for rotary_dim, head_size in ((64, 64), (64, 192)):
                    for use_offsets in (False, True):
                        with self.subTest(
                            dtype=dtype,
                            neox=neox,
                            rotary_dim=rotary_dim,
                            head_size=head_size,
                            offsets=use_offsets,
                        ):
                            if neox and rotary_dim < head_size:
                                # forward_native's neox branch cannot run
                                # this shape (its cos repeat adds a dim).
                                continue
                            rope = _make(rotary_dim, head_size, neox, dtype)
                            t = 7
                            positions = torch.randint(0, 900, (t,))
                            offsets = (
                                torch.randint(0, 100, (t,)) if use_offsets else None
                            )
                            q = torch.randn(t, 16, head_size, dtype=dtype)
                            k = torch.randn(t, 1, head_size, dtype=dtype)
                            with patch.object(rope_variant, "_is_tpu", False):
                                q_ref, k_ref = rope.forward_native(
                                    positions, q, k, offsets
                                )
                            # The native neox branch returns a leading dim of 1.
                            q_ref, k_ref = q_ref.reshape(q.shape), k_ref.reshape(
                                k.shape
                            )
                            q_tpu, k_tpu = rope._forward_tpu(positions, q, k, offsets)
                            self.assertEqual(tuple(q_tpu.shape), tuple(q.shape))
                            self.assertEqual(q_tpu.dtype, dtype)
                            torch.testing.assert_close(q_tpu, q_ref, atol=0, rtol=0)
                            torch.testing.assert_close(k_tpu, k_ref, atol=0, rtol=0)

    def test_tables_are_non_persistent_buffers(self):
        rope = _make(64, 64, True, torch.bfloat16)
        names = dict(rope.named_buffers())
        self.assertIn("_tpu_cos_sin", names)
        self.assertIn("_tpu_rot", names)
        self.assertNotIn("_tpu_cos_sin", rope.state_dict())
        self.assertEqual(tuple(names["_tpu_rot"].shape), (64, 64))
        self.assertEqual(names["_tpu_cos_sin"].shape[-1], 2 * 64)


if __name__ == "__main__":
    unittest.main()
