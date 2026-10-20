"""Kimi-K3's TPU vision runner computes what the tower's own layers compute.

The runner drives the encoder layers' weights with segment-id attention and
real-arithmetic RoPE; these check it against the layers' eager forward (per
image SDPA, complex RoPE) on real Kimi-K3 layers, with several images packed
and a padded tail, plus the policy's bucket ladder and unpadding. CPU tensors;
a segment-masked SDPA stands in for the Pallas flash kernel.
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.graph_runner.kimi_k3_vit_tpu_compile_runner import (
    KimiK3VisionTpuCompilePolicy,
    KimiK3VitTpuCompileRunner,
    rotate_interleaved,
)

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

HEADS, HEAD_DIM, HIDDEN = 2, 16, 24


class _SegmentSdpa:
    """VisionTpuAttention's contract on CPU: attend within a segment id."""

    supports_segment_ids = True

    def __init__(self, head_dim, num_heads, num_kv_heads):
        del head_dim, num_heads, num_kv_heads

    def __call__(self, q, k, v, bsz, segment_ids=None):
        ids = segment_ids[0]
        mask = ids[:, None] == ids[None, :]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
        )
        return out.transpose(0, 1)


def _encoder():
    from sglang.srt.models.kimi_k3_vl import MoonViT3dEncoder

    torch.manual_seed(0)
    encoder = MoonViT3dEncoder(
        hidden_dim=HIDDEN,
        num_layers=2,
        block_cfg={
            "num_heads": HEADS,
            "hidden_dim": HIDDEN,
            "qkv_hidden_size": HEADS * HEAD_DIM,
            "mlp_dim": 48,
            "norm_type": "rmsnorm",
            "attn_bias": False,
            "linear_bias": False,
        },
    ).float()
    with torch.no_grad():
        for param in encoder.parameters():
            param.copy_(torch.randn_like(param) * 0.2)
    return encoder


class TestKimiK3VitRunner(CustomTestCase):
    def test_real_rope_matches_complex_rope(self):
        from sglang.srt.models.kimi_k3_vl import apply_rope

        x = torch.randn(10, HEADS, HEAD_DIM)
        angles = torch.randn(10, HEAD_DIM // 2)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        want, _ = apply_rope(x, x, freqs_cis)
        got = rotate_interleaved(x, freqs_cis.real, freqs_cis.imag)
        torch.testing.assert_close(got, want)

    def test_runner_matches_the_layers_eager_forward(self):
        encoder = _encoder()
        runner = KimiK3VitTpuCompileRunner(encoder, attention_cls=_SegmentSdpa)
        # Two images (4x4 and 2x6 patches), packed as the tower packs them.
        grids = torch.tensor([[1, 4, 4], [1, 2, 6]])
        seq = 16 + 12
        x = torch.randn(seq, HIDDEN)
        with torch.no_grad():
            want = encoder(x, grids)
            freqs = encoder.rope_2d.get_freqs_cis(grids, x.device)
            cu = torch.tensor([0, 16, seq], dtype=torch.int32)
            got = runner.run(x, freqs.real, freqs.imag, cu)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_padding_never_reaches_real_patches(self):
        encoder = _encoder()
        runner = KimiK3VitTpuCompileRunner(encoder, attention_cls=_SegmentSdpa)
        policy = KimiK3VisionTpuCompilePolicy(encoder, owner=encoder)
        policy._patch_buckets = [36]
        policy.compiled_fn = MagicMock(side_effect=runner.forward)
        policy.eager_fn = MagicMock(side_effect=runner.forward)
        policy._install_run_adapter(runner)
        grids = torch.tensor([[1, 4, 4], [1, 2, 6]])
        x = torch.randn(28, HIDDEN)
        with torch.no_grad():
            freqs = encoder.rope_2d.get_freqs_cis(grids, x.device)
            want = encoder(x, grids)
            got = runner.run(
                x,
                freqs.real.contiguous(),
                freqs.imag.contiguous(),
                torch.tensor([0, 16, 28], dtype=torch.int32),
            )
        # One compiled call per image, each padded to the 36-patch bucket.
        shapes = [call.args[0].shape[0] for call in policy.compiled_fn.call_args_list]
        self.assertEqual(shapes, [36, 36])
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_buckets_align_to_the_merger(self):
        policy = KimiK3VisionTpuCompilePolicy(_encoder(), owner=None)
        with envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.override(1024):
            buckets = policy.compute_patch_buckets()
        self.assertEqual(buckets, [g * g for g in range(14, 33, 2)])
        with envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("256, 36"):
            policy._patch_buckets = None
            self.assertEqual(policy.compute_patch_buckets(), [36, 256])
        with envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("25"):
            with self.assertRaisesRegex(ValueError, "divisible by 4"):
                policy.compute_patch_buckets()


if __name__ == "__main__":
    unittest.main()
