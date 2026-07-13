"""CPU-only unit test for the TPU NVFP4 (``modelopt_fp4``) MoE weight prep.

Validates ``prepare_tpu_nvfp4_moe_weights`` without a device: the two-level scale
fold, the gmm_v2 scale/weight layout, and that unused params are cleared. Pins
the highest-risk detail (the gate/up-half global mapping) in per-commit CI; the
on-device parity check lives in ``test/manual/tpu/test_nvfp4_native_moe.py``.
Helpers are self-contained so the CI test has no dependency on that manual test.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

NVFP4_BLOCK = 16
E2M1_MAX = 6.0
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# Real routed-MoE expert dims (hidden 4096, intermediate 4096); E is a small
# subset since the fold is per-expert and shape-independent in E.
REAL_E, REAL_HIDDEN, REAL_INTERMEDIATE = 2, 4096, 4096


def _quantize_rows_nvfp4(x, g, block_size=NVFP4_BLOCK):
    """Quantize ``(E, R, K)`` fp32 to NVFP4 with a per-expert global ``g`` (E,).

    Returns ``(packed uint8 (E, R, K/2), block_scale fp8-e4m3 (E, R, K/16))``.
    The stored fp8 block scale is ``(amax/6) / global`` so that
    ``code * fp8_scale.float() * global`` reconstructs the value.
    """
    E, R, K = x.shape
    nb = K // block_size
    blocks = x.float().reshape(E, R, nb, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    real = (amax / E2M1_MAX).clamp(min=1e-12)  # real per-block scale
    g4 = g.float()[:, None, None, None]
    fp8 = (real / g4).to(torch.float8_e4m3fn)  # stored fp8 block scale
    eff = (fp8.float() * g4).clamp(min=1e-12)  # effective scale used to code
    q = (blocks / eff).clamp(-E2M1_MAX, E2M1_MAX)
    sign = torch.sign(q)
    sign_bit = ((2 - sign) // 2).to(torch.uint8)  # 0 for +, 1 for -
    ord_ = torch.bucketize(q.abs(), E2M1_BOUNDS).to(torch.uint8)  # e2m1 magnitude
    codes = sign_bit * 0b1000 + ord_  # (E, R, nb, block_size)
    low = codes[..., 0::2]
    high = codes[..., 1::2] << 4
    packed = (low + high).to(torch.uint8).reshape(E, R, nb * (block_size // 2))
    return packed, fp8.squeeze(-1)


def _make_nvfp4_moe_layer(E, hidden, intermediate, seed):
    """Build synthetic NVFP4 MoE tensors matching ModelOpt checkpoint params."""
    g = torch.Generator().manual_seed(seed)
    I = intermediate
    w1f = (torch.randn(E, I, hidden, generator=g) * 0.1).float()  # gate
    w3f = (torch.randn(E, I, hidden, generator=g) * 0.1).float()  # up
    w2f = (torch.randn(E, hidden, I, generator=g) * 0.1).float()  # down
    # Distinct per-half / per-expert globals so the fold is actually exercised.
    g_gate = 0.04 + 0.01 * torch.rand(E, generator=g)
    g_up = 0.07 + 0.01 * torch.rand(E, generator=g)
    g_down = 0.05 + 0.01 * torch.rand(E, generator=g)

    w1_p, w1_s = _quantize_rows_nvfp4(w1f, g_gate)
    w3_p, w3_s = _quantize_rows_nvfp4(w3f, g_up)
    w2_p, w2_s = _quantize_rows_nvfp4(w2f, g_down)

    return {
        "w13_weight": torch.cat([w1_p, w3_p], dim=1),  # (E, 2I, H/2) u8, [Gate, Up]
        "w2_weight": w2_p,
        "w13_weight_scale": torch.cat([w1_s, w3_s], dim=1),  # (E, 2I, H/16) fp8
        "w2_weight_scale": w2_s,
        "w13_weight_scale_2": torch.stack([g_gate, g_up], dim=1),  # (E, 2) fp32
        "w2_weight_scale_2": g_down,
    }


def _build_fake_layer(data):
    """Wrap synthetic tensors in an nn.Module with a moe_runner_config."""
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    E = data["w13_weight"].shape[0]
    for name, tensor in data.items():
        layer.register_parameter(
            name, torch.nn.Parameter(tensor, requires_grad=False)
        )
    # Params the TPU path must clear.
    layer.w13_input_scale = torch.nn.Parameter(torch.ones(E, 2), requires_grad=False)
    layer.w2_input_scale = torch.nn.Parameter(torch.ones(E), requires_grad=False)
    return layer


class TestNvfp4ScaleFold(CustomTestCase):
    """Verify the scale fold + gmm_v2 layout in prepare_tpu_nvfp4_moe_weights."""

    def test_fold_and_layout(self):
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        # Real routed-MoE dims; intermediate is 128-aligned => pad_n == 0.
        E, hidden, intermediate = REAL_E, REAL_HIDDEN, REAL_INTERMEDIATE
        data = _make_nvfp4_moe_layer(E, hidden, intermediate, seed=0)
        layer = _build_fake_layer(data)

        # Expected fold, computed independently.
        I = intermediate
        gg = data["w13_weight_scale_2"][:, 0]
        gu = data["w13_weight_scale_2"][:, 1]
        g2 = data["w2_weight_scale_2"]
        exp_w13_s = data["w13_weight_scale"].float().clone()
        exp_w13_s[:, :I, :] *= gg[:, None, None]
        exp_w13_s[:, I:, :] *= gu[:, None, None]
        exp_w13_s = exp_w13_s.permute(0, 2, 1).contiguous().unsqueeze(2)
        exp_w2_s = (
            (data["w2_weight_scale"].float() * g2[:, None, None])
            .permute(0, 2, 1)
            .contiguous()
            .unsqueeze(2)
        )
        exp_w13_w = data["w13_weight"].permute(0, 2, 1).contiguous()
        exp_w2_w = data["w2_weight"].permute(0, 2, 1).contiguous()

        prepare_tpu_nvfp4_moe_weights(layer, group_size=NVFP4_BLOCK)

        # Shapes: gmm_v2 (E, K_packed, N) weights + (E, num_blocks, 1, N) scales.
        self.assertEqual(tuple(layer.w13_weight.shape), (E, hidden // 2, 2 * I))
        self.assertEqual(tuple(layer.w2_weight.shape), (E, I // 2, hidden))
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape), (E, hidden // NVFP4_BLOCK, 1, 2 * I)
        )
        self.assertEqual(
            tuple(layer.w2_weight_scale.shape), (E, I // NVFP4_BLOCK, 1, hidden)
        )
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)

        # Values.
        self.assertTrue(torch.equal(layer.w13_weight.data, exp_w13_w))
        self.assertTrue(torch.equal(layer.w2_weight.data, exp_w2_w))
        torch.testing.assert_close(layer.w13_weight_scale.data, exp_w13_s)
        torch.testing.assert_close(layer.w2_weight_scale.data, exp_w2_s)

        # Unused params cleared.
        for name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            self.assertIsNone(getattr(layer, name, None), f"{name} not cleared")

    def test_padding_unaligned_intermediate(self):
        """A non-128-aligned intermediate is zero-padded up to ``align_to``; the
        padded weight and scale regions must be zero so they contribute nothing
        through SwiGLU (silu(0)*0) and GMM2. Real dims are 128-aligned, so this
        path only fires for odd shapes — exercise it explicitly."""
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        E, hidden, I = 2, 256, 112  # 112 -> pad to 128 (pad_n=16, mult of 16 & 2)
        I_pad = 128
        data = _make_nvfp4_moe_layer(E, hidden, I, seed=1)
        layer = _build_fake_layer(data)
        prepare_tpu_nvfp4_moe_weights(layer, group_size=NVFP4_BLOCK, align_to=I_pad)

        # Shapes use the padded intermediate.
        self.assertEqual(tuple(layer.w13_weight.shape), (E, hidden // 2, 2 * I_pad))
        self.assertEqual(tuple(layer.w2_weight.shape), (E, I_pad // 2, hidden))
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape), (E, hidden // NVFP4_BLOCK, 1, 2 * I_pad)
        )
        self.assertEqual(
            tuple(layer.w2_weight_scale.shape), (E, I_pad // NVFP4_BLOCK, 1, hidden)
        )

        # W13 N-axis is [gate(I_pad), up(I_pad)]; the [I:I_pad] tail of each half
        # (weights and folded scales) must be zero.
        w13w, w13s = layer.w13_weight, layer.w13_weight_scale
        self.assertTrue((w13w[:, :, I:I_pad] == 0).all())
        self.assertTrue((w13w[:, :, I_pad + I :] == 0).all())
        self.assertTrue((w13s[:, :, :, I:I_pad] == 0).all())
        self.assertTrue((w13s[:, :, :, I_pad + I :] == 0).all())

        # W2 K-axis padded: packed rows past I/2 and scale blocks past I/16.
        self.assertTrue((layer.w2_weight[:, I // 2 :, :] == 0).all())
        self.assertTrue((layer.w2_weight_scale[:, I // NVFP4_BLOCK :, :, :] == 0).all())

    def test_shared_global_2d_single_column(self):
        """A gated ``(E, 1)`` global folds via the shared-scalar branch, not the
        per-half split (regression guard for the empty-slice edge case)."""
        from sglang.srt.hardware_backend.tpu.fp4_moe import _fold_w13_global_scale

        E, two_I, nb = 2, 8, 4
        block_scale = torch.ones(E, two_I, nb, dtype=torch.float8_e4m3fn)
        g = torch.tensor([[2.0], [3.0]])  # (E, 1), 2-D but single column

        folded = _fold_w13_global_scale(block_scale, g, is_gated=True)

        expected = block_scale.float() * g[:, 0][:, None, None]
        torch.testing.assert_close(folded, expected)


if __name__ == "__main__":
    unittest.main()
