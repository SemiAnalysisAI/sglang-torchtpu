"""Unit tests for the TPU W4A8 MoE fp8-requant mode (int4 groups -> channelwise fp8).

Guards the two host-side transforms behind SGLANG_TPU_USE_W4A8_MOE=fp8-requant:
the offset-binary nibble unpack must invert the checkpoint packing exactly, and
the fp8 requant must reproduce the group-dequantized weights within e4m3
round-to-nearest error.
"""

import unittest

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.tpu.w4a16_moe import (
    _unpack_int4_offset_binary,
    prepare_tpu_w4a8_moe_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_EXPERTS = 2
K = 64
N = 16
GROUP_SIZE = 16


def _pack_offset_binary(q: torch.Tensor) -> torch.Tensor:
    """[E, K, N] int4 values in [-8, 7] -> [E, K//8, N] int32, LSB-first."""
    e, k, n = q.shape
    nibbles = (q.to(torch.int64) + 8).reshape(e, k // 8, 8, n)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64).view(1, 1, 8, 1)
    words = (nibbles << shifts).sum(dim=2)
    # Wrap to signed int32 (nibble 7 can set the sign bit).
    return words.to(torch.int32)


class _FakeMoELayer(torch.nn.Module):
    def __init__(self, packed13, scale13, packed2, scale2):
        super().__init__()
        self.w13_weight_packed = Parameter(packed13, requires_grad=False)
        self.w13_weight_scale = Parameter(scale13, requires_grad=False)
        self.w2_weight_packed = Parameter(packed2, requires_grad=False)
        self.w2_weight_scale = Parameter(scale2, requires_grad=False)


class TestUnpackInt4OffsetBinary(CustomTestCase):
    def test_roundtrip_inverts_packing(self):
        torch.manual_seed(0)
        q = torch.randint(-8, 8, (NUM_EXPERTS, K, N))
        unpacked = _unpack_int4_offset_binary(_pack_offset_binary(q))
        self.assertEqual(unpacked.dtype, torch.int8)
        torch.testing.assert_close(unpacked.to(q.dtype), q)


class TestPrepareTpuW4A8MoeWeights(CustomTestCase):
    def test_requant_matches_group_dequant_reference(self):
        torch.manual_seed(1)
        q = torch.randint(-8, 8, (NUM_EXPERTS, K, N))
        scales = (torch.rand(NUM_EXPERTS, K // GROUP_SIZE, N) * 0.1 + 0.01).to(
            torch.bfloat16
        )
        ref = q.float() * scales.float().repeat_interleave(GROUP_SIZE, dim=1)

        layer = _FakeMoELayer(
            _pack_offset_binary(q),
            scales.clone(),
            _pack_offset_binary(q),
            scales.clone(),
        )
        prepare_tpu_w4a8_moe_weights(layer, group_size=GROUP_SIZE)

        self.assertTrue(layer.is_tpu_w4a8_converted)
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.w13_weight.shape), (NUM_EXPERTS, K, N))
        # gmm_v2 RHS scale layout, own storage.
        self.assertEqual(tuple(layer.w13_weight_scale.shape), (NUM_EXPERTS, 1, 1, N))
        self.assertIsNone(layer.w13_weight_scale.data._base)
        # Packed source is dropped.
        self.assertIsNone(layer.w13_weight_packed)

        dequant = layer.w13_weight.float() * layer.w13_weight_scale.data.float().view(
            NUM_EXPERTS, 1, N
        )
        # e4m3 round-to-nearest: |err| <= |value| * 2^-4 (half ulp, 3 mantissa
        # bits), plus one denormal step of the per-channel scale for tiny values.
        denorm_step = (
            ref.abs().amax(dim=1, keepdim=True)
            / torch.finfo(torch.float8_e4m3fn).max
            * 2**-6
        )
        self.assertTrue(
            (dequant - ref).abs().le(ref.abs() * 2**-4 + denorm_step + 1e-8).all()
        )


class TestVmemFp8Preparation(CustomTestCase):
    def test_packed_storage_scales_and_idempotence(self):
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        torch.manual_seed(4)
        q = torch.randint(-8, 8, (2, 64, 16))
        q[:, :, 0] = 0
        scales = (torch.rand(2, 2, 16) * 0.1).bfloat16()
        packed = _pack_offset_binary(q)
        layer = _FakeMoELayer(
            packed.clone(), scales.clone(), packed.clone(), scales.clone()
        )
        dequant = q.bfloat16() * scales.repeat_interleave(32, dim=1)
        channel = (dequant.float().abs().amax(dim=1, keepdim=True) / 448).clamp_min(
            1e-12
        )
        normalized = scales.float() / channel
        with envs.SGLANG_TPU_USE_W4A8_MOE.override("vmem-fp8"):
            prepare_tpu_w4a16_moe_weights(layer, 32)
            for prefix in ("w13", "w2"):
                weight = getattr(layer, prefix + "_weight_packed")
                group_scale = getattr(layer, prefix + "_weight_scale")
                channel_scale = getattr(layer, prefix + "_vmem_channel_scale")
                self.assertEqual(weight.dtype, torch.int32)
                self.assertEqual(weight.numel(), packed.numel())
                torch.testing.assert_close(weight, packed ^ -2004318072, rtol=0, atol=0)
                torch.testing.assert_close(
                    group_scale, normalized.unsqueeze(2), rtol=0, atol=0
                )
                torch.testing.assert_close(channel_scale, channel, rtol=0, atol=0)
                self.assertIsNone(group_scale.data._base)
                self.assertTrue(torch.isfinite(group_scale).all())
            before = layer.w13_weight_packed.clone()
            prepare_tpu_w4a16_moe_weights(layer, 32)
            torch.testing.assert_close(layer.w13_weight_packed, before, rtol=0, atol=0)

    def test_unsupported_groups_fail_before_mutation(self):
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        packed = torch.zeros(2, 8, 16, dtype=torch.int32)
        scales = torch.ones(2, 4, 16, dtype=torch.bfloat16)
        layer = _FakeMoELayer(
            packed.clone(), scales.clone(), packed.clone(), scales.clone()
        )
        with envs.SGLANG_TPU_USE_W4A8_MOE.override("vmem-fp8"):
            with self.assertRaisesRegex(ValueError, "group size 32"):
                prepare_tpu_w4a16_moe_weights(layer, 16)
        torch.testing.assert_close(layer.w13_weight_packed, packed, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
