"""Byte-level contracts for host MXFP4 -> G512 expert preparation."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.dsv41.quantization import (
    _quantizer,
    dequantize_checkpoint,
    prepare_expert_matrix,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def reference(packed, scales, gate_up, h, i):
    """Previous dense Torch decode + JAX quantizer, with per-expert tiling."""
    weights, block_scales = [], []
    for e in range(packed.shape[0]):
        values = dequantize_checkpoint(packed[e : e + 1], scales[e : e + 1])
        if gate_up:
            gate, up = values.chunk(2, dim=1)
            pad = (0, h - values.shape[-1], 0, i - gate.shape[1])
            values = torch.cat((F.pad(gate, pad), F.pad(up, pad)), dim=1)
        else:
            values = F.pad(values, (0, i - values.shape[-1], 0, h - values.shape[1]))
        w, s = _quantizer()(values.transpose(1, 2).numpy())
        weights.append(torch.from_numpy(np.array(w)))
        s = torch.from_numpy(np.array(s))
        block_scales.append(F.pad(s, (0, 0, 0, -s.shape[1] % 8)).flatten(0, 1))
    return torch.cat(weights), torch.cat(block_scales)


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("k", [32, 512, 544])
@pytest.mark.parametrize("scale_kind", ["float", "e8m0", "zero", "extreme"])
def test_packed_conversion_matches_dense_bits(gate_up, k, scale_kind):
    rng = np.random.default_rng(14)
    packed = rng.integers(0, 256, (2, 8, k // 2), dtype=np.uint8)
    packed.reshape(-1)[:16] = np.arange(16, dtype=np.uint8) * 17
    scales = rng.uniform(-2, 2, (2, 8, k // 32)).astype(np.float32)
    if scale_kind == "e8m0":
        scales = rng.integers(110, 145, scales.shape, dtype=np.uint8)
    elif scale_kind == "zero":
        scales.reshape(-1)[::2] = 0
        scales.reshape(-1)[1::2] = -0.0
    elif scale_kind == "extreme":
        scales.reshape(-1)[:8] = [
            np.nan,
            np.inf,
            -np.inf,
            np.finfo(np.float32).max,
            np.nextafter(np.float32(0), np.float32(1)),
            np.finfo(np.float32).tiny,
            0.0,
            -0.0,
        ]
    packed = torch.from_numpy(packed).view(torch.int8)
    scales = torch.from_numpy(scales)
    h, i = (1024, 512) if gate_up else (512, 1024)
    expected_w, expected_s = reference(packed, scales, gate_up, h, i)
    weight, scale = prepare_expert_matrix(
        packed, scales, gate_up=gate_up, padded_hidden=h, padded_inter=i
    )
    assert weight.dtype == torch.float4_e2m1fn_x2
    assert weight.device.type == scale.device.type == "cpu"
    assert torch.equal(weight.view(torch.uint8), expected_w)
    assert torch.equal(scale.view(torch.int32), expected_s.view(torch.int32))


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("scale_kind", ["subnormal", "tiny", "boundary"])
def test_no_max_code_and_finite_underflow_matches_dense(gate_up, scale_kind):
    # Each row omits magnitudes above one chosen code, including all-zero rows.
    # This catches the incorrect shortcut max(abs(weight)) == 6 * abs(scale).
    codes = np.resize(np.arange(16, dtype=np.uint8), (1, 16, 512))
    codes &= np.arange(16, dtype=np.uint8)[None, :, None]
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    if scale_kind == "subnormal":
        choices = np.array(
            [
                np.nextafter(np.float32(0), np.float32(1)),
                np.finfo(np.float32).tiny / 2,
                -np.finfo(np.float32).tiny / 2,
                np.finfo(np.float32).tiny,
            ],
            dtype=np.float32,
        )
    elif scale_kind == "tiny":
        choices = np.array(
            [
                np.finfo(np.float32).tiny,
                np.nextafter(np.finfo(np.float32).tiny, np.float32(0)),
                -np.finfo(np.float32).tiny,
            ],
            dtype=np.float32,
        )
    else:
        # Adjacent scales make reciprocal rounding / E2M1 ties observable.
        choices = np.array(
            [
                1.0,
                np.nextafter(np.float32(1), np.float32(0)),
                np.nextafter(np.float32(1), np.float32(2)),
                -1.0,
            ],
            dtype=np.float32,
        )
    scales = np.resize(choices, (1, 16, 16))
    packed, scales = torch.from_numpy(packed).view(torch.int8), torch.from_numpy(scales)
    expected_w, expected_s = reference(packed, scales, gate_up, 512, 512)
    weight, scale = prepare_expert_matrix(
        packed, scales, gate_up=gate_up, padded_hidden=512, padded_inter=512
    )
    assert torch.equal(weight.view(torch.uint8), expected_w)
    assert torch.equal(scale.view(torch.int32), expected_s.view(torch.int32))


def test_rejects_invalid_checkpoint_block_or_padding():
    with pytest.raises(ValueError, match="block32"):
        prepare_expert_matrix(
            torch.zeros(1, 8, 17, dtype=torch.int8),
            torch.ones(1, 8, 1),
            gate_up=False,
            padded_hidden=512,
            padded_inter=512,
        )
