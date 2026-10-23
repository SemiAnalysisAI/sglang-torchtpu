# SPDX-License-Identifier: Apache-2.0
"""V4.1 FP32 pair state and EPIC FP8-rounded BF16 index operands."""

import torch
import torch.nn.functional as F


def pair_compress(kv, score, req, positions, valid, state, ring_size):
    """Pair adjacent positions, including an even token from the previous chunk.

    Read the old ring before writing this chunk. Only its last ``ring_size`` rows
    per request may write live slots; discarded and padded rows use the spare row.
    The returned value is meaningful only at odd, valid positions.
    """
    dim = kv.shape[-1]
    previous = req * ring_size + (positions - 1) % ring_size
    previous = torch.where(valid & (positions > 0), previous, state.shape[0] - 1)
    carried = state[previous]
    in_batch = torch.cat(
        [
            valid.new_zeros(1),
            valid[1:]
            & valid[:-1]
            & (req[1:] == req[:-1])
            & (positions[1:] == positions[:-1] + 1),
        ]
    )
    previous_kv = torch.where(in_batch[:, None], torch.roll(kv, 1, 0), carried[:, :dim])
    previous_score = torch.where(
        in_batch[:, None], torch.roll(score, 1, 0), carried[:, dim:]
    )
    # A missing partner must not create NaNs in unused even/padding rows.
    previous_score = torch.where(
        (valid & (positions % 2 == 1))[:, None], previous_score, score
    )
    weights = torch.stack([previous_score, score], 1).softmax(1)
    pooled = (torch.stack([previous_kv, kv], 1) * weights).sum(1)

    keep = valid
    if kv.shape[0] > ring_size:
        keep = torch.cat(
            [
                valid[:-ring_size]
                & ((req[:-ring_size] != req[ring_size:]) | ~valid[ring_size:]),
                valid[-ring_size:],
            ]
        )
    slots = torch.where(
        keep, req * ring_size + positions % ring_size, state.shape[0] - 1
    )
    state.index_copy_(0, slots.to(torch.int64), torch.cat([kv, score], -1))
    state[-1, :dim] = 0
    state[-1, dim:] = float("-inf")
    return pooled


def compressor_project(compressor, x):
    # Both C1 and C2 keep their checkpoint projection weights in FP32.
    kv = F.linear(x.float(), compressor.wkv.weight)
    if compressor.compress_ratio == 1:
        return kv, None
    return kv, F.linear(x.float(), compressor.wgate.weight)


def round_index_fp8(values):
    """Round each D128 head to E4M3 with a power-of-two scale, then store BF16.

    Matches EPIC v2/ops.py: the BF16 index pool contains these rounded values;
    its physical halfplane packing does not perform another quantization.
    """
    x = values.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    rounded = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (rounded.float() * scale).to(torch.bfloat16)


def hc_combine(x_flat, pre, hc, out_dtype):
    """Unfused mHC combine; the model still owns shifted-pre ordering."""
    return (
        (x_flat.unflatten(-1, (hc, -1)).float() * pre.unsqueeze(-1))
        .sum(-2)
        .to(out_dtype)
    )


def signed_sqrt(dot, clamp_value):
    """FP32 signed square root, including the sign of negative zero."""
    magnitude = dot.abs().clamp_min(clamp_value).sqrt()
    # Integer comparison reads the sign bit; a floating comparison loses -0.
    return torch.where(dot.view(torch.int32) < 0, -magnitude, magnitude)


def engram_gate(x, kv, q_weight, k_weight, eps, clamp_value):
    """Core Engram gate with a bitcast/select form of FP32 copysign."""
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(signed_sqrt(dot, clamp_value))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
