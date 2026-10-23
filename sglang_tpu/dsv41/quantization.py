# SPDX-License-Identifier: Apache-2.0
"""Load-time MXFP4 block-32 to EPIC block-512 preparation.

Quantization uses the imported EPIC E2M1 cast, including ties to even.
The checkpoint loader retains SGLang names and contiguous expert ownership.
"""

from functools import cache

import numpy as np
import torch
import torch.nn.functional as F

_E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def dequantize_checkpoint(packed, scales):
    """Decode [E,N,K/2] E2M1 bytes and [E,N,K/32] effective FP32 scales."""
    codes = packed.detach().cpu().view(torch.uint8)
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    values = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1)
    values = values.flatten(-2).unflatten(-1, (-1, 32))
    scale = scales.detach().cpu()
    if scale.dtype == torch.uint8:
        scale = torch.exp2(scale.float() - 127)
    else:
        scale = scale.float()
    return (values * scale.unsqueeze(-1)).flatten(-2)


@cache
def _quantizer():
    import jax
    import jax.numpy as jnp

    from sglang_tpu.dsv41.kernels.fp4_quant import quantize_to_native_fp4_kmajor

    @jax.jit
    def quantize(values):
        quantized, scales = quantize_to_native_fp4_kmajor(values, 512, pack=False)
        # Torch native FP4 packs the final N dimension, unlike the old GMM
        # uint8 transport which packs K. The kernel sees unpacked [E,K,N].
        packed = jax.lax.bitcast_convert_type(
            quantized.reshape(*quantized.shape[:-1], quantized.shape[-1] // 2, 2),
            jnp.uint8,
        )
        return packed, scales.squeeze(-2)

    return quantize


def prepare_expert_matrix(packed, scales, *, gate_up, padded_hidden, padded_inter):
    """Bound CPU staging to one expert and allocate genuine device parameters."""
    device = packed.device
    bytes_parts, scale_parts = [], []
    for expert in range(packed.shape[0]):
        values = dequantize_checkpoint(
            packed[expert : expert + 1], scales[expert : expert + 1]
        )
        if gate_up:
            gate, up = values.chunk(2, dim=1)
            pad = (0, padded_hidden - values.shape[-1], 0, padded_inter - gate.shape[1])
            values = torch.cat((F.pad(gate, pad), F.pad(up, pad)), dim=1)
        else:
            values = F.pad(
                values,
                (
                    0,
                    padded_inter - values.shape[-1],
                    0,
                    padded_hidden - values.shape[1],
                ),
            )
        native_bytes, block_scales = _quantizer()(values.transpose(1, 2).numpy())
        bytes_parts.append(torch.from_numpy(np.array(native_bytes)).to(device))
        s = torch.from_numpy(np.array(block_scales)).to(device)
        # Whole eight-row scale tiles, flattened over E. Padded rows are zero.
        scale_parts.append(F.pad(s, (0, 0, 0, -s.shape[1] % 8)).flatten(0, 1))
    weight_bytes = torch.cat(bytes_parts, dim=0)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError("V4.1 fused EP requires native float4_e2m1fn_x2 tensors")
    native = torch.empty_like(weight_bytes, dtype=fp4_dtype)
    native.copy_(weight_bytes.view(fp4_dtype))
    return native, torch.cat(scale_parts, dim=0)
