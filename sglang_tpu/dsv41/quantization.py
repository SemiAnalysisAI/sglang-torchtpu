# SPDX-License-Identifier: Apache-2.0
"""Load-time MXFP4 block-32 to EPIC block-512 preparation.

Quantization uses the imported EPIC E2M1 cast, including ties to even.
The checkpoint loader retains SGLang names and contiguous expert ownership.
"""

from functools import cache, partial

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


@cache
def _checkpoint_quantizer(gate_up, padded_hidden, padded_inter):
    """Reference fallback: preserve Torch's first multiply for unusual scales.

    JAX CPU can flush subnormal inputs before multiplication, even where the
    Torch product is normal. Barriers alone cannot preserve that behavior.
    """

    def quantize(packed, scales):
        values = dequantize_checkpoint(
            torch.from_numpy(packed), torch.from_numpy(scales)
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
        return _quantizer()(values.transpose(1, 2).numpy())

    return quantize


@cache
def _direct_quantizer(gate_up, padded_hidden, padded_inter):
    """Re-quantize finite checkpoint scales using block32 maxima and code tables."""
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, backend="cpu")
    def quantize(packed, scales):
        # N-major checkpoint: one scale per 16 packed bytes / 32 codes.
        codes = packed.astype(jnp.uint8)
        target_n = 2 * padded_inter if gate_up else padded_hidden
        target_k = padded_hidden if gate_up else padded_inter

        def pad(x, target_last):
            if gate_up:
                gate, up = jnp.split(x, 2, axis=1)
                widths = (
                    (0, 0),
                    (0, padded_inter - gate.shape[1]),
                    (0, target_last - x.shape[2]),
                )
                return jnp.concatenate(
                    (jnp.pad(gate, widths), jnp.pad(up, widths)), axis=1
                )
            return jnp.pad(
                x,
                ((0, 0), (0, target_n - x.shape[1]), (0, target_last - x.shape[2])),
            )

        codes = pad(codes, target_k // 2)
        scales = pad(scales, target_k // 32)
        groups = target_k // 32
        codes = codes.reshape(codes.shape[0], target_n, groups, 16)
        lut = jnp.asarray(_E2M1, dtype=jnp.float32)
        # Magnitude code c&7 is monotone. Do NOT assume every group contains 6.
        max_code = jnp.max(jnp.maximum(codes & 7, (codes >> 4) & 7), axis=-1)
        maxima = jax.lax.optimization_barrier(lut[max_code] * jnp.abs(scales))
        absmax = maxima.reshape(*maxima.shape[:-1], groups // 16, 16).max(-1)
        block_scales = absmax / 6.0
        inv = jnp.where(block_scales == 0, 0.0, 1.0 / block_scales)
        inv = jnp.repeat(inv, 16, axis=-1)

        # Preserve Torch LUT*scale rounding, then JAX reciprocal multiplication.
        # The barriers prohibit reassociation into lut*(scale*inv), including
        # signed zero and tie-sensitive FP32 results. Cast implements RNE E2M1.
        values = jax.lax.optimization_barrier(lut * scales[..., None])
        normalized = jax.lax.optimization_barrier(values * inv[..., None])
        table = jnp.clip(normalized, -6.0, 6.0).astype(jnp.float4_e2m1fn)
        low = jnp.take_along_axis(table, codes & 15, axis=-1, mode="clip")
        high = jnp.take_along_axis(table, codes >> 4, axis=-1, mode="clip")
        values = jnp.stack((low, high), axis=-1)
        values = values.reshape(codes.shape[0], target_n, target_k)
        values = values.transpose(0, 2, 1)  # K-major; native packing is along N.
        native_bytes = jax.lax.bitcast_convert_type(
            values.reshape(values.shape[0], target_k, target_n // 2, 2),
            jnp.uint8,
        )
        return native_bytes, block_scales.transpose(0, 2, 1)

    return quantize


def prepare_expert_matrix(packed, scales, *, gate_up, padded_hidden, padded_inter):
    """Prepare native K-major FP4 and eight-row-tiled scales on the CPU.

    FP32 staging (including the exceptional dense fallback) is bounded to one
    expert. Only packed bytes and output scales have whole-matrix host buffers;
    each completed matrix is uploaded once, without per-expert device copies.
    """
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError("V4.1 fused EP requires native float4_e2m1fn_x2 tensors")
    experts, n, k_bytes = packed.shape
    target_n = 2 * padded_inter if gate_up else padded_hidden
    target_k = padded_hidden if gate_up else padded_inter
    if (
        padded_hidden % 512
        or padded_inter % 512
        or k_bytes % 16
        or (gate_up and n % 2)
        or n > target_n
        or 2 * k_bytes > target_k
        or scales.shape != (experts, n, k_bytes // 16)
    ):
        raise ValueError("Expected block32 checkpoint fitting G512-padded dimensions")
    quantize = _direct_quantizer(gate_up, padded_hidden, padded_inter)
    weight_host = np.empty((experts, target_k, target_n // 2), dtype=np.uint8)
    scale_rows = target_k // 512
    # Whole eight-row scale tiles, flattened over E; padded rows stay zero.
    scale_host = np.zeros(
        (experts, (scale_rows + 7) // 8 * 8, target_n), dtype=np.float32
    )
    for expert in range(experts):
        codes = packed[expert : expert + 1].detach().cpu().view(torch.uint8)
        scale = scales[expert : expert + 1].detach().cpu()
        # CPU JAX exp2 is not bit-identical to the checkpoint's Torch exp2.
        scale = (
            torch.exp2(scale.float() - 127)
            if scale.dtype == torch.uint8
            else scale.float()
        )
        magnitude = scale.abs()
        safe = bool(
            (
                torch.isfinite(magnitude * 6.0)
                & ((magnitude == 0) | (magnitude >= torch.finfo(torch.float32).tiny))
            ).all()
        )
        # Nonfinite/overflow scales invalidate monotone maxima. Subnormal input
        # scales need Torch's first multiply, not JAX CPU's input-flush semantics.
        convert = (
            quantize
            if safe
            else _checkpoint_quantizer(gate_up, padded_hidden, padded_inter)
        )
        native_bytes, block_scales = convert(codes.numpy(), scale.numpy())
        # Synchronize/copy each CPU result before reusing bounded expert staging.
        weight_host[expert] = np.asarray(native_bytes)[0]
        scale_host[expert, :scale_rows] = np.asarray(block_scales)[0]
        del native_bytes, block_scales
    # Upload whole matrices, never 48 independent expert buffers. Keep the
    # production byte->native device copy contract (no FP4 host arithmetic).
    weight_bytes = torch.from_numpy(weight_host).to(packed.device)
    native = torch.empty_like(weight_bytes, dtype=fp4_dtype)
    native.copy_(weight_bytes.view(fp4_dtype))
    return native, torch.from_numpy(scale_host.reshape(-1, target_n)).to(packed.device)
