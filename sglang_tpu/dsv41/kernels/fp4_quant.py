# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/layers/core/fused_moe_gmm.py
# Revision: bdb38700dd25959c029c7d77f00cee91ccd3a0bc
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
import jax.numpy as jnp


def quantize_to_native_fp4_kmajor(
    w: jax.Array, block: int, pack: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Quantize K-major float weight ([..., K, N]) to packed uint8 e2m1 and
    per-block FP32 scale.
    Returns uint8 to safely cross the PyTorch/JAX bridge before unpacking in gmm_v2.

    ``pack=False`` returns the same values as ``float4_e2m1fn`` [..., K, N]
    instead -- what the fused EP MoE kernel's FP4 form reads, and what torch
    receives as a native ``torch.float4_e2m1fn_x2`` tensor. Only the transport
    differs: the quantization above is one cast either way, so the two forms
    carry bit-identical weights and a caller may choose per layer.

    TODO: Make a more generic version of this function;
    possibly combine with quantize_tensor_to_fp4
    """

    size_k = w.shape[-2]
    num_blocks = size_k // block
    blocked = w.astype(jnp.float32).reshape(
        *w.shape[:-2], num_blocks, block, w.shape[-1]
    )
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    abs_max = jnp.max(jnp.abs(blocked), axis=-2, keepdims=True)
    scale = abs_max / fp4_max
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    quantized = jnp.clip(blocked * scale_inv, -fp4_max, fp4_max).astype(
        jnp.float4_e2m1fn
    )
    quantized = quantized.reshape(*w.shape[:-2], size_k, w.shape[-1])  # [..., K, N]

    scale = scale.astype(jnp.float32)  # [..., num_blocks, 1, N]
    if not pack:
        return quantized, scale

    # Real e2m1 bit-packing (2 values/byte along K), matching gmm_v2's own
    # should_unpack/bitcast(quant_dtype) unpacking convention.
    pairs = quantized.reshape(*w.shape[:-2], size_k // 2, 2, w.shape[-1])
    pairs = jnp.swapaxes(pairs, -1, -2)  # [..., K/2, N, 2] -- pair axis last
    packed = jax.lax.bitcast_convert_type(pairs, jnp.uint8)  # [..., K/2, N]

    return packed, scale
