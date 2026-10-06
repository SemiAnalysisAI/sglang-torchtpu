"""GLM BS1 RMS epilogue plus shared projection input quantization.

The native residual sum, variance and reciprocal square root stay outside.
Return BF16 hidden[1,6144], FP8 input[16,6144] and flat FP32 scales[16].
The native projection consumer reshapes the scales to [16,1]. Keeping the
custom-op output flat avoids a singleton reduction introduced by TPU layout
conversion. Module import does not load JAX or initialize a TPU.
"""

from threading import Lock

_TORCH_OP = None
_REGISTRATION_LOCK = Lock()


def _quantize_and_pad(normalized):
    import jax.numpy as jnp

    maximum = jnp.max(jnp.abs(normalized), axis=1, keepdims=True).astype(jnp.float32)
    scale = jnp.maximum(maximum / jnp.float32(448), jnp.float32(1e-12))
    quantized = (normalized.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn)
    return (
        jnp.pad(quantized, ((0, 15), (0, 0))),
        jnp.pad(scale, ((0, 15), (0, 0)), constant_values=jnp.float32(1e-12)),
    )


def pallas_epilogue_quant(z, inverse_rms, weight, *, interpret=False):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if (z.shape != (1, 6144) or inverse_rms.shape != (1, 1)
            or weight.shape != (6144,)
            or any(value.dtype != jnp.float32 for value in (z, inverse_rms, weight))):
        raise ValueError("epilogue artifact requires F32 z[1,6144], inv[1,1], w[6144]")

    def kernel(z_ref, inverse_ref, weight_ref, normalized_ref, q_ref, scale_ref):
        # Two FP32 multiplies followed by BF16 rounding, before quantization.
        scaled = z_ref[...] * inverse_ref[...]
        normalized = (scaled * weight_ref[...][None, :]).astype(jnp.bfloat16)
        quantized, scale = _quantize_and_pad(normalized)
        normalized_ref[...] = normalized
        q_ref[...] = quantized
        scale_ref[...] = scale.reshape(16)

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16),
            jax.ShapeDtypeStruct((16, 6144), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((16,), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[
                pl.BlockSpec((1, 6144), lambda _: (0, 0)),
                pl.BlockSpec((1, 1), lambda _: (0, 0)),
                pl.BlockSpec((6144,), lambda _: (0,)),
            ],
            out_specs=(
                pl.BlockSpec((1, 6144), lambda _: (0, 0)),
                pl.BlockSpec((16, 6144), lambda _: (0, 0)),
                pl.BlockSpec((16,), lambda _: (0,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="glm_norm_shared_quant_flat_scale",
    )(z, inverse_rms, weight)


def register_epilogue_quant_op():
    """Register before tracing; no JAX import occurs on module import."""
    global _TORCH_OP
    if _TORCH_OP is not None:
        return _TORCH_OP
    with _REGISTRATION_LOCK:
        if _TORCH_OP is not None:
            return _TORCH_OP
        import jax
        from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        def epilogue(
            z: jax.Array, inverse_rms: jax.Array, weight: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return pallas_epilogue_quant(z, inverse_rms, weight)

        _TORCH_OP = cpu_export_jax_op(
            "sglang_tpu::glm_norm_shared_quant_flat_scale", epilogue
        )
    return _TORCH_OP
