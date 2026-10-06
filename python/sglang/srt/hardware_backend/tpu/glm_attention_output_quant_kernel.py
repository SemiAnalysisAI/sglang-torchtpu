"""GLM decode value expansion and dynamic FP8 quantization candidate.

The native output-projection dot, scaling, BF16 conversion and collective
remain outside. This module does not import JAX or initialize TPU on import.
Inputs are the actual BF16 attention output [1,4,512] and physical WV storage
[4,256,512], obtained by transposing existing logical WV [4,512,256].
"""

from threading import Lock

_TORCH_OPS = {}
_REGISTRATION_LOCK = Lock()


def pallas_wv_quant(attention, wv_physical, *, diagnostic=False, interpret=False):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if (
        attention.shape != (1, 4, 512)
        or wv_physical.shape != (4, 256, 512)
        or attention.dtype != jnp.bfloat16
        or wv_physical.dtype != jnp.bfloat16
    ):
        raise ValueError("WV quant requires BF16 attention[1,4,512] and WV[4,256,512]")

    def kernel(attention_ref, wv_ref, quantized_ref, scale_ref, *diagnostic_refs):
        # Match the native HLO's weight-left contraction. Preserve its BF16
        # output boundary before head flattening and activation quantization.
        heads = []
        for head in range(4):
            vector = jnp.pad(attention_ref[...][:, head, :], ((0, 7), (0, 0)))
            expanded = jax.lax.dot_general(
                wv_ref[...][head],
                vector,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )[:, :1].T
            heads.append(expanded.astype(jnp.bfloat16))
        values = jnp.stack(heads, axis=1).reshape(1, 1024)
        maximum = jnp.max(jnp.abs(values), axis=1, keepdims=True).astype(jnp.float32)
        scale = jnp.maximum(maximum / jnp.float32(448), jnp.float32(1e-12))
        quantized = (values.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn)
        quantized_ref[...] = jnp.pad(quantized, ((0, 15), (0, 0)))
        scale_ref[...] = jnp.pad(
            scale.reshape(1), (0, 15), constant_values=jnp.float32(1e-12)
        )
        if diagnostic:
            diagnostic_refs[0][...] = values

    outputs = [
        jax.ShapeDtypeStruct((16, 1024), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((16,), jnp.float32),
    ]
    output_specs = [
        pl.BlockSpec((16, 1024), lambda _: (0, 0)),
        pl.BlockSpec((16,), lambda _: (0,)),
    ]
    if diagnostic:
        outputs.append(jax.ShapeDtypeStruct((1, 1024), jnp.bfloat16))
        output_specs.append(pl.BlockSpec((1, 1024), lambda _: (0, 0)))
    return pl.pallas_call(
        kernel,
        out_shape=tuple(outputs),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=(
                pl.BlockSpec((1, 4, 512), lambda _: (0, 0, 0)),
                pl.BlockSpec((4, 256, 512), lambda _: (0, 0, 0)),
            ),
            out_specs=tuple(output_specs),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="glm_attention_output_quant" + ("_diagnostic" if diagnostic else ""),
    )(attention, wv_physical)


def register_wv_quant_op(*, diagnostic=False):
    """Register eagerly before tracing; normal mode has exactly two outputs."""
    if diagnostic in _TORCH_OPS:
        return _TORCH_OPS[diagnostic]
    with _REGISTRATION_LOCK:
        if diagnostic in _TORCH_OPS:
            return _TORCH_OPS[diagnostic]
        import jax
        from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        if diagnostic:
            def operation(
                attention: jax.Array, wv_physical: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return pallas_wv_quant(attention, wv_physical, diagnostic=True)
        else:
            def operation(
                attention: jax.Array, wv_physical: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                return pallas_wv_quant(attention, wv_physical)

        name = "sglang_tpu::glm_attention_output_quant"
        if diagnostic:
            name += "_diagnostic"
        _TORCH_OPS[diagnostic] = cpu_export_jax_op(name, operation)
    return _TORCH_OPS[diagnostic]
