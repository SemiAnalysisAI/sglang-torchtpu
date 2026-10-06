# SPDX-License-Identifier: Apache-2.0
"""Opt-in BS1 GLM FP8 MLA front fusion with native numerical boundaries.

Called only by the guarded TPU dispatch. QKV-A and QB keep dynamic per-row
activation scaling, FP32 dot/scaling and BF16 output rounding. Native RMSNorm,
interleaved rotary multiply rounding and absorbed-Q BMM are preserved. Cache
writes, residual/input RMSNorm and attention backend padding remain outside.

Imports and custom-op registration are lazy. The kernel accepts transposed A
and either KC orientation. The Torch wrapper keeps A's conversion inside the
compiled call so any required data reformat is included in model execution.
"""

from threading import Lock

from sglang.srt.environ import envs

_FRONT_OPS = {}
_FRONT_OP_LOCK = Lock()


def _pallas_attention_front(
    x,
    qkv_weight,
    qkv_scale,
    q_norm_weight,
    kv_norm_weight,
    qb_weight,
    qb_scale,
    w_kc,
    positions,
    cos_sin_cache,
    *,
    q_eps=1e-5,
    kv_eps=1e-5,
    neox=False,
    absorb=True,
    interpret=False,
    kc_logical=False,
    emit_q_lora=False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    expected = {
        "x": ((1, 6144), x),
        "qkv_weight": ((2624, 6144), qkv_weight),
        "q_norm_weight": ((2048,), q_norm_weight),
        "kv_norm_weight": ((512,), kv_norm_weight),
        "qb_weight": ((2048, 1024), qb_weight),
        # The opt-in logical view matches native BMM's K,N orientation and
        # avoids the existing custom-op boundary's KC formatting copy.
        "w_kc": ((4, 192, 512) if kc_logical else (4, 512, 192), w_kc),
        "positions": ((1,), positions),
    }
    for name, (shape, value) in expected.items():
        if value.shape != shape:
            raise ValueError(f"GLM front requires {name}{shape}; got {value.shape}")
    if x.dtype != jnp.bfloat16 or w_kc.dtype != jnp.bfloat16:
        raise ValueError("GLM front requires BF16 hidden states and absorbed weight")
    if qkv_weight.dtype != jnp.float8_e4m3fn or qb_weight.dtype != jnp.float8_e4m3fn:
        raise ValueError("GLM front requires e4m3fn projection weights")
    if positions.dtype != jnp.int32:
        raise ValueError("Positions must be int32 for the scalar-prefetch path")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64:
        raise ValueError("Expected existing rotary cache [max_position,64]")
    if cos_sin_cache.dtype != jnp.bfloat16:
        raise ValueError("Expected BF16 rotary cache (the production query dtype)")
    if qkv_scale.size not in (1, 2624) or qb_scale.size not in (1, 1024):
        raise ValueError("Projection scales must be scalar or per-output-channel")

    def kernel(
        position_ref,
        x_ref,
        a_ref,
        a_scale_ref,
        q_norm_ref,
        kv_norm_ref,
        b_ref,
        b_scale_ref,
        kc_ref,
        cs_ref,
        q_latent_out,
        q_rope_out,
        kv_latent_out,
        k_rope_out,
        *q_lora_out,
    ):
        def fp8_linear(
            value, weight, scale, transposed_weight=False, output_width=None
        ):
            if output_width is None:
                output_width = weight.shape[0 if transposed_weight else 1]
            # Keep the tested 256-column alignment for FP8 dots, including
            # QKV-A's output tail.
            # A's input BlockSpec directly pads its outer N dimension. Each
            # output column is independent, so discarding OOB columns before
            # scaling/norm also excludes any unspecified DMA padding values.
            value_f32 = value.astype(jnp.float32)
            maximum = jnp.max(jnp.abs(value_f32), axis=-1, keepdims=True)
            row_scale = jnp.maximum(maximum / 448.0, jnp.float32(1e-12))
            quantized = (value_f32 / row_scale).astype(jnp.float8_e4m3fn)
            # An internal MXU-aligned tile. Additional zero rows do not alter
            # the real row's dynamic quantization or any emitted result.
            quantized = jnp.pad(quantized, ((0, 31), (0, 0)))
            raw = jax.lax.dot_general(
                quantized,
                weight,
                (((1,), (1 if transposed_weight else 0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            # Preserve production's two sequential FP32 multiplies, then its
            # BF16 result boundary before norm/split/absorbed BMM.
            result = raw[:1, :output_width] * row_scale
            # Both the original [N,1] input and its opt-in flat view use the
            # same FP32 scale values and multiplication boundary here.
            result = result * scale.reshape(-1).astype(jnp.float32)[None, :]
            return result.astype(jnp.bfloat16)

        def rms(value, weight, epsilon):
            value_f32 = value.astype(jnp.float32)
            # Match the native producer's FP32 reduction order: QA has
            # 1024-element tiles, reduced across tiles, sublanes, then lanes.
            # KV has 512 active elements; zero-padding supplies the same
            # active sublanes without changing its 512-element divisor.
            # A generic mean/flatten changes rare BF16 rounding boundaries.
            squares = (value_f32 * value_f32).reshape(-1)
            if squares.shape[0] < 1024:
                squares = jnp.pad(squares, (0, 1024 - squares.shape[0]))
            physical = squares.reshape(-1, 8, 128)
            tiles = jnp.sum(physical, axis=0)
            sublanes = jnp.sum(tiles, axis=0)
            total = jnp.sum(sublanes, axis=0)
            variance = (total * jnp.float32(1.0 / value.shape[-1])).reshape(1, 1)
            normalized = value_f32 * jax.lax.rsqrt(variance + epsilon)
            return (normalized * weight.astype(jnp.float32)).astype(jnp.bfloat16)

        qkv = fp8_linear(
            x_ref[...],
            a_ref[...],
            a_scale_ref[...],
            transposed_weight=True,
            output_width=2624,
        )
        qa = rms(qkv[:, :2048], q_norm_ref[...], q_eps)
        kv = rms(qkv[:, 2048:2560], kv_norm_ref[...], kv_eps)
        k_rope = qkv[:, 2560:2624].reshape(1, 1, 64)
        q = fp8_linear(qa, b_ref[...], b_scale_ref[...]).reshape(1, 4, 256)
        q_nope, q_rope = q[:, :, :192], q[:, :, 192:]

        if absorb:
            # Four unrolled rank-2 MXU dots avoid requiring a Pallas batch-dot
            # lowering. Padding K to256 is exact because both tails are zero.
            absorbed_heads = []
            for head in range(4):
                lhs = jnp.pad(q_nope[:, head, :], ((0, 7), (0, 64)))
                rhs = jnp.pad(
                    kc_ref[head, :, :],
                    ((0, 64), (0, 0)) if kc_logical else ((0, 0), (0, 64)),
                )
                raw = jax.lax.dot_general(
                    lhs,
                    rhs,
                    (((1,), (0 if kc_logical else 1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                absorbed_heads.append(raw[:1].astype(jnp.bfloat16))
            q_latent = jnp.stack(absorbed_heads, axis=1)
        else:
            q_latent = q_nope

        # Cache DMA has eight rows to meet TPU block alignment. Select the
        # requested row in VMEM; the full production cache never enters VMEM.
        cache_rows = cs_ref[...]
        selected_row = jnp.arange(8)[:, None] == position_ref[0] % 8
        cs = jnp.sum(
            jnp.where(selected_row, cache_rows.astype(jnp.float32), 0.0),
            axis=0,
            keepdims=True,
        ).astype(jnp.bfloat16)
        cos, sin = jnp.split(cs, 2, axis=-1)

        def rotate(value):
            cos_bf16 = cos[:, None, :].astype(jnp.bfloat16)
            sin_bf16 = sin[:, None, :].astype(jnp.bfloat16)
            if neox:
                first, second = jnp.split(value, 2, axis=-1)
            else:
                pairs = value.reshape((*value.shape[:-1], 32, 2))
                first, second = pairs[..., 0], pairs[..., 1]
            # Native Torch executes BF16 multiply, then BF16 add/sub. A fused
            # FP32 multiply-add would change the model's rounding contract.
            first_cos = (first * cos_bf16).astype(jnp.bfloat16)
            second_sin = (second * sin_bf16).astype(jnp.bfloat16)
            second_cos = (second * cos_bf16).astype(jnp.bfloat16)
            first_sin = (first * sin_bf16).astype(jnp.bfloat16)
            out_first = (first_cos - second_sin).astype(jnp.bfloat16)
            out_second = (second_cos + first_sin).astype(jnp.bfloat16)
            if neox:
                return jnp.concatenate((out_first, out_second), axis=-1)
            return jnp.stack((out_first, out_second), axis=-1).reshape(value.shape)

        q_latent_out[...] = q_latent
        q_rope_out[...] = rotate(q_rope)
        kv_latent_out[...] = kv.reshape(1, 1, 512)
        k_rope_out[...] = rotate(k_rope)
        if emit_q_lora:
            q_lora_out[0][...] = qa

    def full_spec(value):
        return pl.BlockSpec(value.shape, lambda i, pos: (0,) * value.ndim)

    inputs = (
        x,
        qkv_weight,
        qkv_scale,
        q_norm_weight,
        kv_norm_weight,
        qb_weight,
        qb_scale,
        w_kc,
    )
    output_shapes = (
        (1, 4, 512 if absorb else 192),
        (1, 4, 64),
        (1, 1, 512),
        (1, 1, 64),
    ) + (((1, 2048),) if emit_q_lora else ())
    return pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in output_shapes
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(
                *(
                    pl.BlockSpec((2816, 6144), lambda i, pos: (0, 0))
                    if index == 1
                    else full_spec(value)
                    for index, value in enumerate(inputs)
                ),
                pl.BlockSpec((8, 64), lambda i, pos: (pos[0] // 8, 0)),
            ),
            out_specs=tuple(
                pl.BlockSpec(shape, lambda i, pos, shape=shape: (0,) * len(shape))
                for shape in output_shapes
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=48 * 1024 * 1024,
        ),
        interpret=interpret,
        name=(
            f"glm_attention_front_kc_kn_absorb{int(absorb)}"
            if kc_logical
            else f"glm_attention_front_absorb{int(absorb)}"
        )
        + ("_qlora" if emit_q_lora else ""),
    )(positions, *inputs, cos_sin_cache)


def _get_front_op(emit_q_lora=False):
    if emit_q_lora in _FRONT_OPS:
        return _FRONT_OPS[emit_q_lora]
    with _FRONT_OP_LOCK:
        if emit_q_lora in _FRONT_OPS:
            return _FRONT_OPS[emit_q_lora]
        import jax

        from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        def front(
            x: jax.Array,
            qkv_weight: jax.Array,
            qkv_scale: jax.Array,
            q_norm_weight: jax.Array,
            kv_norm_weight: jax.Array,
            qb_weight: jax.Array,
            qb_scale: jax.Array,
            kc_transposed: jax.Array,
            positions: jax.Array,
            cos_sin_cache: jax.Array,
            q_eps: float,
            kv_eps: float,
            kc_logical: bool = False,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return _pallas_attention_front(
                x,
                qkv_weight,
                qkv_scale,
                q_norm_weight,
                kv_norm_weight,
                qb_weight,
                qb_scale,
                kc_transposed,
                positions,
                cos_sin_cache,
                q_eps=q_eps,
                kv_eps=kv_eps,
                kc_logical=kc_logical,
                emit_q_lora=emit_q_lora,
            )

        # A separate op name keeps the 4-output op's exported program unchanged.
        op = cpu_export_jax_op(
            "sglang_tpu::glm_fp8_mla_front" + ("_q_lora" if emit_q_lora else ""),
            front,
        )

        def fake_front(
            x,
            qkv_weight,
            qkv_scale,
            q_norm_weight,
            kv_norm_weight,
            qb_weight,
            qb_scale,
            kc_transposed,
            positions,
            cos_sin_cache,
            q_eps,
            kv_eps,
            kc_logical=False,
        ):
            shapes = ((1, 4, 512), (1, 4, 64), (1, 1, 512), (1, 1, 64))
            if emit_q_lora:
                shapes += ((1, 2048),)
            return tuple(x.new_empty(shape) for shape in shapes)

        # The dispatch fixes these shapes. Avoid re-exporting the entire
        # Pallas program when Dynamo only needs its output metadata.
        op.register_fake(fake_front)
        _FRONT_OPS[emit_q_lora] = op
        return op


def register_glm_fp8_mla_front_op(emit_q_lora=False):
    """Register from the opt-in TPU constructor before Dynamo tracing starts."""
    _get_front_op(emit_q_lora)


def glm_fp8_mla_front(
    x,
    qkv_a_weight,
    qkv_a_scale,
    q_norm_weight,
    kv_norm_weight,
    qb_weight,
    qb_scale,
    kc_transposed,
    positions,
    cos_sin_cache,
    q_eps,
    kv_eps,
    *,
    kc_logical=False,
    emit_q_lora=False,
):
    """Return Qlatent, Qrope, KVlatent and Krope in native MLA layouts.

    With ``emit_q_lora`` a fifth output is the normalized Q-A latent [1, 2048],
    which the DSA indexer consumes.

    A uses its prepared Torch[K,N] shape. KC uses [heads,latent,nope] by
    default, or the existing logical [heads,nope,latent] view when enabled.
    This entry point is restricted by the dispatch guard to the verified
    GLM FP8 BS1 contract; no KC contiguity conversion is introduced here.
    """
    import torch

    if envs.SGLANG_TPU_GLM_FP8_MLA_FLAT_SCALES.get():
        # The existing [N, 1] tensors have N-contiguous physical storage.
        # A flat view lets Pallas read that storage directly; its arithmetic
        # already reshapes both scales to vectors before multiplying.
        qkv_a_scale = qkv_a_scale.reshape(-1)
        qb_scale = qb_scale.reshape(-1)

    return _get_front_op(emit_q_lora)(
        x,
        qkv_a_weight.t().contiguous(),
        qkv_a_scale,
        q_norm_weight,
        kv_norm_weight,
        qb_weight,
        qb_scale,
        kc_transposed,
        positions.reshape(-1).to(torch.int32),
        cos_sin_cache,
        q_eps,
        kv_eps,
        kc_logical,
    )
