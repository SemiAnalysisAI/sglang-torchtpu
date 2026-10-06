# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM FP8 expert-pair fusion on the qualified TPU geometry.

The caller retains its dispatch and native weighted combine. Arithmetic and
DMA ordering below are frozen from the qualified lookahead/flat-scale experiment.
Compact scale matrices stay in VMEM; row selection preserves raw BF16 bits. The
private diagnostic arguments remain to preserve that function verbatim;
the production entry point fixes the qualified schedule and output contract.
All device-library imports are lazy, so importing this module is CPU-safe.
"""


def supports_glm_fused_experts(bs1_routing, *, fp32_scales, tpu_info=None):
    """Extend the existing shape/dtype guard with arithmetic/hardware limits."""
    if not bs1_routing or fp32_scales:
        return False
    if tpu_info is None:
        from jax.experimental.pallas import tpu as pltpu

        tpu_info = pltpu.get_tpu_info()
    return (
        getattr(getattr(tpu_info, "chip_version", None), "value", None) == "7x"
        and getattr(tpu_info, "mxu_column_size", 0) == 256
        and getattr(tpu_info, "num_lanes", 0) == 128
        and getattr(tpu_info, "fp8_ops_per_second", 0) > 0
        and getattr(tpu_info, "vmem_capacity_bytes", 0) >= 67108864
    )


def fused_sorted_experts(x_sorted, w1, w2, group_sizes, s1, s2, *, offset):
    """Return sorted BF16 expert outputs; dispatch/combine remain with caller."""
    return _fused_sorted_experts(
        x_sorted, w1, w2, group_sizes, s1, s2,
        offset=offset, prefetch_down=True, lookahead_a=True,
    )


def select_scale_row(scales, expert_mask):
    """Select one row without arithmetic on BF16 payloads, including NaNs."""
    import jax
    import jax.numpy as jnp

    bits = jax.lax.bitcast_convert_type(scales, jnp.uint16).astype(jnp.int32)
    selected = jnp.sum(
        jnp.where(expert_mask, bits, jnp.int32(0)),
        axis=0, keepdims=True, dtype=jnp.int32,
    )
    return jax.lax.bitcast_convert_type(selected.astype(jnp.uint16), jnp.bfloat16)


def _fused_sorted_experts(
    x_sorted, w1, w2, group_sizes, s1, s2, *, offset, prefetch_down=True,
    lookahead_a=False, capture_intermediate=False, interpret=False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang.srt.layers.moe.pallas.gmm_v2 import (
        _scale_quantized_partial,
        apply_act_fn,
    )

    if x_sorted.shape != (16, 6144) or x_sorted.dtype != jnp.bfloat16:
        raise ValueError("Only GLM BF16 padded single-token inputs are supported")
    if w1.shape != (16, 6144, 4096) or w2.shape != (16, 2048, 6144):
        raise ValueError("Only GLM EP16 full-width experts are supported")
    if w1.dtype != jnp.float8_e4m3fn or w2.dtype != w1.dtype:
        raise ValueError("Weights must be FP8 e4m3fn")
    if s1.shape != (16, 1, 1, 4096) or s2.shape != (16, 1, 1, 6144):
        raise ValueError("Only channelwise scales are supported")
    if s1.dtype != jnp.bfloat16 or s2.dtype != s1.dtype:
        raise ValueError("Scales must preserve the production BF16 boundary")
    if group_sizes.shape != (256,) or group_sizes.dtype != jnp.int32:
        raise ValueError("Expected 256 global expert counts")
    if offset not in range(0, 256, 16):
        raise ValueError("Expert shard offset must match EP16")

    def projection(values, packed_weights, scales, *, width, column_starts):
        """Same inner arithmetic as gmm_v2, retaining every BF16 boundary.

        Columns are independent. Gate/up column order matches each baseline
        1024-column fused tile: gate0, up0, gate1, up1.
        """
        columns = []
        for start_n in column_starts:
            rhs = pltpu.bitcast(
                packed_weights[:, pl.ds(start_n, 256)], jnp.float8_e4m3fn
            )
            acc = jnp.zeros((16, 256), dtype=jnp.bfloat16)
            rhs_scale = scales[:, start_n:start_n + 256]
            for start_k in range(0, width, 512):
                block_lhs = values[:, start_k:start_k + 512]
                block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
                block_scale = block_abs_max / 448.0
                block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
                block_lhs_q = (block_lhs * block_scale_inv).astype(jnp.float8_e4m3fn)
                block_acc = jnp.matmul(
                    block_lhs_q,
                    rhs[start_k:start_k + 512],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                acc += _scale_quantized_partial(
                    block_acc, block_scale, rhs_scale, scales_dtype=jnp.bfloat16
                )
            columns.append(acc)
        return jnp.concatenate(columns, axis=1)

    def kernel(
        counts_ref, x_ref, w1_ref, w2_ref, s1_ref, s2_ref,
        out_ref, *rest,
    ):
        if capture_intermediate:
            middle_out_ref, *rest = rest
            middle_out_ref[...] = jnp.zeros((16, 2048), jnp.bfloat16)
        packed_a, packed_b, middle, sems = rest
        source_a = w1_ref.bitcast(jnp.uint32)
        source_b = w2_ref.bitcast(jnp.uint32)
        out_ref[...] = jnp.zeros((16, 6144), jnp.bfloat16)
        rows = jnp.arange(16, dtype=jnp.int32)[:, None]
        row_start = jax.lax.fori_loop(
            0, offset, lambda i, total: total + counts_ref[i], jnp.int32(0)
        )

        def expert_step(expert, state):
            row_start, a_inflight = state
            count = counts_ref[offset + expert]
            if lookahead_a:
                next_expert = jax.lax.fori_loop(
                    expert + 1, 16,
                    lambda i, found: jnp.where(
                        (found == 16) & (counts_ref[offset + i] > 0), i, found
                    ),
                    jnp.int32(16),
                )
            else:
                next_expert = jnp.int32(16)

            @pl.when(count > 0)
            def run_expert():
                copy_a = pltpu.make_async_copy(source_a.at[expert], packed_a, sems.at[0])
                copy_b = pltpu.make_async_copy(source_b.at[expert], packed_b, sems.at[1])
                if lookahead_a:
                    @pl.when(~a_inflight)
                    def start_first_a():
                        copy_a.start()
                else:
                    copy_a.start()
                if prefetch_down:
                    copy_b.start()
                copy_a.wait()

                values = x_ref[...]
                expert_mask = jnp.arange(16, dtype=jnp.int32)[:, None] == expert
                scale_a = select_scale_row(s1_ref[...], expert_mask)
                scale_b = select_scale_row(s2_ref[...], expert_mask)
                activations = []
                for tile in range(2):
                    starts = tuple(range(tile * 1024, (tile + 1) * 1024, 256))
                    starts += tuple(value + 2048 for value in starts)
                    gate_up = projection(
                        values, packed_a, scale_a, width=6144, column_starts=starts
                    )
                    activations.append(apply_act_fn(gate_up, 1024, "silu"))
                # This store explicitly retains the original GMM1 BF16 output.
                middle[...] = jnp.concatenate(activations, axis=1).astype(jnp.bfloat16)
                if lookahead_a:
                    # All current A reads precede the explicit middle store.
                    # The buffer may now be overwritten while B uses middle.
                    @pl.when(next_expert < 16)
                    def start_next_a():
                        pltpu.make_async_copy(
                            source_a.at[next_expert], packed_a, sems.at[0]
                        ).start()
                if not prefetch_down:
                    copy_b.start()
                copy_b.wait()
                down = projection(
                    middle[...], packed_b, scale_b, width=2048,
                    column_starts=tuple(range(0, 6144, 256)),
                )
                mask = (rows >= row_start) & (rows < row_start + count)
                # Native GMM assembles each group into a shared 16-row tile by
                # adding a masked output to prior groups. Keep that boundary.
                out_ref[...] = out_ref[...] + jnp.where(mask, down, 0)
                if capture_intermediate:
                    middle_out_ref[...] = middle_out_ref[...] + jnp.where(mask, middle[...], 0)

            next_inflight = jnp.where(count > 0, next_expert < 16, a_inflight)
            return row_start + count, next_inflight

        jax.lax.fori_loop(0, 16, expert_step, (row_start, jnp.bool_(False)))

    def vector_spec(shape):
        return pl.BlockSpec(shape, lambda i, counts: (0, 0))

    out_shapes = [jax.ShapeDtypeStruct((16, 6144), jnp.bfloat16)]
    out_specs = [vector_spec((16, 6144))]
    if capture_intermediate:
        out_shapes.append(jax.ShapeDtypeStruct((16, 2048), jnp.bfloat16))
        out_specs.append(vector_spec((16, 2048)))
    result = pl.pallas_call(
        kernel,
        out_shape=tuple(out_shapes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(
                vector_spec((16, 6144)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                vector_spec((16, 4096)),
                vector_spec((16, 6144)),
            ),
            out_specs=tuple(out_specs),
            scratch_shapes=(
                pltpu.VMEM((1536, 4096), jnp.uint32),
                pltpu.VMEM((512, 6144), jnp.uint32),
                pltpu.VMEM((16, 2048), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=60397977, dimension_semantics=("arbitrary",)
        ),
        interpret=interpret,
        name=("glm_bs1_fused_experts_lookahead" if lookahead_a else
              "glm_bs1_fused_experts_prefetch" if prefetch_down else
              "glm_bs1_fused_experts_serial"),
    )(group_sizes, x_sorted, w1, w2, s1.reshape(16, 4096), s2.reshape(16, 6144))
    return result if capture_intermediate else result[0]
