# SPDX-License-Identifier: Apache-2.0
"""TPU dispatch preparation for the qualified GLM batch-one FP8 MoE path.

All eight routed rows contain the same token. Stable pairwise ranks therefore
replace both integer sorts without changing the existing grouped matmuls.
The existing padded native combine retains its original rounding behavior.
"""


def supports_bs1_routing(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    *,
    topk,
    activation_name,
    no_combine,
    global_num_experts,
    expert_group_offset,
    use_sc_gather,
    w_quant_bits,
    permute_dense_threshold,
    skip_pad_experts,
):
    """Use only the configuration qualified against the existing FP32 combine.

    The caller's skip-padding contract maps negative router IDs to the global
    expert count (256). Other precisions, shapes and combine modes retain the
    existing path, including its original rounding behavior.
    """
    import jax.numpy as jnp

    return (
        hidden_states.shape == (1, 6144)
        and hidden_states.dtype == jnp.bfloat16
        and w1.shape == (16, 6144, 4096)
        and w2.shape == (16, 2048, 6144)
        and w1.dtype == w2.dtype == jnp.float8_e4m3fn
        and topk_weights.shape == topk_ids.shape == (1, 8)
        and topk_weights.dtype == jnp.bfloat16
        and topk_ids.dtype == jnp.int32
        and w1_scale is not None
        and w2_scale is not None
        and w1_scale.shape == (16, 1, 1, 4096)
        and w2_scale.shape == (16, 1, 1, 6144)
        and w1_scale.dtype == w2_scale.dtype == jnp.bfloat16
        and w1_bias is None
        and w2_bias is None
        and topk == 8
        and activation_name == "silu"
        and not no_combine
        and global_num_experts == 256
        and expert_group_offset in range(0, 256, 16)
        and use_sc_gather == 1
        and w_quant_bits == 0
        and permute_dense_threshold == 3
        and skip_pad_experts
    )


def dispatch_bs1(hidden_states, topk_ids, *, interpret=False):
    """Return padded routed activations, expert counts and inverse ranks."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    hidden = hidden_states.shape[1]

    def kernel(x_ref, ids_ref, xs_ref, groups_ref, inverse_ref):
        expert = ids_ref[...].reshape(8)
        slots = jnp.arange(8, dtype=jnp.int32)
        earlier = (expert[None, :] < expert[:, None]) | (
            (expert[None, :] == expert[:, None]) & (slots[None, :] < slots[:, None])
        )
        inverse = jnp.sum(earlier.astype(jnp.int32), axis=1)
        groups = jnp.sum(
            (expert[:, None] == jnp.arange(256)[None, :]).astype(jnp.int32), axis=0
        )
        # High sentinel padding sorts last and does not belong to any group.
        xs_ref[...] = jnp.where(jnp.arange(16)[:, None] < 8, x_ref[...], 0)
        groups_ref[...] = groups
        inverse_ref[...] = inverse

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((16, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((256,), jnp.int32),
            jax.ShapeDtypeStruct((8,), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(hidden_states.shape, lambda i: (0, 0)),
                pl.BlockSpec(topk_ids.shape, lambda i: (0, 0)),
            ),
            out_specs=(
                pl.BlockSpec((16, hidden), lambda i: (0, 0)),
                pl.BlockSpec((256,), lambda i: (0,)),
                pl.BlockSpec((8,), lambda i: (0,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="glm_bs1_moe_dispatch",
    )(hidden_states, topk_ids)
