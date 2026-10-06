# SPDX-License-Identifier: Apache-2.0
"""Opt-in BS1 GLM score gather and normalization after native expert top-k.

Sigmoid, bias addition, and top-k remain native: Pallas sigmoid and argmax
were not numerically equivalent, including native tie ordering. This kernel
only gathers eight original FP32 scores and normalizes their FP32 sum.
"""

from threading import Lock

import torch

_GATHER_OP = None
_GATHER_OP_LOCK = Lock()


def can_use_glm_routing_gather(
    scores,
    correction_bias,
    *,
    topk,
    renormalize,
    num_expert_group,
    topk_group,
    num_fused_shared_experts,
    apply_routed_scaling_factor_on_output,
):
    """Metadata-only guard for the qualified FP32, one-group, top-eight path."""
    return (
        scores.device.type == "tpu"
        and scores.shape == (1, 256)
        and scores.dtype == torch.float32
        and correction_bias.dtype == torch.float32
        and correction_bias.numel() == 256
        and topk == 8
        and renormalize
        and num_expert_group == 1
        and topk_group == 1
        and num_fused_shared_experts == 0
        and not apply_routed_scaling_factor_on_output
    )


def _pallas_routing_gather(scores, ids, *, interpret=False):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    def kernel(scores_ref, ids_ref, out_ref):
        source = scores_ref[...].reshape(1, 256)
        selected = ids_ref[...].reshape(8, 1)
        columns = jnp.arange(256, dtype=jnp.int32).reshape(1, 256)
        gathered = jnp.sum(jnp.where(columns == selected, source, 0.0), axis=-1)
        gathered = gathered.reshape(1, 8)
        denominator = jnp.sum(gathered, axis=-1, keepdims=True) + 1e-20
        out_ref[...] = gathered / denominator

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, 8), jnp.float32),
        grid=(),
        interpret=interpret,
        name="glm_routing_gather_normalize",
    )(scores, ids)


def _get_gather_op():
    global _GATHER_OP
    if _GATHER_OP is not None:
        return _GATHER_OP
    with _GATHER_OP_LOCK:
        if _GATHER_OP is not None:
            return _GATHER_OP
        import jax

        from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        def gather(scores: jax.Array, ids: jax.Array) -> jax.Array:
            return _pallas_routing_gather(scores, ids)

        op = cpu_export_jax_op("sglang_tpu::glm_routing_gather_normalize", gather)
        op.register_fake(lambda scores, ids: scores.new_empty((1, 8)))
        _GATHER_OP = op
        return _GATHER_OP


def register_glm_routing_gather_op():
    """Register from the opt-in TopK constructor before model tracing."""
    _get_gather_op()


def glm_routing_gather_normalize(scores, ids):
    return _get_gather_op()(scores, ids.to(torch.int32))
