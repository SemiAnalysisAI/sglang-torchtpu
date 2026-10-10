# SPDX-License-Identifier: Apache-2.0
"""Pallas per-head sigmoid-gated RMSNorm for TPU (the Kimi-K3 KDA o_norm).

``FusedRMSNormGated(activation="sigmoid").forward_native`` in one kernel:
``rmsnorm(x) * weight * sigmoid(g)`` per head in fp32, written as the flat
``[T, H * D]`` o_proj input in x's dtype. Mosaic and XLA reduce and evaluate
rsqrt/sigmoid differently, so a bf16 rounding can differ by one ulp. g is one
column block of its operand, so callers pass the fused ``[q, k, v, g]``
projection output whole (a split view is copied at the op boundary); the weight
is the norm parameter, upcast in the kernel. The batch is one block
(``grid=(1,)``); callers bound the row count.
"""

# PEP 563 must remain disabled so `torch_tpu._internal.pallas` sees `jax.Array`
# type objects rather than stringified annotations at registration time.

import functools
from threading import Lock
from typing import Callable

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl

_OP_NAME = "sglang_tpu_pallas::gated_rms_norm_sigmoid"


def _kernel(x_ref, g_ref, w_ref, o_ref, *, eps: float):
    num_heads, head_dim = x_ref.shape[1], x_ref.shape[2]
    w = w_ref[...].astype(jnp.float32)
    for h in range(num_heads):
        cols = slice(h * head_dim, (h + 1) * head_dim)
        x = x_ref[:, h, :].astype(jnp.float32)
        g = g_ref[:, cols].astype(jnp.float32)
        y = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w
        o_ref[:, cols] = (y * jax.nn.sigmoid(g)).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "g_block"))
def gated_rms_norm_jax(
    x: jax.Array, g: jax.Array, weight: jax.Array, eps: float, g_block: int
) -> jax.Array:
    """x ``[T, H, D]``, weight ``[D]`` -> ``[T, H * D]`` in x's dtype. The gate
    is columns ``[g_block * H * D, (g_block + 1) * H * D)`` of ``g``: a block
    index, so a gate offset that is not a whole number of gate widths cannot be
    expressed."""
    rows, num_heads, head_dim = x.shape
    width = num_heads * head_dim
    assert head_dim % 128 == 0, f"head_dim {head_dim} is not lane aligned"
    assert weight.shape == (head_dim,), f"weight {weight.shape} vs x {x.shape}"
    assert g.ndim == 2 and g.shape[0] == rows, f"g {g.shape} vs x {x.shape}"
    assert 0 <= g_block and (g_block + 1) * width <= g.shape[1], (
        f"gate block {g_block} of width {width} is outside g {g.shape}"
    )
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct((rows, width), x.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec(x.shape, lambda i: (0, 0, 0)),
            pl.BlockSpec((rows, width), lambda i: (0, g_block)),
            pl.BlockSpec((head_dim,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((rows, width), lambda i: (0, 0)),
        name="gated_rms_norm_sigmoid",
    )(x, g, weight)


_OP: Callable | None = None
_OP_LOCK = Lock()


def get_gated_rms_norm_op() -> Callable:
    """Register the custom op; call once before torch.compile traces a model."""
    global _OP
    if _OP is not None:
        return _OP
    with _OP_LOCK:
        if _OP is not None:
            return _OP
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _wrapped(
            x: jax.Array, g: jax.Array, weight: jax.Array, eps: float, g_block: int
        ) -> jax.Array:
            return gated_rms_norm_jax(x, g, weight, eps=eps, g_block=g_block)

        op = cpu_export_jax_op(_OP_NAME, _wrapped)

        def _fake(x, g, weight, eps, g_block):
            del g, weight, eps, g_block
            return x.new_empty((x.shape[0], x.shape[1] * x.shape[2]))

        op.register_fake(_fake)
        _OP = op
        return op


def gated_rms_norm_tpu(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    g_block: int = 0,
) -> torch.Tensor:
    """x ``[T, H, D]``; g ``[T, >= (g_block + 1) * H * D]`` with the gate in
    column block ``g_block``; weight ``[D]``. Returns ``[T, H * D]``."""
    return get_gated_rms_norm_op()(x, g, weight, eps, g_block)
