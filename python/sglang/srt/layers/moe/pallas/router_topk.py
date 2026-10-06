# SPDX-License-Identifier: Apache-2.0
"""Pallas fused router GEMV + sigmoid top-k for TPU MoE decode.

One kernel computes ``x @ W`` for the router and selects the top-k experts by
``k`` max-and-mask passes over the VMEM-resident logits, replacing the router
matmul plus the sigmoid / bias / ``torch.topk`` (a full sort on TPU) / gather /
renormalize ops of ``biased_grouped_topk_impl`` for ungrouped sigmoid routing.
Adapted from vllm-torchtpu ``kernels/router_topk.py``.
"""

# PEP 563 must remain disabled so `torch_tpu._internal.pallas` sees `jax.Array`
# type objects rather than stringified annotations at registration time.

import functools
import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Largest router-weight K block (bytes of a bf16 [tk, N] tile).
_GATE_BLOCK_BYTES = 4 * 1024 * 1024
# Max tokens for the fused kernel; larger batches keep the unfused path.
GATE_ROUTE_MAX_ROWS = 64


def _gate_route_kernel(x_ref, w_ref, bias_ref, w_out_ref, i_out_ref, acc_ref, **kw):
    nk, k, n = kw["nk"], kw["k"], kw["n"]
    step = pl.program_id(0)

    @pl.when(step == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(step == nk - 1)
    def _():
        # Mirror the unfused TPU route: a bf16 F.linear, a bf16 sigmoid, then
        # the fp32 bias add of biased_grouped_topk_impl.
        logits = acc_ref[...].astype(jnp.bfloat16).astype(jnp.float32)
        scores = jax.nn.sigmoid(logits).astype(jnp.bfloat16).astype(jnp.float32)
        x = scores + bias_ref[...]
        masked = jnp.finfo(jnp.float32).min
        iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
        for j in range(k):
            m = jnp.max(x, axis=1, keepdims=True)
            # Exact ties go to the lowest expert id (jax.lax.top_k semantics;
            # Mosaic's argmax does not guarantee it).
            idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
            hit = iota == idx[:, None]
            # Lane gathers wider than one vreg do not lower; select-and-sum.
            w_out_ref[:, j] = jnp.sum(jnp.where(hit, scores, 0.0), axis=1)
            i_out_ref[:, j] = idx
            x = jnp.where(hit, masked, x)
        w = w_out_ref[...]
        if kw["renormalize"]:
            w = w / (jnp.sum(w, axis=1, keepdims=True) + kw["eps"])
        w_out_ref[...] = w * kw["scale"] if kw["scale"] != 1.0 else w


def _gate_block_k(k_dim: int, n: int) -> int:
    """Largest 128-multiple divisor of ``k_dim`` whose weight tile fits; 0 if none."""
    tk = k_dim if k_dim % 128 == 0 else 0
    while tk > 0 and (tk * n * 2 > _GATE_BLOCK_BYTES or k_dim % tk):
        tk -= 128
    return tk


def gate_route_supported(k_dim: int, n: int) -> bool:
    return _gate_block_k(k_dim, n) > 0


@functools.partial(
    jax.jit, static_argnames=("k", "renormalize", "scale", "eps", "interpret")
)
def gate_route_jax(x, weight_kn, bias, k, renormalize, scale, eps, interpret=False):
    """bf16 ``x`` [T, K] @ ``weight_kn`` [K, N] and its biased sigmoid top-k.

    Returns f32 weights (of the unbiased scores, descending by biased score)
    and int32 expert ids, both [T, k].
    """
    rows, k_dim = x.shape
    n = weight_kn.shape[1]
    tk = _gate_block_k(k_dim, n)
    nk = k_dim // tk
    kw = dict(nk=nk, k=k, n=n, renormalize=renormalize, scale=scale, eps=eps)
    out_spec = pl.BlockSpec((rows, k), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(_gate_route_kernel, **kw),
        grid=(nk,),
        in_specs=[
            pl.BlockSpec((rows, tk), lambda i: (0, i)),
            pl.BlockSpec((tk, n), lambda i: (i, 0)),
            pl.BlockSpec((1, n), lambda i: (0, 0)),
        ],
        out_specs=[out_spec, out_spec],
        out_shape=[
            jax.ShapeDtypeStruct((rows, k), jnp.float32),
            jax.ShapeDtypeStruct((rows, k), jnp.int32),
        ],
        scratch_shapes=[pltpu.VMEM((rows, n), jnp.float32)],
        interpret=interpret,
        name=f"gate_route_{k}_of_{n}",
    )(x, weight_kn, bias.reshape(1, n).astype(jnp.float32))


# A dict, not functools.cache: dynamo traces through the cache wrapper.
_gate_route_op_cache: dict[tuple, object] = {}


def get_gate_route_op(k: int, renormalize: bool, scale: float, eps: float):
    """Build the exported op; call once outside torch.compile tracing."""
    key = (k, renormalize, float(scale), float(eps))
    op = _gate_route_op_cache.get(key)
    if op is None:
        from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        def _wrapped(
            x: jax.Array, w: jax.Array, bias: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return gate_route_jax(x, w, bias, *key)

        suffix = f"k{k}_r{int(renormalize)}_s{key[2].hex()}_e{key[3].hex()}"
        for ch in "-+.":
            suffix = suffix.replace(ch, "_")
        op = cpu_export_jax_op(f"tpu::gate_route_{suffix}", _wrapped)
        _gate_route_op_cache[key] = op
    return op


def gate_route_tpu(
    x: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    renormalize: bool,
    scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch TPU entry; ``x`` must have at most ``GATE_ROUTE_MAX_ROWS`` rows."""
    op = get_gate_route_op(k, renormalize, scale, eps)
    return op(x.contiguous(), weight_kn, bias)
