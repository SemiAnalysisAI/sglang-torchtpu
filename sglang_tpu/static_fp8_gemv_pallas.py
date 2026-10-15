# SPDX-License-Identifier: Apache-2.0
"""Pallas decode GEMV for static-input-scale fp8 weights (``StaticFp8Weight``).

``apply_tpu_static_fp8_linear`` lowers to an activation-quantize fusion, an
XLA fp8 dot over 16 padded rows and a scale fusion. At decode the dot reads
the fp8 weight well below HBM bandwidth (Kimi-K3 TP32: the KDA [q, k, v, g]
projection took 12.3 us, the MoE front 12.6 us). This kernel streams the
[K, N] weight in K blocks with the default double buffering, quantizes the
activation in VMEM, accumulates in f32 and applies the per-channel output
scale before the store, so one custom call replaces the three ops (KDA qkvg
12.3 -> 3.6 us, MoE front 12.6 -> 10.7 us in the full model).

The math matches the XLA path: the same clamp-then-cast activation quantize,
an fp8 x fp8 MXU dot with f32 accumulation and the f32 output scale.
"""

import functools
from threading import Lock

import torch

# Decode-sized batches only; larger M keeps the XLA dot.
MAX_TOKENS = 16
# One fp8 sublane tile; the activation is zero-padded to it.
_ROWS = 32
_FP8_MAX = 448.0
# Scoped VMEM the kernel may use: the double-buffered weight block dominates.
_VMEM_LIMIT = 40 * 1024 * 1024
_VMEM_BUDGET = 32 * 1024 * 1024
# Largest K block tried; 1792 measured best on the Kimi-K3 TP32 shapes.
_MAX_BLOCK_K = 1792

_OP: list = []
_LOCK = Lock()
# Plain bool, not list truthiness: Dynamo traces the eligibility check.
_READY = False


def block_k(k: int, n: int) -> int:
    """Largest 128-multiple divisor of ``k`` that is <= _MAX_BLOCK_K and whose
    working set fits the VMEM budget; 0 when none fits."""
    want = min(_MAX_BLOCK_K, k)
    # f32 accumulator + bf16 output tile, plus the double-buffered x block.
    fixed = _ROWS * n * (4 + 2 * 2)
    for bk in range(want - want % 128, 0, -128):
        if k % bk:
            continue
        if fixed + 2 * bk * n + 2 * _ROWS * bk * 2 <= _VMEM_BUDGET:
            return bk
    return 0


def _kernel(inv_ref, x_ref, w_ref, s_ref, o_ref, acc_ref, *, nk: int):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    k = pl.program_id(0)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    x = x_ref[...].astype(jnp.float32)
    xq = jnp.clip(x * inv_ref[0], -_FP8_MAX, _FP8_MAX).astype(w_ref.dtype)
    acc_ref[...] += jnp.dot(xq, w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == nk - 1)
    def _():
        o_ref[...] = (acc_ref[...] * s_ref[...]).astype(o_ref.dtype)


def static_fp8_gemv_jax(x, w, out_scale, inv_x, bk: int):
    """x [M, K] (M <= MAX_TOKENS), w fp8 [K, N], out_scale f32 [N],
    inv_x f32 [1] (the static activation scale's inverse) -> x.dtype [M, N]."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m, kdim = x.shape
    n = w.shape[1]
    nk = kdim // bk
    xp = jnp.pad(x, ((0, _ROWS - m), (0, 0)))
    out = pl.pallas_call(
        functools.partial(_kernel, nk=nk),
        out_shape=jax.ShapeDtypeStruct((_ROWS, n), x.dtype),
        grid=(nk,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec((_ROWS, bk), lambda i: (0, i)),
            pl.BlockSpec((bk, n), lambda i: (i, 0)),
            pl.BlockSpec((1, n), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((_ROWS, n), lambda i: (0, 0)),
        scratch_shapes=[pltpu.VMEM((_ROWS, n), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT,
        ),
        name=f"static_fp8_gemv_k{kdim}_n{n}_bk{bk}",
    )(inv_x.astype(jnp.float32), xp, w, out_scale.reshape(1, n).astype(jnp.float32))
    return out[:m]


def _build_op():
    import jax  # noqa: F401  (force loud failure if JAX is missing)

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        x: jax.Array, w: jax.Array, out_scale: jax.Array, inv_x: jax.Array
    ) -> jax.Array:
        return static_fp8_gemv_jax(
            x, w, out_scale, inv_x, block_k(x.shape[1], w.shape[1])
        )

    def _fake(x, w, out_scale, inv_x):
        return x.new_empty((x.shape[0], w.shape[1]))

    op = cpu_export_jax_op("sglang_tpu_pallas::static_fp8_gemv", _wrapped)
    op.register_fake(_fake)
    return op


def register() -> None:
    """Register the op; must run before torch.compile traces the model (the
    static fp8 binder calls it post-load)."""
    global _READY
    with _LOCK:
        if not _OP:
            _OP.append(_build_op())
        _READY = True


def eligible(x: torch.Tensor, w) -> bool:
    return (
        _READY
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.dtype == torch.bfloat16
        and block_k(x.shape[1], w.weight.shape[1]) > 0
    )


def run(x: torch.Tensor, w) -> torch.Tensor:
    out = _OP[0](x, w.weight, w.out_scale, w.inv_x)
    return out[:, : w.n_out] if w.n_out else out


def pad_n(w, force: bool = False):
    """Zero-pad N to a 128 multiple (``n_out`` keeps the real width).

    An unaligned fp8 [K, N] operand makes XLA relayout the whole weight before
    every call (Kimi-K3 MLA q_a, N=2112: 166 us per call). Padding is only
    applied when the kernel is registered (or ``force``). The pad is cloned
    into a base tensor: a view graph input costs an eager as_strided per step.
    """
    n = w.weight.shape[1]
    if not (_READY or force) or n % 128 == 0:
        return w
    extra = 128 - n % 128
    weight = (
        torch.nn.functional.pad(w.weight.view(torch.uint8), (0, extra))
        .view(w.weight.dtype)
        .clone()
    )
    out_scale = torch.nn.functional.pad(w.out_scale, (0, extra)).clone()
    return w._replace(weight=weight, out_scale=out_scale, n_out=n)
