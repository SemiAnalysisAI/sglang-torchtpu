# SPDX-License-Identifier: Apache-2.0
"""Collective-barrier and matmul utilities for intra-chip kernels."""

import functools

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def reorder_matmul(
    x, w, *, w_transposed, preferred_element_type=None, out_dtype=None, x_scale=None
):
    """The per-core reorder matmul ``x[m, K] @ W`` -> ``[m, N]`` respecting W's layout.

    ``w_transposed=False``: ``w`` is ``[K, N]`` -> plain ``jnp.dot``.
    ``w_transposed=True``: ``w`` is ``[N, K]`` (the ColumnParallelLinear native ``[out, in]`` layout)
    -> contract K via ``dot_general`` (dims (1, 1)). The MXU consumes the RHS transposed natively, so
    this needs no physical transpose / relayout copy of the weight.

    ``x_scale`` (per-token ``[m]`` or ``[m, 1]``, optional): when given, ``x`` is a RAW fp8 activation
    and this is its online-FP8 row scale -- dequantize as ``(x.astype(f32) * scale) -> w.dtype``
    before the matmul.
    """
    if x_scale is not None:
        s = x_scale if x_scale.ndim == 2 else x_scale[:, None]
        x = (x.astype(jnp.float32) * s).astype(w.dtype)
    elif x.dtype != w.dtype:
        x = x.astype(w.dtype)
    if w_transposed:
        out = lax.dot_general(
            x,
            w,
            (((1,), (1,)), ((), ())),
            preferred_element_type=preferred_element_type,
        )
    else:
        out = jnp.dot(x, w, preferred_element_type=preferred_element_type)
    return out.astype(out_dtype) if out_dtype is not None else out


def dequant_activation(x, x_scale, out_dtype=None):
    """Dequantize a raw fp8 activation ``x`` by its per-token online-FP8 row scale ``x_scale``
    (``[T]`` or ``[T, 1]``): ``(x.astype(f32) * scale).astype(out_dtype or bf16)``. Returns ``x``
    UNCHANGED when ``x_scale is None``."""
    if x_scale is None:
        return x
    s = x_scale if x_scale.ndim == 2 else x_scale[:, None]
    return (x.astype(jnp.float32) * s).astype(out_dtype or jnp.bfloat16)


def axis_tuple_strides(mesh, axis_names):
    """Row-major mixed-radix layout of a token dim sharded over ``axis_names`` (a tuple of mesh axes,
    first = MOST significant). Returns ``(tp_size, strides, sizes)`` where ``strides[j] =
    prod(sizes[j+1:])``, so a device's raveled index is ``sum(axis_index(a_j) * strides[j])``.
    """
    sizes = tuple(int(mesh.shape[a]) for a in axis_names)
    strides = []
    acc = 1
    for s in reversed(sizes):
        strides.append(acc)
        acc *= s
    strides = tuple(reversed(strides))
    return acc, strides, sizes  # acc == prod(sizes) == tp_size


def combined_axis_index(axis_names, strides):
    """Runtime raveled index over ``axis_names``: ``sum(axis_index(a_j) * strides[j])`` (row-major).
    Called inside a Pallas kernel."""
    idx = jnp.int32(0)
    for a, st in zip(axis_names, strides):
        idx = idx + lax.axis_index(a) * jnp.int32(int(st))
    return idx


def make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names):
    """Return ``_peer(q)`` -> the ``DeviceIdType.MESH`` coordinate tuple whose TARGET axes carry
    ``q``'s mixed-radix digits and whose NON-target axes carry THIS device's own ``axis_index``.
    Inverse of ``combined_axis_index``."""
    radix = {
        a: (jnp.int32(int(st)), jnp.int32(int(sz)))
        for a, st, sz in zip(axis_names, strides, sizes)
    }

    def _peer(q):
        return tuple(
            (
                lax.rem(lax.div(q, radix[nm][0]), radix[nm][1])
                if nm in radix
                else lax.axis_index(nm)
            )
            for nm in mesh_axis_names
        )

    return _peer


def product_device_line(mesh, axis_names):
    """The ``tp_size`` devices of the axis-tuple's product, in raveled row-major order, from the
    non-target axes' index-0 slice (a representative collective group). Returns a flat list of
    devices, or ``None`` off a resolvable TPU mesh so callers fall back."""
    try:
        import numpy as np

        dev = np.asarray(mesh.devices)
        names = list(mesh.axis_names)
        _, strides, sizes = axis_tuple_strides(mesh, axis_names)
        axis_pos = [names.index(a) for a in axis_names]
        tp = 1
        for s in sizes:
            tp *= s
        line = []
        for q in range(tp):
            coord = [0] * dev.ndim
            for ap, st, sz in zip(axis_pos, strides, sizes):
                coord[ap] = (q // st) % sz
            line.append(dev[tuple(coord)])
        return line
    except Exception:  # noqa: BLE001 -- non-TPU / no coords -> caller falls back
        return None


def hypercube_local_barrier(neighbor_ids, double_barrier=True):
    """All-to-all barrier: signal every neighbor's barrier sem, wait for all of them. The optional
    ``double_barrier`` does a second barrier on a fresh ``run_scoped`` REGULAR semaphore, preventing
    a collective_id-reuse race where a fast neighbor re-enters and unblocks this device early.

    Args:
      neighbor_ids: list of neighbor device ids (each a tuple of mesh coordinates).
      double_barrier: whether to perform the second (reuse-safe) barrier.
    """
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor_id in neighbor_ids:
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=neighbor_id,
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pltpu.semaphore_wait(barrier_sem, len(neighbor_ids))
    if double_barrier:

        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for neighbor_id in neighbor_ids:
                pltpu.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=neighbor_id,
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(second_barrier, len(neighbor_ids))
