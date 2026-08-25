# SPDX-License-Identifier: Apache-2.0
"""Public API for the intra-chip tensor-parallel (weight-split tp=2) all-gather-matmul unpooler.

``matmul_intra_tp_all_gather`` is the entry point, for a weight SHARDED on N 2-way across
the 2 on-chip cores (the fast core-bridge): each core reads only ``[N/2, K]``. A pre-matmul
core-bridge token exchange assembles each chip's ``[T/8, K]`` block (FLOPs stay non-redundant), the
half-W matmul yields ``[T/8, N/2]`` pieces, and those are gathered to ``[T, N]`` replicated by one of:

  * ``_matmul_all_gather_tp_intra_topo`` -- the topology-aware bridge-last routed gather (schedule in
    ``_routing.py``; the fastest variant; routes by PHYSICAL coord so ANY 16-core 2x2x2 order works,
    incl. the serve's ring/snake).
  * ``_matmul_all_gather_tp_intra_bcast`` -- a flat 15-peer broadcast (order-agnostic fallback).
  * ``_xla_intra_tp_all_gather`` -- a pure-XLA fallback for shapes/meshes the Pallas kernels cannot
    handle (correct for ANY T / mesh; see the ``m_chip`` tiling note below).

Both Pallas kernels write ``[m_chip, bn]`` output blocks into ``o_hbm`` at offset ``chip*m_chip``
(``m_chip = T/8``). The isolated kernel can compile with an 8-row HBM tile, but full decoder-layer
compiles can choose a 16-row HBM tile for the output buffer. ``auto`` therefore only selects a Pallas
kernel when ``m_chip`` is a multiple of 16 -- i.e. ``T`` a multiple of 128 (T>=128) -- so the same API is
valid both in isolation and inside the full model graph. Forced ``gather_kind="topo"`` / ``"flat"``
keeps the lower bound at the kernel's raw tiling requirement and may still compile for ``T=64`` in
microbench-only graphs. The Pallas paths also hold the ``[T/8, K]`` token block in VMEM, capping ``T``
at ~8192 (the ceiling drops with K). Below the ``auto`` tile bound, ABOVE ``T=8192``, and on any
non-resolvable-2x2x2 mesh, ``auto`` falls back to the XLA path. Both Pallas kernels also require
``core_axis`` (size 2) to be the PHYSICAL on-chip pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import util
from ._matmul_tp_intra_bcast import _matmul_all_gather_tp_intra_bcast
from ._matmul_tp_intra_local import _matmul_intra_tp_local_output
from ._matmul_tp_intra_topo import (
    _matmul_all_gather_tp_intra_topo,
    matmul_intra_tp_all_gather_post_moe_epilogue,
)
from ._routing import _is_2x2x2_slice

P = jax.sharding.PartitionSpec

# Above this many tokens the Pallas kernels OOM: they hold the chip's token block x_full=[T/8,K]
# (+ x_sib=[T/16,K]) resident in VMEM, which grows ~linearly with T. 8192 is the measured ceiling at
# K=8192 (67 MB/core VMEM); larger K lowers it further, so this is a safe cap for K<=8192. Above it
# ``auto`` uses the XLA fallback (its intermediates are HBM-backed -- no VMEM ceiling).
_PALLAS_MAX_T = 8192


def _xla_intra_tp_all_gather(
    x: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    chip_axis: str,
    core_axis: str,
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    w_transposed: bool = False,
    x_scale: jax.Array | None = None,
) -> jax.Array:
    """Pure-XLA fallback (correct for ANY T / mesh / dtype, no tiling or core-pairing constraint).
    Cross-core (on-chip) all-gather BEFORE the matmul, the remaining all-gathers AFTER -- gathers
    activations + output, never the weight, and non-redundantly: all-gather ``xx`` over ``core_axis``
    -> the chip's ``[T/8, K]`` token block, matmul against the LOCAL half-W -> ``[T/8, N/2]``, then
    all-gather over ``chip_axis`` (tokens) -> ``[T, N/2]`` and over ``core_axis`` (N) -> ``[T, N]``
    replicated. Keeping W local (no weight gather) + the per-chip token block (each matmul runs once,
    no redundancy) is the win. Used when the Pallas kernels' ``m_chip`` HBM tiling constraint isn't met
    (T not a multiple of 64) or the mesh isn't a 2x2x2 slice.

    ``w_transposed=True`` -> ``w`` is ``[N, K]`` (ColumnParallelLinear ``[out, in]``, sharded on N =
    dim 0 over ``core_axis``); ``False`` -> ``[K, N]`` (sharded on N = dim 1). ``x_scale`` (per-token
    online-FP8 row scale for a raw fp8 ``x``) is a shard_map input, dequantized per-core inside ``f``
    via ``util.dequant_activation`` (a no-op when ``None``).
    """
    axis_names = (chip_axis, core_axis)
    # w sharded on N over core_axis: dim 0 for [N, K] (w_transposed), dim 1 for [K, N].
    w_spec = P(core_axis, None) if w_transposed else P(None, core_axis)

    def f(xx, ww, ss=None):
        # Dequant a raw fp8 xx -> bf16 by its per-token row scale (no-op if ss is None), done
        # per-core INSIDE the shard_map (on the local [T/16, K] shard, [T/16, 1] scale).
        xx = util.dequant_activation(xx, ss, out_dtype)
        od = out_dtype if out_dtype is not None else xx.dtype
        # cross-core (on-chip) all-gather BEFORE the matmul: build the chip's [T/8,K] token block.
        xc = jax.lax.all_gather(
            xx, core_axis, axis=0, tiled=True
        )  # [T/16,K] -> [T/8, K]
        y = util.reorder_matmul(
            xc,
            ww,
            w_transposed=w_transposed,
            preferred_element_type=jnp.float32,
            out_dtype=od,
        )  # [T/8, N/2]  (LOCAL half-W, non-redundant)
        # remaining all-gathers AFTER the matmul: chip on tokens, then core on N.
        yt = jax.lax.all_gather(
            y, chip_axis, axis=0, tiled=True
        )  # [T/8,N/2] -> [T, N/2]
        return jax.lax.all_gather(
            yt, core_axis, axis=1, tiled=True
        )  # [T,N/2] -> [T, N]

    # x_scale (a per-token [T] row scale for a raw fp8 x) is a shard_map INPUT so f dequants the
    # LOCAL [T/16, 1] shard -- closing over the global [T] scale would mis-shard it.
    if x_scale is None:
        operands = (x, w)
        in_specs = (P(axis_names, None), w_spec)
    else:
        operands = (x, w, x_scale.reshape(x.shape[0], 1))
        in_specs = (P(axis_names, None), w_spec, P(axis_names, None))
    return jax.jit(
        jax.shard_map(
            f,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(None, None),
            check_vma=False,
        )
    )(*operands)


def matmul_intra_tp_output_gather(
    x: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    chip_axis: str,
    core_axis: str,
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    w_transposed: bool = False,
    x_scale: jax.Array | None = None,
    gather_kind: str = "auto",
    bn: int | None = None,
) -> jax.Array:
    """Half-W matmul with only the 2-way intra-chip output gather.

    This is the SP pooler twin of ``matmul_intra_tp_all_gather``. ``x`` is token
    sharded over the full ``(chip_axis, core_axis)`` product, so each core owns
    ``[T/tp, K]`` rows. ``w`` is split on N over the 2-way on-chip ``core_axis``.
    The XLA fallback shard body:

      1. gathers the sibling token rows over ``core_axis`` to form this chip's
         ``[T/n_chips, K]`` block,
      2. computes the local half-width matmul, and
      3. gathers the two N halves over ``core_axis`` before slicing back to this
         core's original token rows.

    The Pallas path exchanges only the sibling's ``[T/tp, K]`` rows, overlaps
    that bridge transfer with the local-token matmul, and writes the sibling's
    result directly to its output shard. It avoids materializing either the
    gathered ``[T/n_chips, K]`` activation or output.

    Output stays token-sharded over the full TP product:
    ``P((chip_axis, core_axis), None)``. ``gather_kind="auto"`` selects Pallas
    for measured TP16 shapes with at least eight local rows (global T>=128) on
    a physical 2x2x2 slice and XLA otherwise. ``"pallas"`` and ``"xla"`` force
    a path for benchmarking.
    """
    if gather_kind not in ("auto", "pallas", "xla"):
        raise ValueError(
            f"gather_kind must be 'auto' | 'pallas' | 'xla', got {gather_kind!r}"
        )
    if gather_kind == "pallas" and x_scale is not None:
        raise ValueError("gather_kind='pallas' does not support x_scale")

    axis_names = (chip_axis, core_axis)
    tp_size = int(mesh.shape[chip_axis]) * int(mesh.shape[core_axis])
    m_local = x.shape[0] // tp_size
    pallas_ok = (
        x_scale is None
        and tp_size == 16
        and x.shape[0] % tp_size == 0
        and m_local >= 8
        and m_local % 8 == 0
        and _is_2x2x2_slice(mesh, axis_names)
    )
    if gather_kind == "pallas" or (gather_kind == "auto" and pallas_ok):
        return _matmul_intra_tp_local_output(
            x,
            w,
            mesh,
            chip_axis,
            core_axis,
            bn=bn,
            out_dtype=out_dtype,
            w_transposed=w_transposed,
        )

    w_spec = P(core_axis, None) if w_transposed else P(None, core_axis)

    def f(xx, ww, ss=None):
        xx = util.dequant_activation(xx, ss, out_dtype)
        od = out_dtype if out_dtype is not None else xx.dtype
        xc = jax.lax.all_gather(
            xx, core_axis, axis=0, tiled=True
        )  # [T/tp,K] -> [T/n_chips,K]
        y_half = util.reorder_matmul(
            xc,
            ww,
            w_transposed=w_transposed,
            preferred_element_type=jnp.float32,
            out_dtype=od,
        )  # [T/n_chips,N/2]
        y_full = jax.lax.all_gather(
            y_half, core_axis, axis=1, tiled=True
        )  # [T/n_chips,N]
        m_local = xx.shape[0]
        row0 = jax.lax.axis_index(core_axis) * m_local
        return jax.lax.dynamic_slice_in_dim(y_full, row0, m_local, axis=0)

    if x_scale is None:
        operands = (x, w)
        in_specs = (P(axis_names, None), w_spec)
    else:
        operands = (x, w, x_scale.reshape(x.shape[0], 1))
        in_specs = (P(axis_names, None), w_spec, P(axis_names, None))

    return jax.jit(
        jax.shard_map(
            f,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(axis_names, None),
            check_vma=False,
        )
    )(*operands)


def matmul_intra_tp_all_gather(
    x: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    chip_axis: str,
    core_axis: str,
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    w_transposed: bool = False,
    x_scale: jax.Array | None = None,
    gather_kind: str = "auto",
    bn: int | None = None,
    collective_id: int = 0,
    double_barrier: bool = True,
    disable_bounds_checks: bool = False,
) -> jax.Array:
    """``all_gather(half-W matmul) -> [T, N]`` for a weight split tp=2 over the 2 on-chip cores.

    ``x`` ``[T, K]`` EP-sharded ``P((chip_axis, core_axis), None)`` -> local ``[T/16, K]``;
    ``w`` (default ``[K, N]``, or ``[N, K]`` with ``w_transposed``) sharded on N over ``core_axis``
    -> local half-W; output ``[T, N]`` replicated.

    Parity with ``matmul_replicated_weights_all_gather.matmul_all_gather``:

    * ``w_transposed=True`` -> ``w`` is the ColumnParallelLinear ``[out=N, in=K]`` native layout
      (sharded on N = dim 0 over ``core_axis`` -> local ``[N/2, K]``); ``False`` -> ``[K, N]`` (sharded
      on N = dim 1 -> local ``[K, N/2]``). Every path contracts K without materializing a transpose.
    * ``x_scale`` (per-token online-FP8 row scale ``[T]`` for a RAW fp8 ``x``) -> dequanted via
      ``util.dequant_activation``. Like ``_matmul_topo``, the Pallas paths keep ``x`` fp8 on the wire:
      the raw fp8 ``x`` + its per-token scale are BOTH exchanged over the core-bridge and the assembled
      ``x_full`` is dequanted in-VMEM at the matmul (``reorder_matmul(..., x_scale=...)``). The XLA
      fallback dequants its local shard before the gather. ``x_scale`` is passed as a kernel input
      (omitted, with its BlockSpec, when ``None``).

    ``gather_kind`` (default ``"auto"``):
      * ``"auto"`` -- the topo routed gather when ``T`` is full-graph Pallas-supported
        (``m_chip = T/8`` a multiple of 16 i.e. ``T`` a multiple of 128, AND ``T <= 8192`` the VMEM
        ceiling) AND the mesh is a resolvable 16-core 2x2x2 slice; else the flat broadcast (still
        Pallas, same T constraints); else the XLA fallback (any T -- incl. T>8192 -- and any mesh).
      * ``"topo"`` / ``"flat"`` -- force that Pallas kernel (will fail to compile if ``m_chip`` isn't a
        multiple of 8, OOM above the VMEM ceiling, or -- topo -- the mesh isn't a valid 2x2x2 slice).
      * ``"xla"`` -- force the pure-XLA fallback (correct for any T / mesh).

    ``core_axis`` MUST be the size-2 physical on-chip pair (bit 0) for the Pallas paths: the weight
    N-half split is tied to the core-bridge, so the ``core_axis`` index has to be the physical core.
    (The routed gather works on any 2x2x2 mesh ORDER; only this core-pairing + ``T`` multiple of 128 are
    required. The XLA fallback has neither constraint.)
    """
    axis_names = (chip_axis, core_axis)
    core_size = int(mesh.shape[core_axis])
    tp_size = int(mesh.shape[chip_axis]) * core_size
    # Pallas kernels write [m_chip, bn] blocks to o_hbm at offset chip*m_chip. In isolation the output
    # HBM tile can be 8 rows, but full decoder graphs can choose a 16-row tile for the same output
    # buffer; T=64 then attempts writes at row offsets 8, 24, ... and Mosaic rejects the slice. Make
    # auto conservative for the public/model path. Forced topo/flat remains available to microbench the
    # raw kernel at its lower tiling bound.
    m_chip = core_size * (x.shape[0] // tp_size)
    pallas_ok = (m_chip % 16 == 0) and (x.shape[0] <= _PALLAS_MAX_T)

    if gather_kind == "auto":
        if pallas_ok and _is_2x2x2_slice(mesh, axis_names):
            kind = "topo"
        elif pallas_ok:
            kind = "flat"
        else:
            kind = "xla"
    elif gather_kind in ("topo", "flat", "xla"):
        kind = gather_kind
    else:
        raise ValueError(
            f"gather_kind must be 'auto' | 'topo' | 'flat' | 'xla', got {gather_kind!r}"
        )

    if kind == "xla":
        return _xla_intra_tp_all_gather(
            x,
            w,
            mesh,
            chip_axis,
            core_axis,
            out_dtype=out_dtype,
            w_transposed=w_transposed,
            x_scale=x_scale,
        )
    if kind == "topo":
        return _matmul_all_gather_tp_intra_topo(
            x,
            w,
            mesh,
            chip_axis,
            core_axis,
            bn=bn,
            collective_id=collective_id,
            double_barrier=double_barrier,
            disable_bounds_checks=disable_bounds_checks,
            out_dtype=out_dtype,
            w_transposed=w_transposed,
            x_scale=x_scale,
        )
    return _matmul_all_gather_tp_intra_bcast(
        x,
        w,
        mesh,
        chip_axis,
        core_axis,
        bn=bn,
        collective_id=collective_id,
        double_barrier=double_barrier,
        disable_bounds_checks=disable_bounds_checks,
        out_dtype=out_dtype,
        w_transposed=w_transposed,
        x_scale=x_scale,
    )
