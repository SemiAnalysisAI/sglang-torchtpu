# SPDX-License-Identifier: Apache-2.0
"""Public API for the REPLICATED-weight all-gather-matmul ``all_gather(x, axis=0) @ W``.

``matmul_all_gather`` is the size-dispatching entry point: per-device output size picks the
fastest of the small-T fused broadcast kernel (``_matmul_broadcast._matmul_all_gather_broadcast``),
the mid/upper-T topology-aware hypercube gather (``_matmul_topo._matmul_all_gather_topo``, on any
16-core 2x2x2 slice), the large-T pure-XLA reorder, or plain naive. ``x`` is ``[T/P, K]``
(M-sharded), ``W`` is ``[K, N]`` (or ``[N, K]`` with ``w_transposed``) replicated; output
``[T, N]`` replicated. The Pallas broadcast kernel lives in ``_matmul_broadcast.py`` and the topo
gather in ``_matmul_topo.py``; the XLA reorder/naive fallbacks + the dispatch gates live here. See
``README.md`` for the design + measurements.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import util
from ._matmul_broadcast import _matmul_all_gather_broadcast
from ._matmul_topo import _is_2x2x2_slice, _matmul_all_gather_topo

P = jax.sharding.PartitionSpec


def matmul_all_gather(
    x: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str | tuple[str, ...],
    *,
    fused_out_bytes: int = 3 * 1024 * 1024,
    ring_min_tk_over_n: int = 384,
    w_transposed: bool = False,
    out_dtype: jax.typing.DTypeLike | None = None,
    x_scale: jax.Array | None = None,
    gather_kind: str = "auto",
) -> jax.Array:
    """``all_gather(x, axis=0) @ w`` for a REPLICATED weight, picking the fastest path.

    Picks by a rule calibrated on a real tpu7x 2x2x2 sweep over T in [32, 8192] for the two
    shapes shipped, N/K in {0.5, 2} (device-work MIN-per-iter):

    * ``_matmul_all_gather_broadcast`` -- in-kernel broadcast that overlaps the matmul.
      Best when ``T/P >= 16`` (packed-bf16 DMA alignment) AND the output is small
      enough to be latency-bound (``T*N*itemsize <= fused_out_bytes``, ~<=3MB). 1.15-1.63x.
    * ``_matmul_all_gather_topo`` -- the topology-aware 2x2x2x2 hypercube store-and-forward
      gather (step 1b). Taken for T>=256 (``T/P >= 16``) on ANY 16-core 2x2x2 slice
      (``_is_2x2x2_slice``): it reads the physical topology from the device coords and routes by
      physical hypercube coordinate, so any mesh order works (hypercube, the serve's
      create_device_mesh ring/snake, ...) -- no hypercube-ordering requirement. Measured the
      fastest path for T>=256 (1.09x@256 -> 1.37x@1024 over the next best) and the only fused path
      that fits VMEM at large T.
    * ``_xla_matmul_all_gather`` -- pure-XLA reorder-matmul of own rows + XLA's ring all-gather. The
      large-T fallback (``T*K >= ring_min_tk_over_n * N``) taken when topo does not apply (not a
      resolvable 2x2x2 slice). 1.1x (up-proj) up to ~3.2x (down-proj).
    * ``_xla_all_gather_matmul`` -- plain ``all_gather(x) @ w`` -- otherwise (tiny T) nothing beats it.

    ``gather_kind`` (default ``"auto"``): the size dispatch above. topo has FULL parity with the
    other paths (``w_transposed``, ``out_dtype``, and ``x_scale`` are all supported -- an fp8 ``x``
    with a bf16 ``out_dtype``, a ``[N, K]`` weight, and an online-FP8 row scale all work), so
    nothing but the 2x2x2 mesh gates it. ``"topo"`` is the opt-in override that FORCES topo
    regardless of shape (still needs a 16-core 2x2x2 slice, any order).

    Both ``x`` ``[T/P, K]`` ``P(axis_name, None)``, ``w`` ``[K, N]`` replicated
    ``P(None, None)``; output ``[T, N]`` replicated. ``w_transposed=True`` accepts ``w``
    in the ``[N, K]`` ColumnParallelLinear ``[out, in]`` native layout instead: every path
    contracts K without a ``[N, K] -> [K, N]`` transpose ``copy`` (the Pallas paths slice
    ``[bn, K]`` N-tiles; the XLA paths fold the transpose into the dot), so the caller can
    pass the CPL weight directly rather than a materialized ``w.T``. ``axis_name`` may be a tuple of
    mesh axes -- the token dim is then sharded over all of them (e.g. SP + attn-DP
    shards tokens over ``(data, attn_dp, ..., model)``) and every path gathers the whole
    product natively (the Pallas paths via a mixed-radix combined index over the tuple). The
    ``fused_out_bytes`` / ``ring_min_tk_over_n`` constants are tpu7x-calibrated from accurate
    xprof-trace latency (per-iter union-p10).
    """
    # ``axis_name`` is a single mesh axis OR a tuple of axes -- the row/token dim is sharded
    # over ALL of them, raveled row-major (e.g. SP + attn-DP shards tokens over
    # ``(data, attn_dp, ..., model)``). EVERY path gathers the whole product natively: the
    # XLA reorder/naive via ``lax.all_gather(axes)``, and the Pallas broadcast/topo via a
    # mixed-radix combined index over the tuple (``util.combined_axis_index`` /
    # ``make_peer_id_fn``). Size-1 axes contribute a constant 0 digit (a no-op), so the full
    # tuple is passed with no size-1 special-casing.
    axes = axis_name if isinstance(axis_name, (tuple, list)) else (axis_name,)
    # Opt-in override: force the topology-aware gather (see _matmul_topo) regardless of shape (still
    # needs a 16-core 2x2x2 slice -- any order; it reads the physical topology from the coords).
    if gather_kind == "topo":
        return _matmul_all_gather_topo(
            x,
            w,
            mesh,
            axes,
            out_dtype=out_dtype,
            x_scale=x_scale,
            w_transposed=w_transposed,
        )
    if gather_kind != "auto":
        raise ValueError(f"gather_kind must be 'auto' or 'topo', got {gather_kind!r}.")
    tp_size = 1
    for a in axes:
        tp_size *= mesh.shape[a]
    t, k = x.shape
    n = w.shape[0] if w_transposed else w.shape[1]
    m_local = t // tp_size
    # The output is out_dtype, not x's; matters when x is fp8 while the output is bf16. Size the
    # small-T gate by the output width so an fp8 x doesn't under-count it.
    out_itemsize = (
        jnp.dtype(out_dtype).itemsize if out_dtype is not None else x.dtype.itemsize
    )
    out_bytes = t * n * out_itemsize

    # ``x_scale`` (per-token [T], optional): x is a RAW fp8 activation and this is its online-FP8
    # row scale. Every path OWNS its own dequant -- ``x_scale`` is forwarded to each and applied in
    # that path's builder via ``util.dequant_activation`` (a raw fp8 x with no scale is widened
    # later, inside the matmul). Kept out of this size dispatcher so the paths are uniform.
    # NOTE: an fp8 input does NOT flip the ranking toward naive. Its cheap fp8 input-gather saves
    # naive only a few us, but naive still pays the full redundant [T,K]@[K,N] matmul on every
    # device (~2x the 1/P matmul), so topo wins the mid-T band for fp8 just as for bf16 (topo
    # gathers the bf16 OUTPUT, so its cost is dtype-independent). So the size-only gates below stay
    # dtype-agnostic. (An earlier "fp8 -> naive" fast path was a benchmark artifact: the sweep's
    # op-extractor missed naive's fp8 matmul, lowered as ``convolution_convert_fusion``.)
    # 1) small-T: fused broadcast kernel -- the output is small enough to be latency-bound, so
    #    its in-kernel broadcast overlaps the matmul. Runs over the full axis product via a
    #    combined index; any axis order is correct for the broadcast ring (order only affects
    #    hop efficiency, not correctness).
    # Mosaic's packed bf16 HBM layout requires output-row DMA offsets aligned
    # to 16 rows.  An eight-row local shard (for example TP16 T=128) cannot
    # satisfy that contract, so leave it on the XLA path.
    if m_local >= 16 and out_bytes <= fused_out_bytes:
        return _matmul_all_gather_broadcast(
            x,
            w,
            mesh,
            axes,
            w_transposed=w_transposed,
            out_dtype=out_dtype,
            x_scale=x_scale,
        )
    # 1b) mid/upper-T: the topology-aware store-and-forward gather (``_matmul_topo``) on any 16-core
    #    2x2x2 slice. It reads the physical topology from the device coords and routes by physical
    #    hypercube coordinate, so ANY mesh order works (hypercube, the serve's create_device_mesh
    #    ring/snake, ...) -- no hypercube-ordering requirement. Measured fastest for T>=256 (the
    #    ``benchmark.py`` sweep, K4096 N8192, device us/iter: 1.09x@256 -> 1.30x@512 -> 1.37x@1024
    #    over the next best) and the only fused path that still fits VMEM at large T. Full parity:
    #    ``w_transposed`` / ``out_dtype`` / ``x_scale`` all forwarded. fbcast (step 1) already
    #    claims the small, HBM-aligned output region first.
    if m_local >= 16 and _is_2x2x2_slice(mesh, axes):
        return _matmul_all_gather_topo(
            x,
            w,
            mesh,
            axes,
            out_dtype=out_dtype,
            x_scale=x_scale,
            w_transposed=w_transposed,
        )
    # 2) large T (and/or a non-2x2x2 mesh): pure-XLA reorder -- the 1/P matmul's compute saving
    #    repays the bandwidth-bound output gather, which no hand-rolled gather beats at large T.
    if t * k >= ring_min_tk_over_n * n:
        return _xla_matmul_all_gather(
            mesh, axis_name, w_transposed=w_transposed, out_dtype=out_dtype
        )(x, w, x_scale)
    # 3) otherwise (tiny T, or a non-2x2x2 mesh below the reorder ceiling): plain
    #    all_gather(x) @ w -- nothing beats it there.
    return _xla_all_gather_matmul(
        mesh, axis_name, w_transposed=w_transposed, out_dtype=out_dtype
    )(x, w, x_scale)


def _xla_all_gather_matmul(mesh, axis_name, *, w_transposed=False, out_dtype=None):
    """Build a jitted callable ``(x, w, x_scale=None) -> all_gather(x) @ W`` ([T, N] replicated) --
    the plain baseline: gather x, then the (redundant) full matmul on every core. Shared by the
    size dispatch (tiny-T / non-2x2x2 fallback) AND by ``benchmark.py``.

    Fastest for tiny T on a 1D mesh: single-shot XLA paths all land ~59-62us because XLA won't
    auto-overlap matmul<->gather, and the reorder's larger output gather isn't repaid by the
    redundant-compute it saves. The mid-T dead zone is instead taken by the topology-aware
    ``_matmul_topo`` gather (see ``matmul_all_gather``). ``w_transposed`` -> ``w`` is ``[N, K]``
    (contract K, absorbed by XLA into the dot, no copy). ``x_scale`` (per-token online-FP8 row
    scale) is applied INSIDE ``_fn`` AFTER the gather -- x is gathered in its native dtype and only
    then dequantized (``util.dequant_activation``), so a raw fp8 x stays fp8 on the wire (half the
    gathered bytes) even with a scale; dequanting up front would widen it to bf16 before the gather.
    The ``[T]`` scale is gathered alongside x so the full ``[T, K]`` dequant matches the gathered rows.
    """

    def _run(x, w, x_scale=None):
        def _fn(xx, ww, xs=None):
            # Gather x in its NATIVE dtype (fp8 = half the bytes), THEN dequant the gathered
            # [T, K] activation -- keeps a raw fp8 x fp8 on the wire even with a per-token scale.
            g = jax.lax.all_gather(xx, axis_name, axis=0, tiled=True)
            if xs is not None:
                xs = jax.lax.all_gather(xs, axis_name, axis=0, tiled=True)
            g = util.dequant_activation(g, xs, out_dtype)
            od = out_dtype if out_dtype is not None else g.dtype
            return util.reorder_matmul(g, ww, w_transposed=w_transposed, out_dtype=od)

        # The per-token scale is token-sharded like x's rows; thread it through shard_map only when
        # present (a plain fp8 x with no scale is widened later, inside reorder_matmul).
        in_specs = (P(axis_name, None), P(None, None))
        args = (x, w)
        if x_scale is not None:
            in_specs = in_specs + (P(axis_name),)
            args = args + (x_scale,)
        return jax.shard_map(
            _fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(None, None),
            check_vma=False,
        )(*args)

    return jax.jit(_run)


def _xla_matmul_all_gather(mesh, axis_name, *, w_transposed=False, out_dtype=None):
    """Build a jitted callable ``(x, w, x_scale=None) -> all_gather(x @ W)`` ([T, N] replicated) --
    the emitter reorder, pure XLA. Shared by the size dispatch (large-T / non-2x2x2 fallback) AND
    by ``benchmark.py``.

    Each device matmuls only its own ``[T/P, K]`` rows (1/P the work) and XLA's pipelined-ring
    all-gather distributes the ``[T/P, N]`` output. The **large-T** path: the 64MB output gather is
    bandwidth-bound and no hand-rolled in-kernel gather beats XLA's ring. A Pallas reorder-matmul
    here was measured equal within ~1%, so XLA's matmul is used -- Pallas is reserved for the fused
    kernel + the topo gather, where the matmul and communication must share one kernel to overlap.
    XLA also tiles K itself (no VMEM tile to budget). ~1.35x (up-proj) to ~3.2x (down-proj).
    ``w_transposed`` -> ``w`` is ``[N, K]``. ``x_scale`` (per-token online-FP8 row scale) dequantizes
    the raw fp8 ``x`` INSIDE ``_fn`` -- on the local ``[T/P, K]`` shard, before its 1/P matmul (the
    gather is of the ``[T/P, N]`` output, so x's wire dtype is moot here; the dequant lives with the
    path, matching the gather-then-matmul + Pallas paths).
    """

    def _run(x, w, x_scale=None):
        def _fn(xx, ww, xs=None):
            # Dequant the local [T/P, K] shard, then the 1/P matmul, then gather the [T/P, N] output.
            xx = util.dequant_activation(xx, xs, out_dtype)
            od = out_dtype if out_dtype is not None else xx.dtype
            o = util.reorder_matmul(xx, ww, w_transposed=w_transposed, out_dtype=od)
            return jax.lax.all_gather(o, axis_name, axis=0, tiled=True)

        # The per-token scale is token-sharded like x's rows; thread it through shard_map only when
        # present (a plain fp8 x with no scale is widened later, inside reorder_matmul).
        in_specs = (P(axis_name, None), P(None, None))
        args = (x, w)
        if x_scale is not None:
            in_specs = in_specs + (P(axis_name),)
            args = args + (x_scale,)
        return jax.shard_map(
            _fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(None, None),
            check_vma=False,
        )(*args)

    return jax.jit(_run)


def build_matmul_all_gather_topo_local(
    x_shape_dtype: jax.ShapeDtypeStruct,
    w_shape_dtype: jax.ShapeDtypeStruct,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str | tuple[str, ...],
    *,
    bn: int | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
    w_transposed: bool = False,
    collective_id: int = 0,
    double_barrier: bool = True,
):
    """Build the topo matmul-all-gather for use INSIDE an existing ``shard_map``.

    ``matmul_all_gather`` owns its own ``shard_map``, so it cannot be called from inside
    one. This returns the per-device callable instead, letting a caller that is already
    sharded -- the fused-MoE body, which runs the unpooler + MoE-exit gather next to the
    post-expert norm -- invoke the same kernel with no nesting. Legal because the gather
    is neighbour DMAs inside the Mosaic kernel, not a JAX collective.

    Pass the GLOBAL shapes (the plan is built from them); call the result with LOCAL
    operands::

        call, p2p, ph2p = build_matmul_all_gather_topo_local(gx, gw, mesh, ep_axis, ...)

        def body(x_local, w, p2p, ph2p):        # inside jax.shard_map(...)
            return call(x_local, w, p2p, ph2p)  # this core's copy of [T, N]

        jax.shard_map(body, mesh=mesh,
                      in_specs=(P(ep_axis, None), P(None, None), P(None), P(None)),
                      out_specs=P(None, None), check_vma=False)(x, w, p2p, ph2p)

    Returns ``(call_fn, pos2phys, phys2pos)``; the two maps are replicated int32 arrays
    that must be threaded in as ``P(None)`` operands of the enclosing shard_map.

    Only the 16-core 2x2x2 topo path is available here -- the broadcast and XLA-reorder
    paths express their gather as a JAX collective, which WOULD nest.
    """
    # Same normalization the public dispatch does: a bare "tp" must become ("tp",) or the
    # mesh lookup iterates it per character.
    axes = axis_name if isinstance(axis_name, (tuple, list)) else (axis_name,)
    return _matmul_all_gather_topo(
        x_shape_dtype,
        w_shape_dtype,
        mesh,
        tuple(axes),
        bn=bn,
        out_dtype=out_dtype,
        w_transposed=w_transposed,
        collective_id=collective_id,
        double_barrier=double_barrier,
        _return_local_call=True,
    )
