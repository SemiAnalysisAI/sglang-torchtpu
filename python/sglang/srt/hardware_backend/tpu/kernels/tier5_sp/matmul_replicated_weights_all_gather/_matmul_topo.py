# SPDX-License-Identifier: Apache-2.0
"""The topology-aware output gather for the fused ``all_gather(x) @ W`` (opt-in).

The flat broadcast in ``_matmul_broadcast.py`` sends every output tile DIRECTLY to all 15
peers, so a distance-``d`` peer costs ``d`` ICI hops; this variant instead routes each
per-N-tile gather through the 2x2x2x2 hypercube store-and-forward schedule (built below by
``build_ici_schedule`` + ``antipode_core_split``): a 3+3+1 ICI neighbour exchange + inherent
parity split + core-bridge plane swap + split antipode, so each shard crosses each link ~once
(~2x less ICI traffic). The routing is multi-step, so it is software-pipelined across N-tiles
(5 stages: matmul+seed+ICI-step1 / step2 / step3 / bridge / drain, one tile per stage) to hide
the step latency. Wins T>=256 (bandwidth-bound); flat wins T=128 (latency-bound). Works on ANY
16-core 2x2x2 slice: ``_physical_topo_maps`` reads the physical hypercube coord of each device
from its ``(coords, core_on_chip)``, and the kernel routes by that -- so the mesh's raveled ORDER
(hypercube, the serve's ring/snake, ...) does NOT matter (no hypercube-ordering requirement).

The routing schedule (constants + ``build_ici_schedule`` / ``antipode_core_split`` / the
``simulate_*`` host-side checkers) is PURE PYTHON -- no jax -- so it can be unit-tested off-TPU
(the ``Test{BuildIciSchedule,SimulateAllGather,AntipodeCoreSplit}`` classes in
``tests/kernels/test_matmul_all_gather.py``), decoupled from the Pallas kernel below that
consumes it. Reached via ``matmul_all_gather(..., gather_kind="topo")`` (the opt-in override in
the ``api.py`` dispatch).

Topology
--------
A 2x2x2 TPU slice is 8 chips wired as a 3-cube (each torus axis is 2-wide, so a ring is just
one link); each chip's 2 TensorCores are joined by a fast on-chip core-bridge. Adding the core
dimension makes the machine a 2x2x2x2 hypercube of 16 cores, numbered by hypercube coordinate
``L`` in ``[0, 16)``:

    bit 0 -> on-chip core   (neighbour L^1, the core-bridge)
    bit 1 -> torus x        (neighbour L^2, ICI)
    bit 2 -> torus y        (neighbour L^4, ICI)
    bit 3 -> torus z        (neighbour L^8, ICI)

so physical neighbours are cheap ``xor`` masks on the PHYSICAL coord ``L`` and the schedule is
symmetric -- every core runs the identical entry list. An entry ``(link_mask, slot_xor)`` means:
the device at physical coord ``L`` sends its slot ``L ^ slot_xor`` to neighbour ``L ^ link_mask``
(the receiver stores it at the same global slot id), so it gains the piece at offset
``link_mask ^ slot_xor``. The kernel maps ``L`` to/from the raveled position (data-slot / output
block id) via ``_physical_topo_maps`` so this works on any mesh order.
"""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import util

P = jax.sharding.PartitionSpec


# ---------------------------------------------------------------------------
# Routing schedule -- PURE PYTHON (no jax); host-testable off-TPU.
# ---------------------------------------------------------------------------

NUM_DEVICES = 16  # 8 chips x 2 on-chip cores
BRIDGE_MASK = 1  # flips the on-chip core bit
ICI_MASKS = (2, 4, 8)  # the 3 torus axes (x, y, z)
ANTIPODE_OFFSET = ICI_MASKS[0] | ICI_MASKS[1] | ICI_MASKS[2]  # 14 (all 3 ICI bits)


def build_ici_schedule():
    """The 3-step neighbour-exchange ICI schedule.

    Returns 3 steps; step ``k`` (0-indexed) is a list of ``(link_mask, slot_xor)``
    entries delivering every distance-``(k+1)`` piece exactly once, load-balanced
    across the 3 links (a perfect matching in step 2).
    """
    steps = []
    for k in range(1, len(ICI_MASKS) + 1):
        link_load = {m: 0 for m in ICI_MASKS}
        targets = []
        for combo in itertools.combinations(ICI_MASKS, k):
            offset = 0
            for m in combo:
                offset |= m
            targets.append((offset, combo))
        targets.sort()
        step = []
        for offset, combo in targets:
            # Deliver over the least-loaded eligible link (a set bit of the
            # offset), so the sender forwards a piece one hop closer that it owns.
            link_mask = min(combo, key=lambda m: (link_load[m], m))
            link_load[link_mask] += 1
            step.append((link_mask, offset ^ link_mask))
        steps.append(sorted(step))
    return steps


def antipode_core_split():
    """Per-core routing for the final (antipode) step: one ICI axis per core.

    The farthest piece sits one hop from *every* neighbour, so each on-chip core
    can pull its half over a DIFFERENT ICI axis -- turning the final step from one
    physical link at full width into two links at half width. Returns
    ``{0: (link_mask, slot_xor), 1: (link_mask, slot_xor)}`` (core 0 uses axis 0,
    core 1 uses axis 1).
    """
    off = ANTIPODE_OFFSET
    return {
        0: (ICI_MASKS[0], ICI_MASKS[0] ^ off),
        1: (ICI_MASKS[1], ICI_MASKS[1] ^ off),
    }


def simulate_ici_phase(strict=False, _return_counts=False):
    """Simulate the ICI phase on all 16 cores; return per-device slot sets.

    The final (antipode) step is split across two ICI axes -- one per on-chip
    core -- exactly as the kernel runs it. With ``strict=True`` raises if any step
    forwards a slot the sender does not yet own. With ``_return_counts=True``
    returns device 0's cumulative slot count after each step instead.
    """
    schedule = build_ici_schedule()
    split = antipode_core_split()
    buffers = {d: {d} for d in range(NUM_DEVICES)}
    counts = []
    for step_idx, step in enumerate(schedule):
        is_last = step_idx == len(schedule) - 1
        # Sends within a step read the pre-step buffers; apply after the step.
        deliveries = []
        for D in range(NUM_DEVICES):
            entries = [split[D & 1]] if is_last else step
            for link_mask, slot_xor in entries:
                slot = D ^ slot_xor
                if strict and slot not in buffers[D]:
                    raise AssertionError(
                        f"device {D} tried to forward slot {slot} it does not own "
                        f"(entry {(link_mask, slot_xor)})"
                    )
                deliveries.append((D ^ link_mask, slot))
        for dest, slot in deliveries:
            buffers[dest].add(slot)
        counts.append(len(buffers[0]))
    return counts if _return_counts else buffers


def simulate_ici_counts():
    """Cumulative slots held by a device (including its own) after each ICI step."""
    return simulate_ici_phase(_return_counts=True)


def simulate_full():
    """ICI phase followed by the core-bridge exchange."""
    buffers = simulate_ici_phase()
    snapshot = {d: set(slots) for d, slots in buffers.items()}
    for D in range(NUM_DEVICES):
        buffers[D] |= snapshot[D ^ BRIDGE_MASK]
    return buffers


# ---------------------------------------------------------------------------
# Pallas kernel -- consumes the schedule above.
# ---------------------------------------------------------------------------


def _ici_offsets():
    """The 8 slot offsets a core gathers over ICI (same on-chip parity): OR of every
    subset of the 3 ICI masks -> (0, 2, 4, 6, 8, 10, 12, 14)."""
    offs = {0}
    for r in range(1, len(ICI_MASKS) + 1):
        for combo in itertools.combinations(ICI_MASKS, r):
            o = 0
            for m in combo:
                o |= m
            offs.add(o)
    return tuple(sorted(offs))


def _flatten_ici_schedule():
    """Flatten build_ici_schedule() into steps of (link_mask, slot_xor, gidx) with a
    unique semaphore index per ICI entry; return (steps, n_ici)."""
    steps = []
    n_ici = 0
    for step in build_ici_schedule():
        entries = []
        for link_mask, slot_xor in step:
            entries.append((link_mask, slot_xor, n_ici))
            n_ici += 1
        steps.append(tuple(entries))
    return tuple(steps), n_ici


# Software-pipeline depth: a tile flows through 5 grid stages (a0 matmul+seed+ICI
# step1, a1 step2, a2 step3, a3 bridge, a4 drain), so up to 5 tiles have routing in
# flight; semaphores are ring-indexed by tile%_TOPO_RING to keep concurrent tiles
# from aliasing. Must be >= 5.
_TOPO_RING = 5


def _mm_ag_2x2x2_topo_kernel(
    pos2phys_ref,  # SMEM [16] i32  raveled position -> physical hypercube coord
    phys2pos_ref,  # SMEM [16] i32  physical hypercube coord -> raveled position
    x_ref,  # VMEM [m_local, k]
    w_ref,  # VMEM [k, bn]
    x_scale_ref,  # VMEM [m_local, 1] per-token online-FP8 scale, or None (no scale)
    o_hbm_ref,  # HBM [m, n]
    o_vmem,  # VMEM [2, m_local, bn]  double-buffered matmul result
    seed_sems,  # DMA [RING]           seed self-copy (VMEM -> own HBM rows)
    ici_send_sems,  # DMA [RING, n_ici]
    ici_recv_sems,  # DMA [RING, n_ici]
    br_send_sems,  # DMA [RING, n_bridge]  core-bridge send
    br_recv_sems,  # DMA [RING, n_bridge]  core-bridge recv
    *,
    steps,
    all_masks,
    ici_offsets,
    antipode_split,
    axis_names: tuple[str, ...],
    strides: tuple[int, ...],
    sizes: tuple[int, ...],
    mesh_axis_names: tuple[str, ...],
    w_transposed: bool,
    m_local: int,
    bn: int,
    grid_n: int,
    ring: int,
    double_barrier: bool,
    do_gather: bool,
):
    # Software-pipelined store-and-forward: each grid step advances up to 5 tiles,
    # one per routing stage, so a tile's ICI/bridge DMAs overlap the next tiles'
    # matmul + W-read. Stage `a` operates on tile `i-a`. See README for the schedule.
    i = pl.program_id(0)  # 0 .. grid_n + 4
    # Raveled position (data-slot / output-block id, set by the mesh axis order) vs this device's
    # PHYSICAL hypercube coord (bit0 = on-chip core, bits1-3 = the 3 size-2 torus axes). They are
    # EQUAL only on a hypercube-ordered mesh; `pos2phys`/`phys2pos` (built from the device coords)
    # bridge them, so the XOR routing below is ALWAYS in physical space -- correct 1-hop DMAs on
    # ANY mesh order (hypercube, the serve's ring/snake, ...). No hypercube-ordering requirement.
    my_pos = util.combined_axis_index(axis_names, strides)
    my_phys = pos2phys_ref[my_pos]
    _peer = util.make_peer_id_fn(axis_names, strides, sizes, mesh_axis_names)
    last_step = len(steps) - 1  # the antipode step (split across 2 ICI axes)

    def _tile(g, ti):  # raveled-position g's rows, tile ti's N-slice, in HBM
        return o_hbm_ref.at[pl.ds(g * m_local, m_local), pl.ds(ti * bn, bn)]

    def _rav(x):  # raveled position of the physical neighbour reached by XOR mask `x`
        return phys2pos_ref[my_phys ^ x]

    def _ici(ti, buf, rs, si, entry):
        # The (reconstruct-identically) descriptor for one ICI entry of tile ti;
        # start() to post, wait() to drain. Step 1 forwards our own freshly-matmul'd
        # tile from VMEM; later steps forward an already-received slot from HBM.
        link_mask, slot_xor, gidx = entry
        if si == last_step:
            # Antipode step: the farthest piece is one hop from every neighbour, so
            # each on-chip core pulls it over a DIFFERENT ICI axis -- 2 links at half
            # width instead of 1 at full width. Delivered offset (link ^ sx) is the
            # antipode either way, so `gidx` and the downstream bridge are unchanged.
            la, xa, lb, xb = antipode_split
            parity = lax.rem(my_phys, jnp.int32(2))  # physical on-chip core (bit 0)
            link_mask = jnp.where(parity == 0, la, lb)
            slot_xor = jnp.where(parity == 0, xa, xb)
        sl = _rav(slot_xor)  # output block = raveled pos of the physical slot origin
        src = o_vmem.at[buf] if si == 0 else _tile(sl, ti)
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=_tile(sl, ti),
            send_sem=ici_send_sems.at[rs, gidx],
            recv_sem=ici_recv_sems.at[rs, gidx],
            device_id=_peer(_rav(link_mask)),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _bridge(ti, rs, bi, off):
        # Forward one same-parity slot of tile ti to the on-chip sibling (own slot
        # sourced from HBM -- the seed target -- so o_vmem is free after step 1).
        sl = _rav(off)
        return pltpu.make_async_remote_copy(
            src_ref=_tile(sl, ti),
            dst_ref=_tile(sl, ti),
            send_sem=br_send_sems.at[rs, bi],
            recv_sem=br_recv_sems.at[rs, bi],
            device_id=_peer(_rav(BRIDGE_MASK)),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def _seed_copy(ti, buf, rs):
        return pltpu.make_async_copy(
            src_ref=o_vmem.at[buf], dst_ref=_tile(my_pos, ti), sem=seed_sems.at[rs]
        )

    @pl.when(i == 0)
    def _setup():
        # topo only ever DMAs to its 4 direct physical neighbours (the 3 ICI masks +
        # the core-bridge), so barriering with exactly those peers is sufficient.
        util.hypercube_local_barrier(
            [_peer(_rav(m)) for m in all_masks], double_barrier=double_barrier
        )

    # ---- stage a0: tile i -- matmul, seed, ICI step 1 ----
    @pl.when(i < grid_n)
    def _a0():
        ti = i
        buf = lax.rem(ti, jnp.int32(2))
        rs = lax.rem(ti, jnp.int32(ring))
        # In-kernel requant: an fp8 x_ref stays fp8 in VMEM; reorder_matmul dequants it by the
        # per-token x_scale (raw fp8 -> f32 * scale -> W dtype) at the matmul, else just widens it.
        o_vmem.at[buf][...] = util.reorder_matmul(
            x_ref[...],
            w_ref[...],
            w_transposed=w_transposed,
            x_scale=x_scale_ref[...] if x_scale_ref is not None else None,
            preferred_element_type=jnp.float32,
        ).astype(o_vmem.dtype)
        _seed_copy(ti, buf, rs).start()
        if do_gather:
            for entry in steps[0]:
                _ici(ti, buf, rs, 0, entry).start()
        else:
            _seed_copy(ti, buf, rs).wait()

    if do_gather:
        # ---- stage a1: tile i-1 -- drain step1 + seed, start step2 ----
        # Waiting step1's sends + the seed here (not later) frees o_vmem[buf] before
        # tile i+1 reuses it at its own a0, so the double buffer stays race-free.
        @pl.when(jnp.logical_and(i >= 1, (i - 1) < grid_n))
        def _a1():
            ti = i - 1
            buf = lax.rem(ti, jnp.int32(2))
            rs = lax.rem(ti, jnp.int32(ring))
            # wait_send frees o_vmem[buf] (step 1 sources from VMEM, and the buffer is
            # reused by tile ti+2's matmul at its a0) -- a recv-only wait would NOT reap
            # the outgoing read; wait_recv gates step 2's read of the delivered dist-1 slots.
            for entry in steps[0]:
                op = _ici(ti, buf, rs, 0, entry)
                op.wait_send()
                op.wait_recv()
            _seed_copy(ti, buf, rs).wait()  # local copy: frees the seed's o_vmem read
            for entry in steps[1]:
                _ici(ti, buf, rs, 1, entry).start()

        # ---- stage a2: tile i-2 -- drain step2, start step3 ----
        @pl.when(jnp.logical_and(i >= 2, (i - 2) < grid_n))
        def _a2():
            ti = i - 2
            buf = lax.rem(ti, jnp.int32(2))
            rs = lax.rem(ti, jnp.int32(ring))
            for entry in steps[1]:
                op = _ici(ti, buf, rs, 1, entry)
                op.wait_send()  # reap the send sem (HBM src -- not reused, but keep sems clean)
                op.wait_recv()  # dist-2 slot present for step 3
            for entry in steps[2]:
                _ici(ti, buf, rs, 2, entry).start()

        # ---- stage a3: tile i-3 -- drain step3, start core-bridge swap ----
        @pl.when(jnp.logical_and(i >= 3, (i - 3) < grid_n))
        def _a3():
            ti = i - 3
            buf = lax.rem(ti, jnp.int32(2))
            rs = lax.rem(ti, jnp.int32(ring))
            for entry in steps[2]:
                op = _ici(ti, buf, rs, 2, entry)
                op.wait_send()
                op.wait_recv()  # antipode slot present for the bridge
            for bi, off in enumerate(ici_offsets):
                _bridge(ti, rs, bi, off).start()

        # ---- stage a4: tile i-4 -- drain the bridge (tile complete) ----
        @pl.when(jnp.logical_and(i >= 4, (i - 4) < grid_n))
        def _a4():
            ti = i - 4
            rs = lax.rem(ti, jnp.int32(ring))
            for bi, off in enumerate(ici_offsets):
                op = _bridge(ti, rs, bi, off)
                op.wait_send()
                op.wait_recv()  # opposite-parity slot landed -> our output is complete


def _topo_bn(n, t, k, w_itemsize=2, out_itemsize=2):
    """topo N-tile width, tpu7x-tuned (K4096 N8192 bf16): target ~2 MB per gathered N-tile
    (``[t, bn]``) so SMALL T uses fewer/bigger tiles (its cost is the depth-5 drain ramp + per-step
    overhead) and LARGE T uses more/smaller tiles (bandwidth-bound -> finer matmul/gather overlap).
    Beats the flat ``_pick_bn`` (~8 tiles) default -- measured (device us/iter): T256 bn2048 42.6 vs
    bn1024 46.1 (-7.6%), T512 2048 (-1%), T1024 1024 (tie), T2048/4096 512 (~-2%).

    ``out_itemsize`` sizes the gathered ``[t, bn]`` output tile (the ``out_dtype`` width -- 2 for a
    bf16 output even when ``x`` is fp8); ``w_itemsize`` sizes the streamed ``[?, bn]`` W-tile
    (the ``w`` width). The perf target is capped so the W-tile's double buffer stays <=32 MB of
    VMEM, so ``bn`` shrinks for a bigger ``k`` (down-proj K=8192) or a wider weight dtype (fp32
    itemsize=4) -- both OOM at bn=2048. Returns a multiple of 128 dividing ``n``; falls back to
    ``_pick_bn``.
    """
    perf = (2 * 1024 * 1024) // (
        t * out_itemsize
    )  # ~2 MB per gathered [t, bn] output tile
    vmem = (16 * 1024 * 1024) // (
        k * w_itemsize
    )  # keep the [?, bn] W-tile double buffer <= 32 MB
    target = min(min(vmem, 2048), max(512, perf))
    for bn in (2048, 1024, 512, 256, 128):
        if bn <= target and n % bn == 0:
            return bn
    from ._matmul_broadcast import _pick_bn

    return _pick_bn(n)


def _physical_topo_maps(mesh, axis_names):
    """Raveled-position <-> physical-hypercube-coord permutation for the 16 devices sharded over
    ``axis_names``, computed from each device's ``(coords, core_on_chip)`` -- so topo's XOR routing
    is by PHYSICAL topology regardless of the mesh's raveled ORDER (hypercube, the serve's
    ring/snake, anything). The physical coord is the device's rank in the ``(torus-coords, core)``
    sort: bit 0 = the on-chip core, bits 1-3 = the 3 size-2 torus axes, so ``coord ^ mask`` is a
    physical 1-hop (the same ordering ``common.tpu.make_hypercube_tp_mesh`` builds -- on which this
    permutation is the identity).

    Returns ``(pos_to_phys, phys_to_pos)`` as length-16 int lists. Raises ``ValueError`` if the
    slice is not a resolvable 8-chip x 2-core 2x2x2 torus (topo cannot route it -- e.g. a 4x2x2 or
    a partial slice), so callers can gate on it.
    """
    line = util.product_device_line(mesh, axis_names)
    if line is None or len(line) != NUM_DEVICES:
        raise ValueError(f"topo needs a resolvable {NUM_DEVICES}-device 2x2x2 slice.")
    coords = [tuple(d.coords) for d in line]
    cores = [int(d.core_on_chip) for d in line]
    chips = {}
    for c, co in zip(coords, cores):
        chips.setdefault(c, set()).add(co)
    if len(chips) != 8 or any(v != {0, 1} for v in chips.values()):
        raise ValueError("topo needs 8 physical chips x 2 on-chip cores.")
    axis_vals = [{c[d] for c in coords} for d in range(len(coords[0]))]
    prod = 1
    for v in axis_vals:
        prod *= len(v)
    if prod != 8 or any(len(v) > 2 for v in axis_vals):
        raise ValueError("topo needs a 2x2x2 torus (each torus axis size <= 2).")
    # Physical coord = rank in the (coords, core) sort (== hypercube_device_order): bit0 the core,
    # bits1-3 the 3 size-2 axes, so coord ^ {1,2,4,8} is a physical 1-hop.
    order = sorted(range(NUM_DEVICES), key=lambda p: (coords[p], cores[p]))
    phys_to_pos = order  # phys_to_pos[physical coord] = raveled position
    pos_to_phys = [0] * NUM_DEVICES
    for coord, p in enumerate(order):
        pos_to_phys[p] = coord
    return pos_to_phys, phys_to_pos


def _is_2x2x2_slice(mesh, axis_names):
    """True if ``axis_names``'s product is a routable 16-core 2x2x2 slice (topo can figure out the
    physical topology). Unlike the old hypercube-ORDER gate, ANY raveled order qualifies.
    """
    try:
        _physical_topo_maps(mesh, axis_names)
        return True
    except Exception:  # noqa: BLE001 -- non-TPU / non-2x2x2 -> caller falls through
        return False


def _matmul_all_gather_topo(
    x,
    w,
    mesh,
    axis_names,
    *,
    bn=None,
    out_dtype=None,
    x_scale=None,
    w_transposed=False,
    collective_id=0,
    double_barrier=True,
    do_gather=True,
    _return_local_call=False,
):
    """Topology-aware fused matmul-all-gather (software-pipelined hypercube routing).

    The ``gather_kind`` path of ``matmul_all_gather`` (the public dispatch in
    ``api.py`` routes here for the opt-in override AND the T>=256 auto-select).
    Per-N-tile hypercube store-and-forward (3+3+1 ICI + parity split + core-bridge swap +
    split antipode), pipelined across N-tiles so a tile's routing DMAs overlap the next
    tiles' matmul / W-read. Beats the flat broadcast, ``naive``, and the XLA reorder for T>=256
    (bandwidth-bound); flat wins T=128 (latency-bound). Works on ANY 16-core 2x2x2 slice -- it
    reads the physical topology from the device coords (``_physical_topo_maps``) and routes by
    physical hypercube coordinate, so the mesh's raveled ORDER (hypercube, the serve's ring/snake,
    ...) is irrelevant.

    Full-parity args (same result as the other paths): ``w_transposed`` accepts ``w`` as
    ``[N, K]`` (ColumnParallelLinear layout, sliced as natural ``[bn, K]`` row-tiles -- no
    transpose copy); ``out_dtype`` sets the output/gather dtype (e.g. an fp8 ``x`` -> a bf16
    output); ``x_scale`` (per-token ``[T]``) is passed INTO the kernel and dequantizes a raw fp8
    ``x`` in VMEM at the matmul (``reorder_matmul``), so ``x`` stays fp8 in VMEM -- the kernel is
    given a None ``x_scale_ref`` (the operand + BlockSpec are omitted) when there is none.
    """
    tp_size, strides, sizes = util.axis_tuple_strides(mesh, axis_names)
    if tp_size != NUM_DEVICES:
        raise ValueError(f"topo gather needs a {NUM_DEVICES}-core mesh, got {tp_size}.")
    mesh_axis_names = tuple(mesh.axis_names)
    # Physical topology from the device coords -- raveled position <-> physical hypercube coord.
    # Routing is by physical coord, so ANY mesh order works (raises if not a 2x2x2 slice).
    pos_to_phys, phys_to_pos = _physical_topo_maps(mesh, axis_names)
    m, k = x.shape
    n = w.shape[0] if w_transposed else w.shape[1]
    m_local = m // tp_size
    # x_scale (per-token online-FP8 row scale) is dequanted IN-KERNEL by reorder_matmul (a raw fp8
    # x stays fp8 in VMEM), not up front. When there is no scale the operand + its BlockSpec are
    # omitted and the kernel is given a None x_scale_ref (no dummy tensor). od defaults to bf16 when
    # a scale is present (fp8 x -> bf16 output), else x's dtype.
    has_x_scale = x_scale is not None
    od = (
        out_dtype
        if out_dtype is not None
        else (jnp.bfloat16 if has_x_scale else x.dtype)
    )
    if has_x_scale:
        # [m, 1] column -- the shape reorder_matmul + the Pallas VMEM ref expect.
        x_scale = x_scale.reshape(m, 1)
    if bn is None:
        bn = _topo_bn(n, m, k, w.dtype.itemsize, jnp.dtype(od).itemsize)
    if n % bn != 0:
        raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")
    grid_n = n // bn
    steps, n_ici = _flatten_ici_schedule()
    ici_offsets = _ici_offsets()
    n_bridge = len(ici_offsets)
    all_masks = ICI_MASKS + (BRIDGE_MASK,)
    # Antipode-step split (one ICI axis per on-chip core): (link0, sx0, link1, sx1).
    _split = antipode_core_split()
    (_l0, _x0), (_l1, _x1) = _split[0], _split[1]
    antipode_split = (_l0, _x0, _l1, _x1)
    ring = _TOPO_RING
    # The physical<->raveled permutation maps, passed as SMEM scalar-prefetch (dynamic-indexed
    # by the runtime device id inside the kernel), replicated across the mesh.
    pos2phys = jnp.asarray(pos_to_phys, jnp.int32)
    phys2pos = jnp.asarray(phys_to_pos, jnp.int32)

    # x_scale is an optional Pallas input: a None BlockSpec paired with a None operand makes the
    # kernel's x_scale_ref None (fixed kernel signature, no dummy tensor).
    x_scale_spec = (
        pl.BlockSpec((m_local, 1), lambda i, *_: (0, 0))
        if has_x_scale
        else None  # [m, 1]
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,  # pos2phys, phys2pos
        in_specs=[
            pl.BlockSpec((m_local, k), lambda i, *_: (0, 0)),
            # W N-tile: a [bn, K] row-slice of a [N, K] weight (w_transposed) or a [K, bn]
            # column-slice of a [K, N] weight -- both natural layouts, no transpose copy.
            (
                pl.BlockSpec((bn, k), lambda i, *_: (jnp.minimum(i, grid_n - 1), 0))
                if w_transposed
                else pl.BlockSpec(
                    (k, bn), lambda i, *_: (0, jnp.minimum(i, grid_n - 1))
                )
            ),
            x_scale_spec,
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=(
            pltpu.VMEM((2, m_local, bn), od),  # o_vmem (out_dtype, double-buffered)
            pltpu.SemaphoreType.DMA((ring,)),  # seed
            pltpu.SemaphoreType.DMA((ring, n_ici)),  # ici send
            pltpu.SemaphoreType.DMA((ring, n_ici)),  # ici recv
            pltpu.SemaphoreType.DMA((ring, n_bridge)),  # bridge send
            pltpu.SemaphoreType.DMA((ring, n_bridge)),  # bridge recv
        ),
        grid=(grid_n + 4,),
    )

    def _call(*operands):
        # operands = (xx, ww, p2ph, ph2p[, ss]); the pallas_call wants the 2 scalar-prefetch maps
        # first, then the blocked inputs (x, w, x_scale). ss (and its BlockSpec) is None when
        # there is no scale -> the kernel's x_scale_ref is None.
        xx, ww, p2ph, ph2p = operands[0], operands[1], operands[2], operands[3]
        ss = operands[4] if has_x_scale else None
        return pl.pallas_call(
            functools.partial(
                _mm_ag_2x2x2_topo_kernel,
                steps=steps,
                all_masks=all_masks,
                ici_offsets=ici_offsets,
                antipode_split=antipode_split,
                axis_names=axis_names,
                strides=strides,
                sizes=sizes,
                mesh_axis_names=mesh_axis_names,
                w_transposed=w_transposed,
                m_local=m_local,
                bn=bn,
                grid_n=grid_n,
                ring=ring,
                double_barrier=double_barrier,
                do_gather=do_gather,
            ),
            out_shape=jax.ShapeDtypeStruct((m, n), od),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=128 * 1024 * 1024,
            ),
            name=f"mm_ag_2x2x2_topo_bn{bn}_g{int(do_gather)}",
        )(p2ph, ph2p, xx, ww, ss)

    if _return_local_call:
        # Escape hatch for callers ALREADY inside a shard_map over `axis_names` -- the
        # fused-MoE body, which runs the unpooler + MoE-exit gather next to the
        # post-expert norm instead of as a separate model-level op.
        #
        # Nesting shard_map is illegal, but nothing here needs it: the gather is
        # neighbour DMAs inside the Mosaic kernel, not a JAX collective, so `_call` is a
        # plain pallas_call over LOCAL operands. Everything above is the static plan, so
        # the emitted program is identical to the wrapped path.
        #
        # Call as `_call(x_local, w, pos2phys, phys2pos)` with x_local [m // tp_size, k];
        # it returns this core's copy of the replicated [m, n], so the enclosing
        # shard_map's out_spec for it is P(None, None). `x`/`w` may be ShapeDtypeStructs.
        return _call, pos2phys, phys2pos

    sm_in_specs = [P(axis_names, None), P(None, None), P(None), P(None)]
    operands = [x, w, pos2phys, phys2pos]
    if has_x_scale:
        sm_in_specs.append(P(axis_names, None))
        operands.append(x_scale)

    return jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=tuple(sm_in_specs),
            out_specs=P(None, None),
            check_vma=False,
        )
    )(*operands)
