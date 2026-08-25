# SPDX-License-Identifier: Apache-2.0
"""Topology-aware 2x2x2x2 hypercube routing schedule + physical-topo maps + N-tile pickers.

Pure-Python routing machinery (no jax) for the tp-intra-topo combine's output gather. The 16 cores of
a 2x2x2 TPU slice form a 2x2x2x2 hypercube: physical coord ``L`` in ``[0,16)`` with bit 0 = on-chip
core
(core-bridge), bits 1-3 = the 3 size-2 torus axes (ICI). Entries ``(link_mask, slot_xor)`` route by
``xor`` on the PHYSICAL coord; ``_physical_topo_maps`` reads each device's ``(coords, core_on_chip)``
so any raveled mesh ORDER works (hypercube, the serve's ring/snake).
"""

from __future__ import annotations

import itertools

from . import util

NUM_DEVICES = 16  # 8 chips x 2 on-chip cores
BRIDGE_MASK = 1  # flips the on-chip core bit
ICI_MASKS = (2, 4, 8)  # the 3 torus axes (x, y, z)
ANTIPODE_OFFSET = ICI_MASKS[0] | ICI_MASKS[1] | ICI_MASKS[2]  # 14 (all 3 ICI bits)

# Software-pipeline depth: a tile flows through 5 grid stages (matmul+seed+ICI step1 / step2 / step3 /
# bridge / drain); semaphores are ring-indexed by tile%_TOPO_RING. Must be >= 5.
_TOPO_RING = 5


def build_ici_schedule():
    """The 3-step neighbour-exchange ICI schedule. Step ``k`` (0-indexed) delivers every
    distance-``(k+1)`` piece once, load-balanced across the 3 links (perfect matching in step 2).
    Returns a list of 3 steps, each a sorted list of ``(link_mask, slot_xor)``."""
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
            link_mask = min(combo, key=lambda m: (link_load[m], m))
            link_load[link_mask] += 1
            step.append((link_mask, offset ^ link_mask))
        steps.append(sorted(step))
    return steps


def antipode_core_split():
    """Per-core routing for the final (antipode) step: each on-chip core pulls the farthest piece
    over a DIFFERENT ICI axis (2 links at half width). Returns
    ``{0: (link_mask, slot_xor), 1: (link_mask, slot_xor)}``."""
    off = ANTIPODE_OFFSET
    return {
        0: (ICI_MASKS[0], ICI_MASKS[0] ^ off),
        1: (ICI_MASKS[1], ICI_MASKS[1] ^ off),
    }


def _ici_offsets():
    """The 8 slot offsets a core gathers over ICI (same on-chip parity): OR of every subset of the 3
    ICI masks -> (0, 2, 4, 6, 8, 10, 12, 14)."""
    offs = {0}
    for r in range(1, len(ICI_MASKS) + 1):
        for combo in itertools.combinations(ICI_MASKS, r):
            o = 0
            for m in combo:
                o |= m
            offs.add(o)
    return tuple(sorted(offs))


def _flatten_ici_schedule():
    """Flatten build_ici_schedule() into steps of (link_mask, slot_xor, gidx) with a unique semaphore
    index per ICI entry; return (steps, n_ici)."""
    steps = []
    n_ici = 0
    for step in build_ici_schedule():
        entries = []
        for link_mask, slot_xor in step:
            entries.append((link_mask, slot_xor, n_ici))
            n_ici += 1
        steps.append(tuple(entries))
    return tuple(steps), n_ici


def _pick_bn(n: int, target_tiles: int = 8) -> int:
    """N-tile width giving ~``target_tiles`` tiles, each a multiple of 128 (MXU lane). Searches tile
    counts outward from ``target_tiles``; ~8 was the measured sweet spot for the flat broadcast.
    """
    order = sorted(
        range(1, 2 * target_tiles + 1), key=lambda g: (abs(g - target_tiles), g)
    )
    for g in order:
        if n % g == 0 and (n // g) % 128 == 0:
            return n // g
    return n


def _topo_bn(n, t, k, w_itemsize=2, out_itemsize=2):
    """Routed-gather N-tile width, tpu7x-tuned: target ~2 MB per gathered ``[t, bn]`` tile (small T ->
    fewer/bigger tiles; large T -> more/smaller). Capped so the streamed ``[?, bn]`` W-tile double
    buffer stays <=32 MB of VMEM. Returns a multiple of 128 dividing ``n``; falls back to ``_pick_bn``.
    """
    perf = (2 * 1024 * 1024) // (t * out_itemsize)  # ~2 MB per gathered [t, bn] tile
    vmem = (16 * 1024 * 1024) // (k * w_itemsize)  # W-tile double buffer <= 32 MB
    target = min(min(vmem, 2048), max(512, perf))
    for bn in (2048, 1024, 512, 256, 128):
        if bn <= target and n % bn == 0:
            return bn
    return _pick_bn(n)


def _physical_topo_maps(mesh, axis_names):
    """Raveled-position <-> physical-hypercube-coord permutation for the 16 devices sharded over
    ``axis_names``, from each device's ``(coords, core_on_chip)`` -- so the XOR routing is by PHYSICAL
    topology regardless of the mesh's raveled ORDER. Physical coord = rank in the ``(torus-coords,
    core)`` sort: bit 0 = on-chip core, bits 1-3 = the 3 size-2 torus axes.

    Returns ``(pos_to_phys, phys_to_pos)`` as length-16 int lists. Raises ``ValueError`` if the slice
    is not a resolvable 8-chip x 2-core 2x2x2 torus."""
    line = util.product_device_line(mesh, axis_names)
    if line is None or len(line) != NUM_DEVICES:
        raise ValueError(f"needs a resolvable {NUM_DEVICES}-device 2x2x2 slice.")
    coords = [tuple(d.coords) for d in line]
    cores = [int(d.core_on_chip) for d in line]
    chips = {}
    for c, co in zip(coords, cores):
        chips.setdefault(c, set()).add(co)
    if len(chips) != 8 or any(v != {0, 1} for v in chips.values()):
        raise ValueError("needs 8 physical chips x 2 on-chip cores.")
    axis_vals = [{c[d] for c in coords} for d in range(len(coords[0]))]
    prod = 1
    for v in axis_vals:
        prod *= len(v)
    if prod != 8 or any(len(v) > 2 for v in axis_vals):
        raise ValueError("needs a 2x2x2 torus (each torus axis size <= 2).")
    order = sorted(range(NUM_DEVICES), key=lambda p: (coords[p], cores[p]))
    phys_to_pos = order  # phys_to_pos[physical coord] = raveled position
    pos_to_phys = [0] * NUM_DEVICES
    for coord, p in enumerate(order):
        pos_to_phys[p] = coord
    return pos_to_phys, phys_to_pos


def _is_2x2x2_slice(mesh, axis_names):
    """True if ``axis_names``'s product is a routable 16-core 2x2x2 slice (any raveled order)."""
    try:
        _physical_topo_maps(mesh, axis_names)
        return True
    except Exception:  # noqa: BLE001 -- non-TPU / non-2x2x2 -> caller falls through
        return False
