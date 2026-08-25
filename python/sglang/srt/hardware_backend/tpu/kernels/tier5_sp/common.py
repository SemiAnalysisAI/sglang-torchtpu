# SPDX-License-Identifier: Apache-2.0
"""Small topology and sizing helpers shared by Tier-5 Pallas kernels."""

import functools

import jax
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cdiv(a: int, b: int) -> int:
    """Return ``ceil(a / b)``."""
    if b == 0:
        raise ValueError("divisor must be non-zero")
    return (a + b - 1) // b


def align_to(value: int, alignment: int) -> int:
    """Round ``value`` up to an ``alignment`` multiple."""
    return cdiv(value, alignment) * alignment


def get_vmem_size_bytes() -> int:
    """Return the current TPU's per-core VMEM capacity."""
    return pltpu.get_tpu_info().vmem_capacity_bytes


def hypercube_device_order(devices) -> list:
    """Order a 2x2x2 TPU slice by chip coordinates and on-chip core."""
    return sorted(
        devices, key=lambda device: (tuple(device.coords), int(device.core_on_chip))
    )


def is_2x2x2_topology(mesh) -> bool:
    """Return whether ``mesh`` contains eight chips with two cores each."""
    devices = list(np.asarray(mesh.devices, dtype=object).flat)
    if len(devices) != 16:
        return False

    cores_by_chip = {}
    for device in devices:
        coords = tuple(device.coords)
        if len(coords) != 3:
            return False
        cores_by_chip.setdefault(coords, set()).add(int(device.core_on_chip))
    if len(cores_by_chip) != 8:
        return False
    if any(cores != {0, 1} for cores in cores_by_chip.values()):
        return False
    return all(len({coords[dim] for coords in cores_by_chip}) == 2 for dim in range(3))


def make_hypercube_tp_mesh(axis_name: str = "tp"):
    """Build a one-dimensional TP16 mesh in physical hypercube order."""
    devices = jax.devices()
    if len(devices) != 16:
        raise ValueError(
            f"make_hypercube_tp_mesh expects 16 devices, got {len(devices)}"
        )
    return jax.sharding.Mesh(
        np.asarray(hypercube_device_order(devices)),
        (axis_name,),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )


def make_ring_tp_mesh(axis_name: str = "tp"):
    """Build a one-hop cyclic Gray-code TP16 mesh."""
    devices = jax.devices()
    if len(devices) != 16:
        raise ValueError(f"make_ring_tp_mesh expects 16 devices, got {len(devices)}")
    hypercube = hypercube_device_order(devices)
    ring = [hypercube[position ^ (position >> 1)] for position in range(16)]
    return jax.sharding.Mesh(
        np.asarray(ring),
        (axis_name,),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )


def local_barrier(left_neighbor_id, right_neighbor_id, double_barrier=True):
    """Synchronize with two mesh neighbors, optionally using a second barrier."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor_id in (left_neighbor_id, right_neighbor_id):
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=neighbor_id,
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)
    if double_barrier:

        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def second_barrier_fn(second_barrier):
            for neighbor_id in (left_neighbor_id, right_neighbor_id):
                pltpu.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=neighbor_id,
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(second_barrier, 2)

        second_barrier_fn()


__all__ = [
    "align_to",
    "cdiv",
    "get_vmem_size_bytes",
    "hypercube_device_order",
    "is_2x2x2_topology",
    "local_barrier",
    "make_hypercube_tp_mesh",
    "make_ring_tp_mesh",
]
