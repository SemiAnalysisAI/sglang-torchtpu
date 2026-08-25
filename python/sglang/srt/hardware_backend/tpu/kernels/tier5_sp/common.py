# SPDX-License-Identifier: Apache-2.0
"""Small topology and sizing helpers shared by Tier-5 Pallas kernels."""

import functools

import jax
import numpy as np
from jax import dtypes as jax_dtypes
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


def storage_bytes_for_elements(elements: int, dtype) -> int:
    """Return packed storage bytes, including for sub-byte JAX dtypes."""
    if elements < 0:
        raise ValueError("tensor element count must be nonnegative")
    return (elements * jax_dtypes.itemsize_bits(dtype) + 7) // 8


def estimate_activation_compute_cost(
    activation: str | None,
    elements: int,
    *,
    multiply_by_up: bool = False,
) -> tuple[int, int]:
    """Return ordinary FLOPs and special-function operations."""
    if elements < 0:
        raise ValueError("activation element count must be nonnegative")
    if activation is None:
        return 0, 0
    if activation == "sigmoid":
        flops_per_element, transcendentals_per_element = 3, 1
    elif activation == "silu":
        flops_per_element, transcendentals_per_element = 4, 1
    elif activation == "gelu":
        flops_per_element, transcendentals_per_element = 8, 1
    elif activation == "swigluoai":
        if not multiply_by_up:
            raise ValueError("swigluoai cost requires multiply_by_up=True")
        return 7 * elements, elements
    else:
        raise ValueError(f"unsupported activation {activation!r}")
    if multiply_by_up:
        flops_per_element += 1
    return elements * flops_per_element, elements * transcendentals_per_element


def cost_estimate_metadata(
    *,
    flops_basis: str,
    bytes_accessed_basis: str,
    remote_bytes_transferred: int | None = None,
    remote_bytes_transferred_basis: str | None = None,
    flops_kind: str | None = None,
    bytes_accessed_kind: str | None = None,
) -> dict[str, str]:
    """Return stable metadata describing a manual Pallas cost estimate."""
    metadata = {
        "cost_model_version": "v1",
        "cost_flops_basis": flops_basis,
        "cost_bytes_accessed_basis": bytes_accessed_basis,
        "cost_bytes_accessed_contract": "modeled_hbm_reads_and_writes",
    }
    if remote_bytes_transferred is not None:
        if remote_bytes_transferred < 0:
            raise ValueError("remote bytes transferred must be nonnegative")
        if remote_bytes_transferred_basis is None:
            raise ValueError("remote bytes transferred metadata requires a basis")
        metadata["cost_remote_bytes_transferred"] = str(remote_bytes_transferred)
    if remote_bytes_transferred_basis is not None:
        metadata["cost_remote_bytes_transferred_basis"] = remote_bytes_transferred_basis
    if flops_kind is not None:
        metadata["cost_flops_kind"] = flops_kind
    if bytes_accessed_kind is not None:
        metadata["cost_bytes_accessed_kind"] = bytes_accessed_kind
    return metadata


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
    "cost_estimate_metadata",
    "estimate_activation_compute_cost",
    "get_vmem_size_bytes",
    "hypercube_device_order",
    "is_2x2x2_topology",
    "local_barrier",
    "make_hypercube_tp_mesh",
    "make_ring_tp_mesh",
    "storage_bytes_for_elements",
]
