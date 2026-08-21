# SPDX-License-Identifier: Apache-2.0
"""Cloud TPU v7x Hardware Invariants & Pallas AOT Export Infrastructure.

This module encapsulates the proven physical hardware execution rules and AOT
export topology definitions derived from physical testing on Google Cloud TPU v7x
accelerators (Megachip / Torus 2x2x1 and 2x2x4 topologies).

===============================================================================
ARCHITECTURAL HARDWARE INVARIANTS FOR TPU v7x PALLAS KERNELS
===============================================================================

1. Hardware Optical Torus Routing (`tp_rank_to_device_id`):
   - `device_id={'model': target_device_id}` in `make_async_remote_copy` with
     `device_id_type=pltpu.DeviceIdType.MESH` consumes the PJRT mesh device ID
     reported by TorchTPU for the destination rank.
   - Launcher rank assignment is not guaranteed to follow the physical torus.
     The route for each TP/EP rank must therefore be gathered from TorchTPU's
     per-device topology metadata after distributed initialization.
   - To send to destination TP rank `D`, the kernel maps through the cached
     rank-to-device-ID tensor passed as a scalar-prefetch argument.

2. Explicit DMA Lifecycle (`.start()` -> `.wait()`):
   - `make_async_copy` and `make_async_remote_copy` construct lazy DMA descriptors.
   - Every copy descriptor MUST be explicitly initiated via `.start()`
     (lowered to `llo::EnqueueDMAOp`) before invoking `.wait()` (lowered to
     `llo::DMADoneOp`). Calling `.wait()` without `.start()` causes sequencer
     sflag deadlock and hardware timeout.

3. Cross-Rank Collective Barrier Protocol:
   - All send DMAs must be enqueued via `.start()` and drained via `send_sem.wait()`
     BEFORE invoking `sync_barrier()`.
   - `sync_barrier()` performs an optical hardware semaphore barrier across all
     cores in the mesh axis.
   - All receive DMAs must be drained via `recv_sem.wait()` AFTER `sync_barrier()`.

4. Input Tensor Buffer Passing vs In-Kernel Instantiation:
   - Static mapping tables and decoder rings must NEVER be created with `jnp.array(...)`
     inside the kernel loop body (which forces dynamic allocation in SMEM/VMEM).
   - They must be registered as PyTorch tensor buffers on the model layer and passed
     through the Pallas argument pipeline as prefetch scalar arguments.

5. Multi-Iteration Semaphore Slotting:
   - For all-to-all transfers across N peers, DMA semaphores must be indexed
     per-peer slot (`.at[k]`) to prevent semaphore counter drift and starvation
     across successive kernel calls.

6. Zero-Initialization for Top-K Reductions:
   - In fused MoE kernels whose epilogue performs post-reduction
     (`sum(out_3d) * topk_weight`),
     all rows in `out_buf_ref` must be zero-initialized before peer ICI writes land,
     guaranteeing unwritten slots contribute exactly 0.0 to the token reduction sum.
"""

import threading
from contextlib import contextmanager
from typing import Any

_TPU_INFO_PATCH_LOCK = threading.RLock()
_PALLAS_DEVICE_KIND_BY_TORCHTPU = {
    "TPU v6e": "TPU v6 lite",
    "TPU v7": "TPU7x",
}

class _ExportMockClient:
    """Mock PJRT client for abstract AOT compilation devices."""

    platform: str = "tpu"
    platform_version: str = "TPU v7"
    runtime_type: str = "tpu"

    def __init__(self, device_count: int):
        self._device_count = device_count

    def process_index(self) -> int:
        return 0

    def device_count(self) -> int:
        return self._device_count

    def devices(self) -> list:
        return []

    def local_devices(self) -> list:
        return []


class ExportAbstractDevice:
    """Compile-only device supporting JAX mesh and sharding inspection."""

    def __init__(
        self,
        device_kind: str,
        num_cores: int = 1,
        platform: str = "tpu",
        id: int = 0,
        process_index: int = 0,
        client: _ExportMockClient | None = None,
    ):
        self.device_kind = device_kind
        self.num_cores = num_cores
        self.platform = platform
        self.id = id
        self.process_index = process_index
        self.client = client or _ExportMockClient(1)

    def __repr__(self) -> str:
        return (
            f"AbstractDevice(id={self.id}, device_kind='{self.device_kind}', "
            f"platform='{self.platform}')"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExportAbstractDevice):
            return False
        return (
            self.id == other.id
            and self.device_kind == other.device_kind
            and self.platform == other.platform
        )

    def __hash__(self) -> int:
        return hash((self.id, self.device_kind, self.platform))


def configure_jax_export_platform() -> None:
    """Keep the legacy initialization hook without forcing a JAX platform."""
    pass


def configure_jax_export_env(world_size: int = 8) -> None:
    """Keep the legacy initialization hook without creating CPU devices."""
    pass


def device_attributes_to_group_routes(
    device_attributes_by_group_rank: tuple[dict[str, Any], ...],
) -> tuple[int, ...]:
    """Return the Pallas DMA route ID reported by each TorchTPU group rank."""
    routes = tuple(
        int(attributes["id"]) for attributes in device_attributes_by_group_rank
    )
    if sorted(routes) != list(range(len(routes))):
        raise ValueError(
            "TPU route IDs must be a permutation of "
            f"range({len(routes)}), got {routes}"
        )
    slice_indices = {
        int(attributes.get("slice_index", 0))
        for attributes in device_attributes_by_group_rank
    }
    if len(slice_indices) != 1:
        raise ValueError("Fused MoE reduce-scatter requires a single TPU slice")
    return routes


def get_tpu_group_rank_to_device_id(
    group_ranks: tuple[int, ...],
    cpu_group,
) -> tuple[int, ...]:
    """Return the PJRT device ID addressed by each TorchTPU group rank."""
    import torch
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_resources

    route_cache = get_resources().tpu_group_routes
    cached = route_cache.get(group_ranks)
    if cached is not None:
        return cached

    local_attributes = dict(
        torch.get_device_module("tpu").get_local_device_attributes()
    )
    gathered_attributes = [None] * len(group_ranks)
    dist.all_gather_object(
        gathered_attributes,
        local_attributes,
        group=cpu_group,
    )
    routes = device_attributes_to_group_routes(tuple(gathered_attributes))
    route_cache[group_ranks] = routes
    return routes


def get_torchtpu_export_devices(
    rank_to_device_id: tuple[int, ...],
):
    """Return abstract TPU devices ordered consistently with TorchTPU ranks."""
    world_size = len(rank_to_device_id)
    if sorted(rank_to_device_id) != list(range(world_size)):
        raise ValueError(
            "Rank-to-device IDs must be a permutation of "
            f"range({world_size}), got {rank_to_device_id}"
        )

    device_kind = get_pallas_tpu_device_kind()
    client = _ExportMockClient(world_size)
    abstract_devices = [
        ExportAbstractDevice(
            device_kind=device_kind,
            num_cores=1,
            platform="tpu",
            id=i,
            process_index=0,
            client=client,
        )
        for i in range(world_size)
    ]
    return tuple(abstract_devices[i] for i in rank_to_device_id)


get_jax_export_devices = get_torchtpu_export_devices


def get_jax_mesh(
    rank_to_device_id: tuple[int, ...],
    axis_names: tuple[str, ...] = ("data", "model"),
):
    """Return an abstract TPU mesh using the runtime-derived rank ordering."""
    import numpy as np
    from jax.sharding import Mesh

    ordered_devices = np.asarray(
        get_torchtpu_export_devices(rank_to_device_id)
    )
    if len(axis_names) == 2:
        ordered_devices = ordered_devices.reshape(1, -1)
    return Mesh(ordered_devices, tuple(axis_names))


def get_torch_tpu_device_kind() -> str:
    """Return physical TPU hardware metadata."""
    from torch_tpu._internal.utils.hardware import get_tpu_device_name

    return get_tpu_device_name()


def get_pallas_tpu_device_kind() -> str:
    """Return the physical device spelling expected by JAX Pallas."""
    device_kind = get_torch_tpu_device_kind()
    return _PALLAS_DEVICE_KIND_BY_TORCHTPU.get(device_kind, device_kind)


@contextmanager
def tpu_info_for_cpu_export():
    """Ensure Pallas kernels receive physical TPU7x hardware metadata during export."""
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    with _TPU_INFO_PATCH_LOCK:
        original_public = pltpu.get_tpu_info
        original_private = tpu_info.get_tpu_info
        original_device_kind = getattr(tpu_info, "get_device_kind", None)
        original_is_tpu_device = getattr(tpu_info, "is_tpu_device", None)
        device_name = get_pallas_tpu_device_kind()
        chip_version = tpu_info.chip_version_from_device_kind(device_name)
        if chip_version is not None:
            info = tpu_info.get_tpu_info_for_chip(chip_version, 1)
            pltpu.get_tpu_info = lambda: info
            tpu_info.get_tpu_info = lambda: info
            tpu_info.get_device_kind = lambda: device_name
            tpu_info.is_tpu_device = lambda: True
        try:
            yield
        finally:
            pltpu.get_tpu_info = original_public
            tpu_info.get_tpu_info = original_private
            if original_device_kind is not None:
                tpu_info.get_device_kind = original_device_kind
            if original_is_tpu_device is not None:
                tpu_info.is_tpu_device = original_is_tpu_device
