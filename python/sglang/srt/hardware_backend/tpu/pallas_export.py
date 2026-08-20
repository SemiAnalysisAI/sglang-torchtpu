# SPDX-License-Identifier: Apache-2.0
"""Cloud TPU v7x Hardware Invariants & Pallas AOT Export Infrastructure.

This module encapsulates the proven physical hardware execution rules and AOT
export topology definitions derived from physical testing on Google Cloud TPU v7x
accelerators (Megachip / Torus 2x2x1 and 2x2x4 topologies).

===============================================================================
ARCHITECTURAL HARDWARE INVARIANTS FOR TPU v7x PALLAS KERNELS
===============================================================================

1. Hardware Optical Torus Decoder Ring (`tp_rank_to_device_id`):
   - `device_id={'model': target_device_id}` in `make_async_remote_copy` with
     `device_id_type=pltpu.DeviceIdType.MESH` addresses the physical optical torus
     routing engine directly rather than indexing through Python's `Mesh.devices`.
   - On Cloud TPU v7x topologies, logical ranks map to physical routing device IDs as:
       RANK/Process ID 0 -> device_id 0
       RANK/Process ID 1 -> device_id 1
       RANK/Process ID 2 -> device_id 4
       RANK/Process ID 3 -> device_id 5
       RANK/Process ID 4 -> device_id 6
       RANK/Process ID 5 -> device_id 7
       RANK/Process ID 6 -> device_id 2
       RANK/Process ID 7 -> device_id 3
   - To send from logical TP rank to destination TP rank `D`, the kernel must map
     through `TPU_V7X_TP_RANK_TO_DEVICE_ID = (0, 1, 4, 5, 6, 7, 2, 3)`:
       `target_device_id = tp_rank_to_device_id[D]`

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
   - In fused MoE kernels whose epilogue performs post-reduction (`sum(out_3d) * topk_weight`),
     all rows in `out_buf_ref` must be zero-initialized before peer ICI writes land,
     guaranteeing unwritten slots contribute exactly 0.0 to the token reduction sum.
"""

import os
import re
import sys
import threading
from contextlib import contextmanager

_TPU_INFO_PATCH_LOCK = threading.RLock()
_PALLAS_DEVICE_KIND_BY_TORCHTPU = {
    "TPU v6e": "TPU v6 lite",
    "TPU v7": "TPU7x",
}

# Physical optical torus routing decoder ring for Cloud TPU v7x
TPU_V7X_TP_RANK_TO_DEVICE_ID = (0, 1, 4, 5, 6, 7, 2, 3)
TPU_V7X_ORDERED_DEVICE_IDS = (0, 1, 4, 5, 6, 7, 2, 3)


class _ExportMockClient:
    """Mock PJRT client for abstract AOT compilation devices."""

    platform: str = "tpu"
    platform_version: str = "TPU v7"
    runtime_type: str = "tpu"

    def process_index(self) -> int:
        return 0

    def device_count(self) -> int:
        return 8

    def devices(self) -> list:
        return []

    def local_devices(self) -> list:
        return []


_MOCK_CLIENT = _ExportMockClient()


class ExportAbstractDevice:
    """Compile-only abstract device compatible with JAX mesh/sharding property inspection."""

    def __init__(
        self,
        device_kind: str,
        num_cores: int = 1,
        platform: str = "tpu",
        id: int = 0,
        process_index: int = 0,
    ):
        self.device_kind = device_kind
        self.num_cores = num_cores
        self.platform = platform
        self.id = id
        self.process_index = process_index
        self.client = _MOCK_CLIENT

    def __repr__(self) -> str:
        return f"AbstractDevice(id={self.id}, device_kind='{self.device_kind}', platform='{self.platform}')"

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
    """No-op stub for backward compatibility with sglang.srt.hardware_backend.tpu.__init__."""
    pass


def configure_jax_export_env(world_size: int = 8) -> None:
    """No-op stub for backward compatibility with sglang.srt.hardware_backend.tpu.__init__."""
    pass


def get_torchtpu_export_devices(world_size: int = 8):
    """Return the ordered abstract TPU devices for global partitions without calling jax.devices()."""
    device_kind = get_pallas_tpu_device_kind()
    abstract_devices = [
        ExportAbstractDevice(
            device_kind=device_kind,
            num_cores=1,
            platform="tpu",
            id=i,
            process_index=0,
        )
        for i in range(world_size)
    ]
    if world_size == 8:
        return tuple(abstract_devices[i] for i in TPU_V7X_ORDERED_DEVICE_IDS)
    return tuple(abstract_devices)


get_jax_export_devices = get_torchtpu_export_devices


def get_jax_mesh(
    world_size: int = 8,
    axis_names: tuple[str, ...] = ("data", "model"),
):
    """Return the logical TPU mesh using abstract devices with explicit optical ring ordering."""
    import numpy as np
    from jax.sharding import Mesh

    ordered_devices = np.asarray(get_torchtpu_export_devices(world_size))
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
