# SPDX-License-Identifier: Apache-2.0
"""TPU platform for sglang-diffusion, backed by torch_tpu (PrivateUse1/PJRT).

All TPU behavior lives here, reached via the Platform hooks in
`interface.py`; torch_tpu imports are deferred so this module imports
without torch_tpu installed. One chip per process (chosen by the
slicebuilder env from `prepare_server_environment`; local index always 0).
torch_tpu latches RANK/LOCAL_RANK/WORLD_SIZE at import — hence
`worker_spawn_env`.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

logger = init_logger(__name__)

_MIB = 1024**2


class _DitEvictionContext:
    """Evicts DiTs to host during VAE decode: large DiTs plus decode
    activations can exceed HBM, and PJRT has no swappable allocator."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self._evicted: list[torch.nn.Module] = []

    def __enter__(self) -> _DitEvictionContext:
        if self._pipeline is None:
            return self
        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
            is_dit_component_name,
        )

        for name, module in self._pipeline.modules.items():
            if is_dit_component_name(name) and isinstance(module, torch.nn.Module):
                module.cpu()
                self._evicted.append(module)
        if self._evicted:
            torch.tpu.synchronize()
        else:
            logger.warning(
                "TPU decode eviction found no DiT-class component among %s; "
                "decode runs with all components resident and may exhaust "
                "HBM. Extend is_dit_component_name for this pipeline.",
                sorted(self._pipeline.modules.keys()),
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._evicted:
            torch.tpu.synchronize()
            device = TpuPlatform.get_local_torch_device()
            for module in self._evicted:
                module.to(device)
            # Drain restores: the next request's dynamo re-trace fakeifies
            # these weights, and pending lazy H2D yields bogus metadata.
            torch.tpu.synchronize()
            self._evicted.clear()


@contextlib.contextmanager
def _xprof_session(log_dir: str):
    """torch_tpu xprof capture, run alongside torch.profiler (which cannot
    see TPU device activity); writes a trace viewable with xprof/tensorboard.
    """
    from torch_tpu._internal.profiler import profiler_api

    with profiler_api.profile(
        activities=[
            profiler_api.ProfilerActivity.TPU,
            profiler_api.ProfilerActivity.CPU,
        ],
        on_trace_ready=profiler_api.xprof_trace_handler(dir_name=log_dir),
    ):
        try:
            yield
        finally:
            # Flush outstanding lazy work so device activity lands in the trace.
            torch.tpu.synchronize()


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    # torch_tpu registers the "tpu" dynamo backend on import.
    simple_compile_backend: str = "tpu"

    # ------------------------------------------------------------------
    # Device identity and memory
    # ------------------------------------------------------------------

    # Identity/memory queries must not initialize PJRT: the parent process
    # calls them before workers spawn, and the chips belong to the workers.

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        # One chip per process; the process-local index is always 0.
        return torch.device("tpu", 0)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # Static: torch.tpu.get_device_name() would initialize the runtime.
        return "tpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        if torch.tpu.is_initialized():
            _free, total = torch.accelerator.get_memory_info()
            return total
        # ponytail: pre-init static floor (tpu7x ~95 GiB); topology lookup
        # if smaller-HBM TPUs become targets.
        return 90 * (1 << 30)

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if not torch.tpu.is_initialized():
            return cls.get_device_total_memory() / (1 << 30)
        free, _total = torch.accelerator.get_memory_info()
        return free / (1 << 30)

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        if not torch.tpu.is_initialized():
            return 0.0
        return float(
            torch.accelerator.memory_stats().get("allocated_bytes.all.current", 0)
        )

    @classmethod
    def empty_cache(cls) -> None:
        pass  # PJRT owns device memory; there is no cache to release.

    @classmethod
    def reset_peak_memory_stats(cls) -> None:
        if torch.tpu.is_initialized():
            torch.accelerator.reset_peak_memory_stats()

    @classmethod
    def get_memory_snapshot(cls) -> MemorySnapshot:
        from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

        if not torch.tpu.is_initialized():
            return MemorySnapshot(
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_allocated_mb=0.0,
                peak_reserved_mb=0.0,
            )
        stats = torch.accelerator.memory_stats()
        # reserved_bytes is not populated by the TPU PJRT client (reads 0).
        return MemorySnapshot(
            allocated_mb=stats.get("allocated_bytes.all.current", 0) / _MIB,
            reserved_mb=stats.get("reserved_bytes.all.current", 0) / _MIB,
            peak_allocated_mb=stats.get("allocated_bytes.all.peak", 0) / _MIB,
            peak_reserved_mb=stats.get("reserved_bytes.all.peak", 0) / _MIB,
        )

    @classmethod
    def is_float64_supported(cls) -> bool:
        return False  # TPUs have no f64 compute units.

    @classmethod
    def get_index_dtype(cls) -> torch.dtype:
        # int64 is emulated; srt pins positions to int32 for the same reason.
        return torch.int32

    @classmethod
    def inference_mode(cls):
        # torch_tpu's lazy tensors do not support torch.inference_mode.
        return torch.no_grad()

    # ------------------------------------------------------------------
    # Process / distributed environment
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        return self.get_local_torch_device()

    def get_torch_distributed_backend_str(self) -> str:
        # tpu_dist needs exactly one device per process; a single-rank run
        # on a multi-chip host sees every chip and raises. srt does the
        # same: gloo for world size 1.
        if os.environ.get("WORLD_SIZE", "1") == "1":
            return "gloo"
        return "tpu_dist"  # torch_tpu c10d backend: in-graph deferred collectives

    @classmethod
    def prepare_server_environment(cls, *, world_size: int) -> None:
        # srt launcher helper: jax-env scrub, LIBTPU_INIT_ARGS overrides,
        # single-host slicebuilder env for the spawned workers.
        from sglang.srt.hardware_backend.tpu.patches import _prepare_tpu_env

        _prepare_tpu_env(world_size=world_size)

    @classmethod
    def worker_spawn_env(
        cls, *, rank: int, world_size: int, master_port: int
    ) -> dict[str, str]:
        # Latched via GetEnvOnce during the child's import of torch_tpu.
        return {
            "LOCAL_RANK": str(rank),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }

    @classmethod
    def init_worker_device_env(
        cls, *, local_rank: int, rank: int, world_size: int, master_port: int
    ) -> None:
        # Env overrides must precede the first device touch (deferred PJRT).
        from sglang.srt.hardware_backend.tpu.env_override import (
            apply_tpu_env_overrides,
        )

        apply_tpu_env_overrides()

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Chip selection comes from the slicebuilder env; local index is 0.
        torch.tpu.set_device(0)

        # Touch the device now so all ranks rendezvous at the same startup
        # point instead of mid-model-load.
        torch.zeros(1, device="tpu")

    # ------------------------------------------------------------------
    # Attention and compile
    # ------------------------------------------------------------------

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend in (None, AttentionBackendEnum.TPU_PALLAS_FLASH):
            logger.info("Using TPU pallas flash attention backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "tpu_pallas.TpuPallasAttentionBackend"
            )
        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "sdpa.SDPABackend"
            )
        raise ValueError(
            f"Attention backend {selected_backend} is not supported on TPU; "
            "supported backends: TPU_PALLAS_FLASH (default), TORCH_SDPA."
        )

    @classmethod
    def supports_op_level_torch_compile(cls) -> bool:
        return False  # the op decorators target inductor, which cannot lower to TPU

    @classmethod
    def supports_fsdp(cls) -> bool:
        # FSDP's _move_states_to_device does `tensor.data = tensor.to(device)`,
        # which torch_tpu rejects across tensor types (set_data mismatch).
        return False

    @classmethod
    def supports_vae_decode_torch_compile(cls) -> bool:
        # VAE frame loops feed caches with per-chunk shapes, which recompile
        # per shape on a static-shape backend; lazy decode is the validated
        # path.
        return False

    # wrap_forward_for_compile stays the identity default: spmd_safe marks
    # the forward no-split, forcing one program whose arguments (all
    # weights) exceed HBM at compile time on large models.

    # ------------------------------------------------------------------
    # Distributed capabilities
    # ------------------------------------------------------------------

    @classmethod
    def supports_p2p_communication(cls) -> bool:
        # tpu_dist has no send/recv (experimental ops only).
        return False

    @classmethod
    def use_functional_seq_all_to_all(cls) -> bool:
        # No CUDA-style streams; the functional collective stays traceable.
        return True

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        return False

    @classmethod
    def vae_decode_context(cls, pipeline: Any) -> Any:
        from sglang.multimodal_gen import envs

        if envs.SGLANG_DIFFUSION_TPU_DISABLE_DECODE_EVICTION:
            return contextlib.nullcontext()
        return _DitEvictionContext(pipeline)

    @classmethod
    def denoise_loop_context(cls) -> Any:
        # DEFER_AND_FUSE fuses each step's ops; scoped to the loop because
        # VAE decode regresses under it.
        from torch_tpu._internal.execution_mode import EagerMode, set_eager_mode

        from sglang.multimodal_gen import envs

        mode_name = envs.SGLANG_DIFFUSION_TPU_EAGER_MODE
        # EagerMode is a pybind11 enum: look up via __members__.
        mode = EagerMode.__members__.get(mode_name)
        if mode is None:
            valid = ", ".join(EagerMode.__members__)
            raise ValueError(
                f"SGLANG_DIFFUSION_TPU_EAGER_MODE={mode_name!r} is not a "
                f"torch_tpu EagerMode; valid values: {valid}."
            )
        return set_eager_mode(mode)

    @classmethod
    def on_denoise_step_end(cls) -> None:
        # Bound the lazy queue to one step; unbounded growth OOMs HBM.
        torch.tpu.synchronize()

    @classmethod
    def create_aux_profiler(cls, log_dir: str) -> Any | None:
        return _xprof_session(log_dir)
