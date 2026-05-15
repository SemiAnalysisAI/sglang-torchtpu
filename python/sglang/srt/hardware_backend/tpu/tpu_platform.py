"""TPU platform backend for SGLang SRT using torch_tpu.

Implements the SRTPlatform interface so that SGLang can run inference
on Google Cloud TPUs via the torch_tpu (PyTorch privateuse1) backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TpuSRTPlatform(SRTPlatform):
    """SGLang SRT platform for Google Cloud TPUs via torch_tpu."""

    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    supported_quantization: list[str] = []

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        from sglang.srt.hardware_backend.tpu.patches import _prepare_tpu_env

        _prepare_tpu_env(server_args.tp_size)

        if server_args.dtype is None or server_args.dtype == "auto":
            server_args.dtype = "bfloat16"

        if server_args.attention_backend is None:
            server_args.attention_backend = self.get_default_attention_backend()

        server_args.disable_cuda_graph = True
        server_args.disable_piecewise_cuda_graph = True
        server_args.disable_overlap_schedule = True
        # `enable_torch_compile` is honored on TPU and wired up by
        # `maybe_install_torch_compile_on_model` below. Off by default
        # (the SGLang-wide default); opt in with `--enable-torch-compile`.

        if server_args.mem_fraction_static is None:
            server_args.mem_fraction_static = 0.40

    def maybe_install_torch_compile_on_model(self, model, server_args) -> None:
        """Wrap the backbone forward(s) with `torch.compile(backend="tpu", ...)`.

        Modeled on torchtpu-vllm, which compiles the model backbone via
        vLLM's @support_torch_compile mechanism with `simple_compile_backend
        = "tpu"`. SGLang has no equivalent platform-level compile path on
        TPU (its torch.compile wiring lives in CudaGraphRunner, which is
        disabled here), so we install the wrapper directly on the model.

        Reassigning the attribute on the instance shadows the class method
        for that instance only — every existing call site in ModelRunner
        (`self.model.<name>(...)`) picks up the compiled version
        automatically. Two entry points are wrapped:

        * ``forward`` — used by forward_decode / forward_extend / forward_idle.
        * ``forward_split_prefill`` — used by forward_split_prefill, when
          the model exposes it (e.g. qwen2, qwen3_moe, glm4, gemma).
        """
        from sglang.srt.hardware_backend.tpu.patches import (
            _register_tpu_compile_backend,
        )

        if not server_args.enable_torch_compile:
            return

        # Honor the same per-model opt-out that the CUDA path uses
        # (see ModelRunner.kernel_warmup): some Transformers-backend
        # models can't be safely compiled (e.g. dynamic RoPE scaling).
        if not getattr(model, "_can_torch_compile", True):
            logger.info(
                "Model %s reports _can_torch_compile=False; skipping "
                "torch.compile install on TPU.",
                type(model).__name__,
            )
            server_args.enable_torch_compile = False
            return

        # The "tpu" backend is registered as a side-effect of importing
        # `torch_tpu._internal.compile`. `apply_tpu_patches` does this in
        # `init_backend`, but model loading may happen on a different
        # process / before patches run in some test paths, so be defensive.
        _register_tpu_compile_backend()

        wrapped = []
        for method_name in ("forward", "forward_split_prefill"):
            fn = getattr(model, method_name, None)
            if fn is None:
                continue
            compiled = torch.compile(
                fn, backend="tpu", fullgraph=True, dynamic=False,
            )
            setattr(model, method_name, compiled)
            wrapped.append(method_name)

        logger.info(
            "Wrapped %s.{%s} with torch.compile(backend='tpu', "
            "fullgraph=True, dynamic=False).",
            type(model).__name__,
            ", ".join(wrapped),
        )

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        return "torch_native"

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.hardware_backend.tpu.graph_runner.tpu_graph_runner import (
            TpuGraphRunner,
        )

        return TpuGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_nsa_kv_pool_cls(self) -> type:
        raise NotImplementedError("NSA is not supported on TPU.")

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.hardware_backend.tpu.allocator_tpu import (
            TpuPagedTokenToKVPoolAllocator,
        )

        return TpuPagedTokenToKVPoolAllocator

    def get_compile_backend(self, mode: str | None = None) -> str:
        return "tpu"

    def get_piecewise_backend_cls(self) -> type:
        raise NotImplementedError(
            "TPU piecewise backend not yet implemented."
        )

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        return False

    def is_pin_memory_available(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        import torch_tpu  # noqa: F401

        from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches

        apply_tpu_patches()

    # ------------------------------------------------------------------
    # MultiPlatformOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        return "tpu"

    # ------------------------------------------------------------------
    # Device identity & operations (DeviceMixin)
    # ------------------------------------------------------------------

    _TPU_HBM_BYTES = {
        "TPU v4": 32 * (1 << 30),
        "TPU v5p": 95 * (1 << 30),
        "TPU v5e": 16 * (1 << 30),
        "TPU v6e": 32 * (1 << 30),
        "TPU v7": 192 * (1 << 30),
    }

    def get_device_total_memory(self, device_id: int = 0) -> int:
        name = self.get_device_name(device_id)
        hbm = self._TPU_HBM_BYTES.get(name)
        if hbm is not None:
            return hbm
        return 32 * (1 << 30)

    def get_current_memory_usage(
        self, device: Optional[torch.device] = None
    ) -> float:
        return 0.0

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("tpu", local_rank)

    def set_device(self, device: torch.device) -> None:
        torch.tpu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        try:
            from torch_tpu._internal.utils import hardware

            return hardware.get_tpu_device_name()
        except Exception:
            return "tpu"

    def get_device_uuid(self, device_id: int = 0) -> str:
        return f"tpu:{device_id}"

    def get_device_capability(self, device_id: int = 0):
        return None

    def synchronize(self) -> None:
        torch.tpu.synchronize()

    def empty_cache(self) -> None:
        pass

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        total = self.get_device_total_memory(device_id)
        used = int(self.get_current_memory_usage())
        return (total - used, total)

    def get_torch_distributed_backend_str(self) -> str:
        return "tpu_dist"

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.tpu.manual_seed_all(seed)
