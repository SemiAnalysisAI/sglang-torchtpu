"""TPU compatibility patches for SGLang.

Ported from torchtpu-vllm/tpu_inference/platforms/tpu_platform.py.
Must run in every process (including spawned workers) because
module-level state is re-initialized on import in spawned processes.
All patches are idempotent.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

_DYNAMIC_COMPILE_TARGETS: list[tuple[str, str]] = [
    ("sglang.srt.layers.vocab_parallel_embedding", "get_masked_input_and_mask"),
]

_dynamic_compile_unwrapped = False


def _configure_eager_mode() -> None:
    from torch_tpu._internal import execution_mode

    execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)
    logger.info("TorchTPU eager mode set to DEFER_AND_FUSE.")


def _register_tpu_compile_backend() -> None:
    # Importing `torch_tpu._internal.compile` registers the "tpu" backend with
    # torch._dynamo so that `torch.compile(backend="tpu", ...)` works. Must run
    # before any code applies that decorator (e.g. on imports of layers/models).
    import torch_tpu._internal.compile  # noqa: F401


def _unwrap_dynamic_compile_fns() -> None:
    global _dynamic_compile_unwrapped
    if _dynamic_compile_unwrapped:
        return
    _dynamic_compile_unwrapped = True

    for module_path, fn_name in _DYNAMIC_COMPILE_TARGETS:
        try:
            mod = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError):
            continue
        fn = getattr(mod, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(mod, fn_name, fn.__wrapped__)
            logger.debug(
                "Unwrapped @torch.compile(dynamic=True) from %s.%s",
                module_path,
                fn_name,
            )


def _prepare_tpu_env(world_size: int = 1) -> None:
    """Set TORCH_TPU_* env vars needed by PjRt for multi-process init.

    Ported from torchtpu-vllm: _prepare_singlehost_tpu_env().
    Slicebuilder addresses must match exactly the number of TP workers
    (world_size), not the total process count — detokenizer and tokenizer
    processes do not participate in PjRt coordination.
    """
    if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
        return

    try:
        import portpicker

        sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
    except ImportError:
        import socket

        # Hold sockets open until env var is written so OS doesn't reassign.
        _held = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(world_size)]
        for s in _held:
            s.bind(("", 0))
        sb_ports = [s.getsockname()[1] for s in _held]
        for s in _held:
            s.close()

    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        f"localhost:{p}" for p in sb_ports
    )

    if world_size == 1:
        os.environ.setdefault("TORCH_TPU_TOPOLOGY", "1,1,1")
    else:
        try:
            from torch_tpu._internal.utils import hardware

            topo = hardware.get_tpu_topology(world_size)
            if topo:
                os.environ.setdefault("TORCH_TPU_TOPOLOGY", topo)
        except Exception:
            pass

    logger.info(
        "TPU env: TORCH_TPU_SLICEBUILDER_ADDRESSES=%s, TORCH_TPU_TOPOLOGY=%s",
        os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES"),
        os.environ.get("TORCH_TPU_TOPOLOGY"),
    )


_set_device_patched = False


def _patch_set_device_tolerate_mismatch() -> None:
    """Make torch.tpu.set_device tolerant of sglang/torch_tpu chiplet-index
    disagreements.

    sglang's ModelRunner.init_torch_distributed calls set_device(self.gpu_id).
    torch_tpu auto-binds each process to a chiplet at first PJRT init based
    on env vars set by run_scheduler_process; if the auto-bound chiplet
    index doesn't match sglang's gpu_id (e.g. on v7 where PJRT may assign
    chiplet 5 to local_rank=3 depending on slicebuilder coordination),
    torch_tpu raises "Cannot set TPU device to index X, current process is
    bound to device index Y". Accept the existing binding — chiplet
    assignment is owned by torch_tpu/PJRT, sglang's gpu_id is just a label.
    """
    global _set_device_patched
    if _set_device_patched:
        return
    _set_device_patched = True

    from torch_tpu._internal.device import _device_module

    orig_set_device = _device_module._DeviceModule.set_device

    def wrapped(cls, device):
        try:
            return orig_set_device(device)
        except ValueError as e:
            if "Cannot set TPU device" in str(e):
                idx = (
                    device.index
                    if hasattr(device, "index") and device.index is not None
                    else int(device)
                )
                logger.warning(
                    "TPU set_device(idx=%s) skipped: %s", idx, e,
                )
                return
            raise

    _device_module._DeviceModule.set_device = classmethod(wrapped)


_compute_position_torch_patched = False


def _patch_compute_position_torch() -> None:
    """Replace forward_batch_info.compute_position_torch with a CPU-staged
    variant that doesn't deadlock the torch_tpu device queue.

    Upstream iterates `for p, s in zip(extend_prefix_lens, extend_seq_lens)`
    and calls `torch.arange(p, p+s, device=tpu)` per element. Each iter mixes
    an implicit scalar materialization of the 0-dim TPU tensor with a
    device-side arange — interleaved sync-and-launch ops wedge the torch_tpu
    device queue after the first batch leaves pending work. Symptom: TP=8
    warmup completes, then all schedulers hang at this line on the next
    request.

    Fix: `.tolist()` both length tensors once (single explicit sync), build
    positions on CPU, transfer to device once. Behavior identical on CUDA/CPU
    (which doesn't use this path anyway — CUDA goes through the triton
    variant).
    """
    global _compute_position_torch_patched
    if _compute_position_torch_patched:
        return
    _compute_position_torch_patched = True

    import torch

    from sglang.srt.model_executor import forward_batch_info

    def compute_position_torch_cpu_staged(
        extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
    ):
        device = extend_prefix_lens.device
        prefix_lens = extend_prefix_lens.tolist()
        seq_lens = extend_seq_lens.tolist()
        positions_cpu = torch.cat(
            [torch.arange(p, p + s) for p, s in zip(prefix_lens, seq_lens)],
            axis=0,
        ).to(torch.int64)
        extend_start_loc = torch.zeros_like(extend_seq_lens)
        if extend_seq_lens.numel() > 1:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
        return positions_cpu.to(device), extend_start_loc

    forward_batch_info.compute_position_torch = compute_position_torch_cpu_staged


def apply_tpu_patches(world_size: int = 1) -> None:
    _prepare_tpu_env(world_size)
    _configure_eager_mode()
    _register_tpu_compile_backend()
    _unwrap_dynamic_compile_fns()
    _patch_set_device_tolerate_mismatch()
    _patch_compute_position_torch()
    logger.info("TPU patches applied.")
