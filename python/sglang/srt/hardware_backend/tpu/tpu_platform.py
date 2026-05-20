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
        # Overlap scheduling on TPU uses the thread-based path in
        # ``hardware_backend.tpu.scheduler_overlap_mixin`` /
        # ``tp_worker_overlap_thread`` — NOT SGLang's CUDA-stream overlap,
        # which hangs because ``torch.tpu.Stream`` is a stub. Leave it
        # enabled by default; users can still opt out via
        # ``--disable-overlap-schedule``.
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

        When the model exposes the standard SGLang split of a transformer
        backbone (``model.model``) and a logits processor
        (``model.logits_processor``), we compile each as a **separate**
        ``torch.compile`` region and leave the outer ``ForCausalLM.forward``
        as an uncompiled orchestrator. This mirrors torchtpu-vllm's pattern
        where ``compute_selected_logits`` is its own
        ``@torch.compile(backend="tpu", fullgraph=True)`` region distinct
        from the model backbone (see
        ``torchtpu-vllm/tpu_inference/runner/tpu_runner.py``). Fusing both
        into a single compiled graph triggers a SparseCore halt
        (``tt_jit_logits_processor`` UserFatal) on Qwen3-32B at TP=8 with
        concurrency=64 because an ``aten.to`` near the final RMSNorm +
        LM-head matmul is offloaded to SparseCore.

        Falling back to the legacy single-graph compile happens for models
        that don't expose the ``(model, logits_processor)`` pair (e.g.
        pure-embedding models). ``forward_split_prefill`` is left
        uncompiled in the split path — its layer-by-layer Python loop
        already breaks the graph anyway, and standard prefill flows
        (chunked_prefill_size ≥ chunk size) don't take that path.
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

        # SGLang's batch shape varies per request (decode-1, prefill-N for
        # several Ns, mixed). With ``dynamic=False`` each distinct shape is
        # a fresh dynamo specialisation, which quickly blows past the
        # default ``cache_size_limit=8``. Bump it to something generous
        # enough to cover the common bucket counts so the scheduler does
        # not crash with FailOnRecompileLimitHit during steady-state
        # serving. A future change should switch the model wrap to
        # ``dynamic=True`` and rely on symbolic-shape specialisation
        # instead, but until that's plumbed through (see Pallas kernel
        # block-size selection) raising the limit is the safest knob.
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 256
        )

        # Pre-warm the Pallas fused-MoE op cache for every distinct
        # ``(topk, activation)`` combo this model uses. ``jax_op`` does
        # Python-introspection-heavy work (``inspect.signature``) at
        # registration time that dynamo cannot trace; mirrors the eager
        # RPA v3 registration in ``PallasAttnBackend.__init__``. Once the
        # cache is warm, the in-graph call is a cheap dict lookup followed
        # by a torch.library op invocation, both of which dynamo handles.
        try:
            from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
            from sglang.srt.layers.moe.pallas.fused_moe_op import (
                prewarm_fused_moe_op,
            )

            seen: set[tuple[int, str]] = set()
            for module in model.modules():
                if not isinstance(module, FusedMoE):
                    continue
                topk = int(getattr(module, "top_k", 0))
                activation = str(
                    getattr(module.moe_runner_config, "activation", "silu")
                )
                key = (topk, activation)
                if key in seen or topk <= 0:
                    continue
                seen.add(key)
                prewarm_fused_moe_op(topk=topk, activation=activation)
            if seen:
                logger.info(
                    "Pre-warmed fused-MoE op cache for %s.",
                    sorted(seen),
                )
        except ImportError:
            # FusedMoE or pallas MoE op not importable on this build —
            # dense-only model, nothing to pre-warm.
            pass

        from sglang.srt.hardware_backend.tpu.compile_shape_buckets import (
            make_token_bucketed_forward,
        )

        backbone = getattr(model, "model", None)
        lp = getattr(model, "logits_processor", None)
        can_split = (
            isinstance(backbone, torch.nn.Module)
            and isinstance(lp, torch.nn.Module)
            and hasattr(model, "forward")
        )

        wrapped = []
        if can_split:
            # Compile the transformer backbone and the LogitsProcessor as
            # two independent torch.compile regions (separate HLO graphs).
            setattr(
                backbone,
                "forward",
                torch.compile(
                    backbone.forward,
                    backend="tpu",
                    fullgraph=True,
                    dynamic=False,
                ),
            )
            # Compile LP, then wrap with K-axis bucket padding so dynamo
            # specialises per ``K_BUCKETS`` (8, 16, 32, 64, ...) instead
            # of per distinct ``extend_seq_lens.shape[0]`` value. Mirrors
            # torchtpu-vllm's ``_get_req_paddings`` rooted at MIN_NUM_SEQS=8.
            from sglang.srt.hardware_backend.tpu.compile_shape_buckets import (
                make_k_bucketed_lp_forward,
            )

            compiled_lp = torch.compile(
                lp.forward,
                backend="tpu",
                fullgraph=True,
                dynamic=False,
            )
            setattr(lp, "forward", make_k_bucketed_lp_forward(compiled_lp))
            # The outer ForCausalLM.forward is *not* compiled; it just
            # orchestrates the two compiled regions and runs token-axis
            # bucket padding around them. The inner regions see padded
            # shapes; the bucket wrapper slices the LogitsProcessorOutput
            # back to the real token count for decode mode.
            outer = getattr(model, "forward")
            setattr(model, "forward", make_token_bucketed_forward(outer))
            wrapped.append("forward(split: model + logits_processor)")
        else:
            # Legacy single-graph path: compile the full forward(s) as one
            # graph. Used when a model doesn't expose the standard
            # (model, logits_processor) pair.
            for method_name in ("forward", "forward_split_prefill"):
                fn = getattr(model, method_name, None)
                if fn is None:
                    continue
                compiled = torch.compile(
                    fn, backend="tpu", fullgraph=True, dynamic=False,
                )
                setattr(model, method_name, make_token_bucketed_forward(compiled))
                wrapped.append(method_name)

        logger.info(
            "Wrapped %s.{%s} with torch.compile(backend='tpu', "
            "fullgraph=True, dynamic=False, cache_size_limit=%d) "
            "+ token-bucket padding.",
            type(model).__name__,
            ", ".join(wrapped),
            torch._dynamo.config.cache_size_limit,
        )

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        # The "pallas" backend wraps the Pallas v3 ragged-paged-attention
        # kernel (see sglang.srt.layers.attention.pallas_backend). Falling
        # back to "torch_native" is still possible via --attention-backend
        # for debugging.
        return "pallas"

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.hardware_backend.tpu.graph_runner.tpu_graph_runner import (
            TpuGraphRunner,
        )

        return TpuGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        # The Pallas backend requires a fused 5D KV cache layout that the
        # ragged-paged-attention kernel writes into directly. When the user
        # opts out of the Pallas backend (e.g. for debugging via
        # ``--attention-backend torch_native``) fall back to the standard
        # MHA pool, whose separate K / V buffers are what the torch-native
        # backend expects.
        from sglang.srt.server_args import get_global_server_args

        if get_global_server_args().attention_backend == "pallas":
            from sglang.srt.hardware_backend.tpu.kv_pool import (
                TpuFusedMHATokenToKVPool,
            )

            return TpuFusedMHATokenToKVPool

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

    # HBM sizes are per PJRT-addressable device (chiplet), not per package.
    # On v7 each chip has 2 cores and PJRT exposes each core as its own
    # device, so the per-rank usable HBM is half the package spec (192 GB
    # → 96 GB; libtpu reserves ~1.25 GB so XLA's effective budget is
    # 94.75 GB — round down to 95 to mirror the v5p entry's margin).
    # Using the 192 GB per-package number caused mem_fraction_static to
    # be applied to 2× the real budget, producing static KV-cache slabs
    # that exceeded HBM at compile time for any mf > ~0.4 with tp=8.
    _TPU_HBM_BYTES = {
        "TPU v4": 32 * (1 << 30),
        "TPU v5p": 95 * (1 << 30),
        "TPU v5e": 16 * (1 << 30),
        "TPU v6e": 32 * (1 << 30),
        "TPU v7": 95 * (1 << 30),
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
