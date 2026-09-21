# SPDX-License-Identifier: Apache-2.0
"""TPU platform backed by torch_tpu (PrivateUse1/PJRT). One device per worker."""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DEVICE_MODULE_SHIMS = {
    "empty_cache": lambda: None,
    "reset_peak_memory_stats": lambda: (
        torch.accelerator.reset_peak_memory_stats()
        if torch.tpu.is_initialized()
        else None
    ),
    "memory_allocated": lambda: _memory_stat("allocated_bytes.all.current"),
    "max_memory_allocated": lambda: _memory_stat("allocated_bytes.all.peak"),
    # gpu_worker sizes residency off reserved; PJRT has no caching allocator,
    # so reserved is allocated and the pool overhead it computes is zero.
    "memory_reserved": lambda: _memory_stat("allocated_bytes.all.current"),
    "max_memory_reserved": lambda: _memory_stat("allocated_bytes.all.peak"),
}


_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE = 4 * 1024 * 1024 * 1024


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _memory_stat(key: str) -> int:
    if not torch.tpu.is_initialized():
        return 0
    return int(torch.accelerator.memory_stats().get(key, 0))


def _install_device_module_shims() -> None:
    for name, fn in _DEVICE_MODULE_SHIMS.items():
        if not hasattr(torch.tpu, name):
            setattr(torch.tpu, name, fn)


# The launcher and benchmarks reach these through torch.get_device_module()
# without entering the worker hook. Every shim is is_initialized()-guarded.
if hasattr(torch, "tpu"):
    _install_device_module_shims()

_total_memory_cache: dict[int, int] = {}
_qwen_vl_rope_installed: bool = False
_tpu_spmd_installed: bool = False


def _move_tensor_to_cpu(value):
    return (
        value.to("cpu")
        if isinstance(value, torch.Tensor) and value.device.type != "cpu"
        else value
    )


def dispatch_rope_index_to_host_on_tpu(fn):
    """Run Qwen-VL get_rope_index on CPU when input_ids lives on TPU."""
    if getattr(fn, "_tpu_host_rope_wrapped", False):
        return fn

    @functools.wraps(fn)
    def _wrapped(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
        **kwargs,
    ):
        target_device = None
        for t in (input_ids, image_grid_thw, video_grid_thw, attention_mask):
            if isinstance(t, torch.Tensor) and t.device.type == "tpu":
                target_device = t.device
                break
        if target_device is None:
            return fn(
                self,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
                **kwargs,
            )
        pos_ids, deltas = fn(
            self,
            input_ids=_move_tensor_to_cpu(input_ids),
            image_grid_thw=_move_tensor_to_cpu(image_grid_thw),
            video_grid_thw=_move_tensor_to_cpu(video_grid_thw),
            second_per_grid_ts=_move_tensor_to_cpu(second_per_grid_ts),
            attention_mask=_move_tensor_to_cpu(attention_mask),
            **{k: _move_tensor_to_cpu(v) for k, v in kwargs.items()},
        )
        return pos_ids.to(target_device), deltas.to(target_device)

    _wrapped._tpu_host_rope_wrapped = True
    return _wrapped


def _install_qwen_vl_host_rope_dispatch() -> None:
    global _qwen_vl_rope_installed
    if _qwen_vl_rope_installed:
        return
    from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import Qwen2_5_VLModel
    from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel

    for cls in (Qwen2_5_VLModel, Qwen3VLModel):
        cls.get_rope_index = dispatch_rope_index_to_host_on_tpu(cls.get_rope_index)
    _qwen_vl_rope_installed = True


class _PreservingSpDict(dict):
    def copy(self):
        return _PreservingSpDict(self)

    def __setitem__(self, k, v):
        if k == "sp_text_sharded" and self.get("sp_text_sharded"):
            return
        super().__setitem__(k, v)


def _install_tpu_spmd_overrides() -> None:
    """Install lightweight TPU subclasses for SPMD/CFG/SP lockstep execution."""
    global _tpu_spmd_installed
    if _tpu_spmd_installed:
        return

    import sglang.multimodal_gen.configs.pipeline_configs.ernie_image as ernie_cfg
    import sglang.multimodal_gen.configs.pipeline_configs.qwen_image as qwen_cfg
    import sglang.multimodal_gen.runtime.models.dits.qwen_image as qwen_dit
    import sglang.multimodal_gen.runtime.pipelines_core.stages as stages_pkg
    import sglang.multimodal_gen.runtime.pipelines_core.stages.decoding as dec_mod
    import sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding as te_mod
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_classifier_free_guidance_world_size as get_cfg_world_size,
        get_world_group,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
        build_shard_plan,
        shard_like,
        should_shard_text,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
        StageParallelismType,
    )

    _orig_barrier = dist.barrier

    def _tpu_barrier(group=None, *args, **kwargs):
        if (
            group is None or group is dist.group.WORLD
        ) and model_parallel_is_initialized():
            return get_world_group().barrier()
        return _orig_barrier(group=group, *args, **kwargs)

    dist.barrier = _tpu_barrier

    class TpuDecodingStage(dec_mod.DecodingStage):
        parallelism_type = StageParallelismType.REPLICATED

        def forward(self, batch, server_args):
            out = super().forward(batch, server_args)
            if out.output is not None and out.output.device.type == "tpu":
                out.output = (
                    out.output.to("cpu", dtype=torch.float32)
                    if get_world_group().rank_in_group == 0
                    else None
                )
            return out

    class TpuTextEncodingStage(te_mod.TextEncodingStage):
        def forward(self, batch, server_args):
            batch = super().forward(batch, server_args)
            if not batch.do_classifier_free_guidance or (
                model_parallel_is_initialized() and get_cfg_world_size() <= 1
            ):
                return batch
            for i, (pos, neg) in enumerate(
                zip(batch.prompt_embeds, batch.negative_prompt_embeds or [])
            ):
                if pos is None or neg is None or pos.shape[1] == neg.shape[1]:
                    continue
                target = max(pos.shape[1], neg.shape[1])
                for embeds, masks in (
                    (
                        batch.prompt_embeds,
                        batch.prompt_embeds_mask or batch.prompt_attention_mask,
                    ),
                    (
                        batch.negative_prompt_embeds,
                        batch.negative_prompt_embeds_mask
                        or batch.negative_attention_mask,
                    ),
                ):
                    t = embeds[i]
                    pad = target - t.shape[1]
                    m = (
                        masks[i].to(dtype=torch.bool)
                        if masks and i < len(masks) and masks[i] is not None
                        else torch.ones(t.shape[:2], dtype=torch.bool, device=t.device)
                    )
                    if pad > 0:
                        embeds[i] = F.pad(t, (0, 0, 0, pad))
                        m = F.pad(m, (0, pad), value=False)
                    if masks is not None and i < len(masks):
                        masks[i] = m
            return batch

    class TpuQwenImagePipelineConfig(qwen_cfg.QwenImagePipelineConfig):
        def _prepare_text_conditioning(
            self, batch, encoder_index, text_seq_len, batch_size, *, negative=False
        ):
            if (
                batch_size == 1
                and batch.do_classifier_free_guidance
                and model_parallel_is_initialized()
                and get_cfg_world_size() > 1
            ):
                lens = self.require_text_seq_lens(
                    batch, encoder_index, negative=negative, expected_batch_size=1
                )
                mask = self._prepare_encoder_hidden_states_mask(
                    batch,
                    encoder_index,
                    lens,
                    text_seq_len,
                    1,
                    negative=negative,
                )
                return [text_seq_len], mask
            return super()._prepare_text_conditioning(
                batch, encoder_index, text_seq_len, batch_size, negative=negative
            )

    class TpuErnieImagePipelineConfig(ernie_cfg.ErnieImagePipelineConfig):
        @staticmethod
        def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
            if model_parallel_is_initialized() and get_cfg_world_size() > 1:
                txt_seq_lens = [max(txt_seq_lens)] * len(img_shapes)
            return ernie_cfg.ErnieImagePipelineConfig.get_freqs_cis(
                img_shapes, txt_seq_lens, rotary_emb, device, dtype
            )

    class TpuQwenImageTransformer2DModel(qwen_dit.QwenImageTransformer2DModel):
        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            encoder_hidden_states_mask=None,
            *args,
            freqs_cis=None,
            attention_kwargs=None,
            **kwargs,
        ):
            ehs = (
                encoder_hidden_states[0]
                if isinstance(encoder_hidden_states, list)
                else encoder_hidden_states
            )
            mask = (
                encoder_hidden_states_mask[0]
                if isinstance(encoder_hidden_states_mask, list)
                else encoder_hidden_states_mask
            )
            if ehs is not None and mask is not None and should_shard_text(ehs.shape[1]):
                plan = build_shard_plan(ehs.shape[1])
                ehs = shard_like(ehs, plan)
                mask = shard_like(mask, plan)
                encoder_hidden_states = (
                    [ehs] if isinstance(encoder_hidden_states, list) else ehs
                )
                encoder_hidden_states_mask = (
                    [mask] if isinstance(encoder_hidden_states_mask, list) else mask
                )
                if freqs_cis is not None:
                    freqs_cis = (freqs_cis[0], shard_like(freqs_cis[1], plan, dim=0))
                attention_kwargs = _PreservingSpDict(
                    {
                        **(attention_kwargs or {}),
                        "sp_text_sharded": True,
                        "sp_txt_pad": plan.pad,
                    }
                )
            return super().forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                *args,
                freqs_cis=freqs_cis,
                attention_kwargs=attention_kwargs,
                **kwargs,
            )

    dec_mod.DecodingStage = TpuDecodingStage
    stages_pkg.DecodingStage = TpuDecodingStage
    te_mod.TextEncodingStage = TpuTextEncodingStage
    stages_pkg.TextEncodingStage = TpuTextEncodingStage
    qwen_cfg.QwenImagePipelineConfig = TpuQwenImagePipelineConfig
    ernie_cfg.ErnieImagePipelineConfig = TpuErnieImagePipelineConfig
    qwen_dit.QwenImageTransformer2DModel = TpuQwenImageTransformer2DModel
    _tpu_spmd_installed = True


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    simple_compile_backend: str = "tpu"

    _worker_initialized: bool = False

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device("tpu", 0)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # Real kind ("TPU v7"); the query reads the host config, no PJRT init.
        try:
            from sglang.srt.hardware_backend.tpu.env_override import (
                apply_tpu_env_overrides,
            )

            apply_tpu_env_overrides()
            from torch_tpu._internal.utils import hardware

            return hardware.get_tpu_device_name() or "tpu"
        except Exception:
            return "tpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # Constant per device; server_args asks several times at startup.
        cached = _total_memory_cache.get(device_id)
        if cached is not None:
            return cached
        if torch.tpu.is_initialized():
            _free, total = torch.accelerator.get_memory_info(
                torch.device("tpu", device_id)
            )
            total = int(total)
        else:
            # Pre-init query for offload sizing; get_device_properties() would
            # initialize PJRT.
            from sglang.srt.hardware_backend.tpu.env_override import (
                apply_tpu_env_overrides,
            )

            apply_tpu_env_overrides()
            from torch_tpu._internal.utils import hardware

            total = hardware.get_hbm_bytes_per_device()
            if total is None:
                raise RuntimeError("No recognized TPU attached.")
        _total_memory_cache[device_id] = total
        return total

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group=None,
    ) -> float:
        total = cls.get_device_total_memory(device_id or 0)
        return (total - cls.get_current_memory_usage()) / (1 << 30)

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        if not torch.tpu.is_initialized():
            return 0.0
        free, total = torch.accelerator.get_memory_info(device)
        return float(total - free)

    @classmethod
    def is_float64_supported(cls) -> bool:
        # tpu7x: elementwise matches a host reference to ~1e-13, matmul
        # downcasts to ~1e-7.
        return True

    @classmethod
    def supports_distributed_p2p(cls) -> bool:
        # tpu_dist implements collectives only; send/recv raise.
        return False

    @classmethod
    def supports_device_tensor_pickle(cls) -> bool:
        # Reducing a PrivateUse1 storage aborts in tensor_to_buffer.cc.
        return False

    @classmethod
    def inference_mode(cls):
        # torch_tpu's lazy tensors do not support torch.inference_mode.
        return torch.no_grad()

    def get_device(self, local_rank: int) -> torch.device:
        return self.get_local_torch_device()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        if not cls._worker_initialized:
            cls._init_worker_process()
            cls._worker_initialized = True
        torch.tpu.set_device(device)

    @classmethod
    def _init_worker_process(cls) -> None:
        from sglang.srt.hardware_backend.tpu.patches import (
            _patch_set_device_tolerate_mismatch,
            _prepare_tpu_env,
            _register_aten_fallbacks,
        )

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault(
            "TPU_PREMAPPED_BUFFER_SIZE", str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE)
        )
        _prepare_tpu_env(world_size=_world_size(), resolve_single_host_topology=True)
        import torch_tpu  # noqa: F401
        import torch_tpu._internal.compile  # noqa: F401

        _patch_set_device_tolerate_mismatch()
        _register_aten_fallbacks()
        _install_device_module_shims()
        _install_qwen_vl_host_rope_dispatch()
        _install_tpu_spmd_overrides()
        torch.zeros(1, device="tpu")

    def get_torch_distributed_backend_str(self) -> str:
        # tpu_dist needs one addressable PJRT device; a lone worker has many.
        return "gloo" if _world_size() == 1 else "tpu_dist"

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend in (None, AttentionBackendEnum.TORCH_SDPA):
            logger.info("Using Torch SDPA backend.")
            return (
                "sglang.multimodal_gen.runtime.layers.attention.backends."
                "sdpa.SDPABackend"
            )
        raise ValueError(
            f"Attention backend {selected_backend} is not supported on TPU; "
            "the supported backend is TORCH_SDPA (default)."
        )

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        return False
