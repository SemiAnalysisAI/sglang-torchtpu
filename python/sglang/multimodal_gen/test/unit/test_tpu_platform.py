# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU platform. Cases needing torch_tpu self-skip."""

import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.platforms import tpu_platform_plugin
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform

_HAS_TORCH_TPU = hasattr(torch, "tpu")


class TestTpuPlatformPlugin(unittest.TestCase):
    def test_selected_when_torch_tpu_is_registered(self):
        with patch.object(torch, "tpu", object(), create=True):
            self.assertEqual(
                tpu_platform_plugin(),
                "sglang.multimodal_gen.runtime.platforms.tpu.TpuPlatform",
            )

    def test_not_selected_without_torch_tpu(self):
        # The wheel being installed is not enough; ``torch.tpu`` must exist.
        with patch.object(torch, "tpu", None, create=True):
            self.assertIsNone(tpu_platform_plugin())


class TestTpuPlatformSurface(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(TpuPlatform.device_type, "tpu")
        self.assertTrue(TpuPlatform.get_device_name().lower().startswith("tpu"))
        self.assertTrue(TpuPlatform().is_tpu())

    def test_float64_supported(self):
        self.assertTrue(TpuPlatform.is_float64_supported())

    def test_inference_mode_is_no_grad(self):
        self.assertIsInstance(TpuPlatform.inference_mode(), torch.no_grad)

    def test_attn_backend_default_is_sdpa(self):
        for selected in (None, AttentionBackendEnum.TORCH_SDPA):
            self.assertIn(
                "sdpa.SDPABackend",
                TpuPlatform.get_attn_backend_cls_str(selected, 128, torch.bfloat16),
            )

    def test_attn_backend_unsupported_raises(self):
        with self.assertRaises(ValueError):
            TpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.FA, 128, torch.bfloat16
            )

    def test_lone_worker_distributed_backend_is_gloo(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            self.assertEqual(TpuPlatform().get_torch_distributed_backend_str(), "gloo")

    def test_multi_worker_distributed_backend_is_tpu_dist(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "16"}):
            self.assertEqual(
                TpuPlatform().get_torch_distributed_backend_str(), "tpu_dist"
            )

    def test_p2p_is_unsupported(self):
        self.assertFalse(TpuPlatform.supports_distributed_p2p())

    def test_spatial_shard_decode_is_refused_without_p2p(self):
        from sglang.multimodal_gen.configs.models.vaes import base as vae_base
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

        platform = MagicMock()
        platform.supports_distributed_p2p.return_value = False
        with patch.object(vae_base, "current_platform", platform):
            self.assertFalse(
                vae_base.should_use_spatial_shard_parallel_decode(
                    WanVAEConfig(), world_size=2
                )
            )

    def test_spatial_shard_decode_survives_where_p2p_works(self):
        from sglang.multimodal_gen.configs.models.vaes import base as vae_base
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

        platform = MagicMock()
        platform.supports_distributed_p2p.return_value = True
        with patch.object(vae_base, "current_platform", platform):
            self.assertTrue(
                vae_base.should_use_spatial_shard_parallel_decode(
                    WanVAEConfig(), world_size=2
                )
            )

    def test_platform_defaults_leave_other_backends_alone(self):
        from sglang.multimodal_gen.runtime.platforms.interface import Platform

        self.assertTrue(Platform.supports_distributed_p2p())
        self.assertTrue(Platform.supports_device_tensor_pickle())
        self.assertFalse(Platform.requires_group_barrier())
        self.assertFalse(Platform.requires_uniform_spmd_shapes())
        self.assertFalse(Platform.decode_output_on_host())
        self.assertTrue(Platform.supports_varlen_attn_mask_meta())

    def test_tpu_spmd_and_attention_capabilities(self):
        self.assertTrue(TpuPlatform.requires_group_barrier())
        self.assertTrue(TpuPlatform.requires_uniform_spmd_shapes())
        self.assertTrue(TpuPlatform.decode_output_on_host())
        self.assertFalse(TpuPlatform.supports_varlen_attn_mask_meta())

    def test_qwen_and_ernie_keep_full_len_mask_under_uniform_cfg_seq_len(self):
        from sglang.multimodal_gen.configs.pipeline_configs.ernie_image import (
            ErnieImagePipelineConfig,
        )
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImagePipelineConfig,
        )

        qwen_cfg = QwenImagePipelineConfig()
        batch = MagicMock()
        batch.do_classifier_free_guidance = True
        batch.prompt_seq_lens = [[8]]
        batch.prompt_embeds_mask = [torch.ones((1, 8), dtype=torch.bool, device="cpu")]
        batch.prompt_embeds = [torch.zeros((1, 8, 16), device="cpu")]
        batch.height = 64
        batch.width = 64

        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.qwen_image.requires_uniform_cfg_seq_len",
            return_value=True,
        ):
            lens, mask = qwen_cfg._prepare_text_conditioning(
                batch, encoder_index=0, text_seq_len=8, batch_size=1
            )
        self.assertEqual(lens, [8])
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (1, 8))

        ernie_cfg = ErnieImagePipelineConfig()
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.ernie_image.requires_uniform_cfg_seq_len",
            return_value=True,
        ):
            cond_kwargs = ernie_cfg._prepare_cond_kwargs(
                batch,
                batch.prompt_embeds,
                rotary_emb=None,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
        self.assertIsNotNone(cond_kwargs["encoder_hidden_states_mask"])
        self.assertEqual(cond_kwargs["encoder_hidden_states_mask"].shape, (1, 8))

    def test_decoding_stage_is_replicated_when_uniform_spmd_shapes_required(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages import base as base_mod
        from sglang.multimodal_gen.runtime.pipelines_core.stages import (
            decoding as decoding_mod,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
            StageParallelismType,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
            DecodingStage,
        )

        server_args = MagicMock(enable_cfg_parallel=True)
        platform = MagicMock()
        platform.requires_uniform_spmd_shapes.return_value = True
        with (
            patch.object(base_mod, "get_global_server_args", return_value=server_args),
            patch.object(
                decoding_mod, "get_global_server_args", return_value=server_args
            ),
            patch.object(decoding_mod, "current_platform", platform),
        ):
            stage = DecodingStage(vae=MagicMock())
            self.assertEqual(stage.parallelism_type, StageParallelismType.REPLICATED)
            platform.requires_uniform_spmd_shapes.return_value = False
            with patch.object(stage, "_can_use_parallel_decode", return_value=False):
                self.assertEqual(
                    stage.parallelism_type, StageParallelismType.MAIN_RANK_ONLY
                )

    def test_device_tensor_pickle_is_unsupported(self):
        self.assertFalse(TpuPlatform.supports_device_tensor_pickle())

    def test_non_tpu_platform_runs_rope_index_directly(self):
        from sglang.multimodal_gen.runtime.models.encoders import qwen_vl_rope

        seen = []

        class _Model:
            @qwen_vl_rope.dispatch_rope_index_to_host_on_tpu
            def get_rope_index(self, input_ids, *_args, **_kwargs):
                seen.append(input_ids)
                return torch.zeros(3, 1, 4), torch.zeros(1)

        input_ids = torch.ones(1, 4, dtype=torch.long)
        platform = MagicMock()
        platform.is_tpu.return_value = False
        with patch.object(qwen_vl_rope, "current_platform", platform):
            pos, deltas = _Model().get_rope_index(input_ids)
        self.assertIs(seen[0], input_ids)
        self.assertEqual(pos.shape, (3, 1, 4))

    def test_tpu_platform_dispatches_rope_to_host_and_restores_device(self):
        from sglang.multimodal_gen.runtime.models.encoders import qwen_vl_rope

        seen_devices = []

        class _Model:
            @qwen_vl_rope.dispatch_rope_index_to_host_on_tpu
            def get_rope_index(self, input_ids, *_args, **_kwargs):
                seen_devices.append(input_ids.device)
                return torch.zeros(3, 1, 4), torch.zeros(1)

        target_device = torch.device("cpu")
        input_ids = torch.ones(1, 4, dtype=torch.long, device=target_device)
        platform = MagicMock()
        platform.is_tpu.return_value = True
        with patch.object(qwen_vl_rope, "current_platform", platform):
            pos, deltas = _Model().get_rope_index(input_ids)
        self.assertEqual(seen_devices[0], torch.device("cpu"))
        self.assertEqual(pos.device, target_device)
        self.assertEqual(deltas.device, target_device)

    def test_qwen2_5_vl_get_rope_index_with_video_grid_thw(self):
        from sglang.multimodal_gen.runtime.models.encoders import qwen_vl_rope
        from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import (
            Qwen2_5_VLModel,
        )

        dummy_self = MagicMock()
        dummy_self.config.vision_start_token_id = 151652
        dummy_self.config.image_token_id = 151655
        dummy_self.config.video_token_id = 151656
        dummy_self.config.vision_config.tokens_per_second = 2
        dummy_self.config.vision_config.spatial_merge_size = 1
        input_ids = torch.tensor(
            [[151652, 151656, 151656, 10]], dtype=torch.long, device="cpu"
        )
        attn_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long, device="cpu")
        video_grid_thw = torch.tensor([[2, 1, 1]], dtype=torch.long, device="cpu")
        platform = MagicMock()
        platform.is_tpu.return_value = True
        with patch.object(qwen_vl_rope, "current_platform", platform):
            pos_ids, deltas = Qwen2_5_VLModel.get_rope_index(
                dummy_self,
                input_ids=input_ids,
                video_grid_thw=video_grid_thw,
                attention_mask=attn_mask,
            )
        self.assertEqual(pos_ids.shape, (3, 1, 4))
        self.assertEqual(deltas.shape, (1, 1))

    def test_supporting_platforms_get_plain_pickle(self):
        import pickle

        from sglang.multimodal_gen.runtime.utils import distributed as dist_utils

        platform = MagicMock()
        platform.supports_device_tensor_pickle.return_value = True
        payload = [torch.arange(4, dtype=torch.float32), {"k": 1}]
        with patch.object(dist_utils, "current_platform", platform):
            self.assertEqual(dist_utils._dumps(payload), pickle.dumps(payload))

    def test_cpu_tensors_are_pickled_untouched(self):
        import pickle

        from sglang.multimodal_gen.runtime.utils import distributed as dist_utils

        platform = MagicMock()
        platform.supports_device_tensor_pickle.return_value = False
        payload = [torch.arange(6, dtype=torch.float32), "text", 7]
        with patch.object(dist_utils, "current_platform", platform):
            restored = pickle.loads(dist_utils._dumps(payload))
        self.assertTrue(torch.equal(restored[0], payload[0]))
        self.assertEqual(restored[1:], ["text", 7])

    def test_layerwise_offload_disabled(self):
        self.assertFalse(TpuPlatform.enable_dit_layerwise_offload_by_default())


@unittest.skipUnless(_HAS_TORCH_TPU, "torch.device('tpu') needs torch_tpu")
class TestWorkerInitOnce(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, TpuPlatform, "_worker_initialized", False)
        TpuPlatform._worker_initialized = False

    def test_init_runs_once_across_call_sites(self):
        with (
            patch.object(TpuPlatform, "_init_worker_process") as init,
            patch.object(torch, "tpu", MagicMock(), create=True),
        ):
            TpuPlatform.set_device(torch.device("tpu", 0))
            TpuPlatform.set_device(torch.device("tpu", 0))
        self.assertEqual(init.call_count, 1)

    def test_failed_init_is_retryable(self):
        with (
            patch.object(
                TpuPlatform, "_init_worker_process", side_effect=RuntimeError("boom")
            ),
            patch.object(torch, "tpu", MagicMock(), create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                TpuPlatform.set_device(torch.device("tpu", 0))
        self.assertFalse(TpuPlatform._worker_initialized)


@unittest.skipUnless(_HAS_TORCH_TPU, "torch_tpu not installed")
class TestTorchTpuIntegration(unittest.TestCase):
    def test_pre_init_memory_queries_do_not_initialize_pjrt(self):
        if torch.tpu.is_initialized():
            self.skipTest("PJRT already initialized in this process")
        self.assertGreater(TpuPlatform.get_device_total_memory(), 0)
        self.assertGreater(TpuPlatform.get_available_gpu_memory(), 0.0)
        self.assertEqual(TpuPlatform.get_current_memory_usage(), 0.0)
        self.assertTrue(TpuPlatform.get_device_name().startswith("TPU "))
        self.assertFalse(torch.tpu.is_initialized())

    def test_srt_symbols_the_platform_imports_exist(self):
        # tpu.py imports this lazily; a rename on the srt side must fail here,
        # not at worker bootstrap.
        from sglang.srt.hardware_backend.tpu.env_override import (
            apply_tpu_env_overrides,
        )
        from sglang.srt.hardware_backend.tpu.patches import (
            _prepare_tpu_env,
            _register_aten_fallbacks,
        )

        self.assertTrue(callable(apply_tpu_env_overrides))
        self.assertTrue(callable(_prepare_tpu_env))
        self.assertTrue(callable(_register_aten_fallbacks))

    def test_importing_the_platform_package_does_not_initialize_pjrt(self):
        # The parent resolves current_platform before spawning the worker;
        # PJRT is one client per process and the worker must be the one to own it.
        code = (
            "import torch, sglang.multimodal_gen.runtime.platforms as p; "
            "p.current_platform; import sglang.multimodal_gen.runtime.platforms.tpu; "
            "raise SystemExit(1 if torch.tpu.is_initialized() else 0)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])

    def test_worker_init_selects_cpu_only_jax_before_calling_srt(self):
        # A worker started without srt's entry-point activation in the parent
        # has no JAX_PLATFORMS; _prepare_tpu_env then raises once JAX is imported.
        from sglang.srt.hardware_backend.tpu import patches

        seen = {}

        def record(**kwargs):
            seen["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS")

        with (
            patch.dict(os.environ),
            patch.object(patches, "_prepare_tpu_env", side_effect=record),
        ):
            os.environ.pop("JAX_PLATFORMS", None)
            TpuPlatform._init_worker_process()
        self.assertEqual(seen.get("JAX_PLATFORMS"), "cpu")

    def test_device_module_shims_installed_once(self):
        from sglang.multimodal_gen.runtime.platforms.tpu import (
            _install_device_module_shims,
        )

        _install_device_module_shims()
        for name in (
            "empty_cache",
            "reset_peak_memory_stats",
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        ):
            self.assertTrue(hasattr(torch.tpu, name))
        torch.tpu.empty_cache()
        if not torch.tpu.is_initialized():
            self.assertEqual(torch.tpu.memory_allocated(), 0)


if __name__ == "__main__":
    unittest.main()
