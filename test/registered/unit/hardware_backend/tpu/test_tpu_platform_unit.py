"""Unit tests for TpuSRTPlatform.

These tests mock torch_tpu so they can run on any machine without TPU hardware.
"""

import builtins
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch.compiler.config as compiler_config

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.tpu_platform import (
    TpuSRTPlatform,
    _validate_decode_context_parallel,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# CPU suite: mocks torch_tpu, so it runs on the CPU runners -- never a TPU node.
register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTpuPlatformIdentity(CustomTestCase):
    """Platform identity and capability flags."""

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_enum_is_tpu(self):
        self.assertEqual(self.platform._enum, PlatformEnum.TPU)

    def test_device_name(self):
        self.assertEqual(self.platform.device_name, "tpu")
        self.assertEqual(self.platform.device_type, "tpu")

    def test_is_tpu(self):
        self.assertTrue(self.platform.is_tpu())
        self.assertFalse(self.platform.is_cuda())
        self.assertFalse(self.platform.is_npu())

    def test_supports_fp8_false(self):
        self.assertFalse(self.platform.supports_fp8())

    def test_pin_memory_unavailable(self):
        self.assertFalse(self.platform.is_pin_memory_available())

    def test_cuda_graph_disabled(self):
        self.assertFalse(self.platform.support_cuda_graph())
        self.assertFalse(self.platform.support_piecewise_cuda_graph())

    def test_pipeline_stages_may_replicate_experts(self):
        self.assertTrue(self.platform.supports_moe_dp_pipeline_stages())

    def test_dispatch_key(self):
        self.assertEqual(self.platform.get_dispatch_key_name(), "tpu")

    def test_compile_backend(self):
        self.assertEqual(self.platform.get_compile_backend(), "tpu")

    def test_distributed_backend(self):
        self.assertEqual(self.platform.get_torch_distributed_backend_str(), "tpu_dist")

    def test_default_attention_backend(self):
        self.assertEqual(
            self.platform.get_default_attention_backend(), "pallas_tpu_rpa_v3"
        )


class TestTpuPlatformInitialization(CustomTestCase):
    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_rejects_backend_initialization_without_worker_world_size(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"
            ) as prepare_tpu_env,
            patch(
                "sglang.srt.hardware_backend.tpu.patches.apply_tpu_patches"
            ) as apply_tpu_patches,
        ):
            os.environ.pop("WORLD_SIZE", None)
            with self.assertRaisesRegex(RuntimeError, "requires worker WORLD_SIZE"):
                self.platform.init_backend()

        prepare_tpu_env.assert_not_called()
        apply_tpu_patches.assert_not_called()

    def test_worker_world_size_prepares_environment_before_runtime_patches(self):
        calls = []
        fake_torch_tpu = ModuleType("torch_tpu")

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16"}, clear=False),
            patch.dict(sys.modules, {"torch_tpu": fake_torch_tpu}),
            compiler_config.patch(cache_key_tag=""),
            patch(
                "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env",
                side_effect=lambda world_size, **kwargs: calls.append(
                    ("prepare", world_size, kwargs)
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.patches.apply_tpu_patches",
                side_effect=lambda: calls.append(("patch", None)),
            ),
        ):
            self.platform.init_backend()

        self.assertEqual(
            calls,
            [
                (
                    "prepare",
                    16,
                    {"resolve_single_host_topology": True},
                ),
                ("patch", None),
            ],
        )


class TestTpuCompileCache(CustomTestCase):
    def test_pallas_source_change_updates_compile_cache_key(self):
        import sglang.srt.hardware_backend.tpu.compile_cache as compile_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            srt_root = Path(temp_dir)
            source_files = (
                srt_root / "layers/attention/pallas/kernel.py",
                srt_root / "layers/moe/pallas/kernel.py",
                srt_root / "hardware_backend/tpu/attention_pallas.py",
                srt_root / "hardware_backend/tpu/attention_pallas_rpa_base.py",
                srt_root / "hardware_backend/tpu/attention_pallas_rpa_batched.py",
                srt_root / "hardware_backend/tpu/attention_pallas_rpa_v3.py",
                srt_root / "hardware_backend/tpu/distributed_topk_pallas.py",
                srt_root / "hardware_backend/tpu/kernels/distributed_topk.py",
                srt_root / "hardware_backend/tpu/kernels/kernel.py",
                srt_root / "hardware_backend/tpu/pallas_export.py",
                srt_root / "hardware_backend/tpu/pallas_jax_op.py",
                srt_root / "hardware_backend/tpu/quantized_matmul.py",
            )
            for source_file in source_files:
                source_file.parent.mkdir(parents=True, exist_ok=True)
                source_file.write_text("original source\n")

            module_path = srt_root / "hardware_backend/tpu/compile_cache.py"
            with (
                patch.object(compile_cache, "__file__", str(module_path)),
                compiler_config.patch(cache_key_tag="user-tag"),
            ):
                compile_cache.add_pallas_kernel_sources_to_compile_cache_key()
                original_key = compiler_config.cache_key_tag

                distributed_top_k_source = (
                    srt_root / "hardware_backend/tpu/kernels/distributed_topk.py"
                )
                distributed_top_k_source.write_text("changed source\n")
                compile_cache.add_pallas_kernel_sources_to_compile_cache_key()
                changed_key = compiler_config.cache_key_tag

                self.assertTrue(original_key.startswith("user-tag|"))
                self.assertNotEqual(changed_key, original_key)


class TestTpuPlatformMemory(CustomTestCase):
    """Device memory methods delegate to torch.accelerator.get_memory_info.

    The per-process HBM limit is read from the runtime (``(free, limit)``), not
    a static per-device-name table, so these mock get_memory_info + get_device
    to run without a real TPU.
    """

    def setUp(self):
        self.platform = TpuSRTPlatform()
        self.platform._backend_initialized = True

    def _patch_mem(self, free, limit):
        # get_device() builds torch.device("tpu", ...) which is invalid off-TPU;
        # stub it, and stub the runtime memory query to (free, limit).
        return (
            patch.object(self.platform, "get_device", return_value=None),
            patch("torch.accelerator.get_memory_info", return_value=(free, limit)),
        )

    def test_uninitialized_returns_zero_without_calling_runtime(self):
        uninit_platform = TpuSRTPlatform()
        with patch("torch.accelerator.get_memory_info") as mock_mem:
            self.assertEqual(uninit_platform.get_device_total_memory(), 0)
            self.assertEqual(uninit_platform.get_current_memory_usage(), 0.0)
            mock_mem.assert_not_called()

    def test_total_memory_returns_reported_limit(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            self.assertEqual(self.platform.get_device_total_memory(), 96 << 30)

    def test_current_memory_usage_is_total_minus_free(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            self.assertEqual(self.platform.get_current_memory_usage(), float(86 << 30))

    def test_available_memory_returns_free_and_total(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            free, total = self.platform.get_available_memory()
            self.assertEqual(total, 96 << 30)
            self.assertEqual(free, 10 << 30)


class TestTpuPlatformServerDefaults(CustomTestCase):
    """apply_server_args_defaults sets correct values."""

    def _args(self, **overrides):
        # apply_server_args_defaults declares rather than writes, so the assertions read the
        # decisions back with resolution_result, as the projection does.
        defaults = dict(
            dtype=None,
            attention_backend=None,
            sampling_backend=None,
            disable_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            enable_torch_compile=False,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            enable_dynamic_chunking=False,
            page_size=None,
            kv_cache_dtype="auto",
            use_ray=False,
            hicache_io_backend=None,
            max_total_tokens=None,
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            speculative_use_rejection_sampling=False,
            dcp_size=1,
            attn_cp_size=1,
            enable_prefill_cp=False,
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
            cp_strategy=None,
            moe_dp_size=1,
            tpu_compile_tracer="dynamo",
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _apply(self, args):
        platform = TpuSRTPlatform()
        with patch("sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"):
            platform.apply_server_args_defaults(args)

    def _resolved(self, args, field):
        """What resolution decided for ``field``, else what the caller supplied."""
        from sglang.srt.arg_groups.overrides import resolution_result

        return resolution_result(args, field)

    def test_defaults_applied(self):
        args = self._args(enable_torch_compile=False)
        self._apply(args)

        self.assertEqual(self._resolved(args, "dtype"), "bfloat16")
        self.assertEqual(self._resolved(args, "attention_backend"), "pallas_tpu_rpa_v3")
        self.assertEqual(self._resolved(args, "sampling_backend"), "tpu")
        self.assertTrue(self._resolved(args, "disable_cuda_graph"))
        graph_config = self._resolved(args, "cuda_graph_config")
        self.assertEqual(graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(graph_config.prefill.backend, Backend.DISABLED)
        # enable_torch_compile is opt-in on TPU and must not be forced off.
        self.assertFalse(self._resolved(args, "enable_torch_compile"))
        # Overlap is enabled on TPU via the thread-based path in
        # scheduler_overlap_mixin; the platform must NOT force-disable it.
        self.assertFalse(self._resolved(args, "disable_overlap_schedule"))
        self.assertEqual(self._resolved(args, "mem_fraction_static"), 0.40)

    def test_resolved_cuda_graphs_disabled_without_disabling_tpu_compile(self):
        graph_config = default_cuda_graph_config()
        graph_config.decode.backend = Backend.FULL
        graph_config.prefill.backend = Backend.BREAKABLE
        graph_config.decode.bs = [1, 4]
        args = self._args(cuda_graph_config=graph_config, enable_torch_compile=True)
        self._apply(args)

        resolved = self._resolved(args, "cuda_graph_config")
        self.assertEqual(resolved.decode.backend, Backend.DISABLED)
        self.assertEqual(resolved.prefill.backend, Backend.DISABLED)
        self.assertEqual(resolved.decode.bs, [1, 4])
        self.assertTrue(self._resolved(args, "enable_torch_compile"))
        self.assertEqual(graph_config.decode.backend, Backend.FULL)
        self.assertEqual(graph_config.prefill.backend, Backend.BREAKABLE)

    def test_rpa_backends_default_to_fp8_kv_cache(self):
        for backend in ("pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched"):
            with self.subTest(backend=backend):
                args = self._args(attention_backend=backend)
                self._apply(args)
                self.assertEqual(self._resolved(args, "kv_cache_dtype"), "fp8_e4m3")

    def test_explicit_kv_cache_dtype_not_overridden(self):
        args = self._args(
            attention_backend="pallas_tpu_rpa_batched", kv_cache_dtype="bfloat16"
        )
        self._apply(args)
        self.assertEqual(self._resolved(args, "kv_cache_dtype"), "bfloat16")

    def test_non_rpa_backend_keeps_auto_kv_cache_dtype(self):
        # pallas_tpu_mla rejects an fp8 KV cache, so the fp8 default is scoped
        # to the RPA backends and must not leak to it.
        args = self._args(attention_backend="pallas_tpu_mla")
        self._apply(args)
        self.assertEqual(self._resolved(args, "kv_cache_dtype"), "auto")
        self.assertEqual(self._resolved(args, "page_size"), 64)

    def test_batched_rpa_page_size_default_meets_the_kernel_floor(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched")
        self._apply(args)
        self.assertEqual(self._resolved(args, "page_size"), 256)

    def test_batched_rpa_rejects_a_page_size_below_the_floor(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched", page_size=64)
        with self.assertRaisesRegex(ValueError, "page-size >= 256"):
            self._apply(args)

    def test_explicit_page_size_above_the_floor_is_kept(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched", page_size=512)
        self._apply(args)
        self.assertEqual(self._resolved(args, "page_size"), 512)

    def test_enable_torch_compile_preserved_when_opted_in(self):
        args = self._args(enable_torch_compile=True)
        self._apply(args)
        self.assertTrue(self._resolved(args, "enable_torch_compile"))

    def test_registered_spec_algorithm_name_is_admitted(self):
        # Plugin registration may happen after platform defaults. Unknown names
        # pass this backend gate and are validated by the shared registry later.
        args = self._args(speculative_algorithm="MY_PLUGIN")
        self._apply(args)

    def test_unsupported_builtin_spec_algorithm_is_rejected(self):
        args = self._args(speculative_algorithm="NGRAM")
        with self.assertRaisesRegex(NotImplementedError, "only EAGLE3"):
            self._apply(args)

    def test_dp_attention_requires_dp_lm_head_when_attn_tp_gt_1(self):
        args = self._args(
            enable_dp_attention=True,
            tp_size=8,
            dp_size=2,
            enable_dp_lm_head=False,
        )
        with self.assertRaisesRegex(
            ValueError, "When attention TP > 1 under DP attention on TPU"
        ):
            self._apply(args)

    def test_dp_attention_allows_valid_dp_lm_head_combos(self):
        # attn_tp_size > 1 with enable_dp_lm_head=True
        args = self._args(
            enable_dp_attention=True,
            tp_size=8,
            dp_size=2,
            enable_dp_lm_head=True,
        )
        self._apply(args)

        # attn_tp_size == 1 with enable_dp_lm_head=False
        args = self._args(
            enable_dp_attention=True,
            tp_size=8,
            dp_size=8,
            enable_dp_lm_head=False,
        )
        self._apply(args)

    def test_configures_export_mesh_from_tp_size(self):
        args = self._args(tp_size=16)
        platform = TpuSRTPlatform()

        with patch(
            "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"
        ) as prepare_tpu_env:
            platform.apply_server_args_defaults(args)

        prepare_tpu_env.assert_called_once_with(16, setup_slicebuilder=True)

    def test_parent_defaults_do_not_import_torch_tpu(self):
        args = self._args(tp_size=8)
        platform = TpuSRTPlatform()
        imported_modules = []
        real_import = builtins.__import__

        def track_import(name, *args, **kwargs):
            imported_modules.append(name)
            return real_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "sglang.srt.hardware_backend.tpu.env_override.apply_tpu_env_overrides"
            ),
            patch("builtins.__import__", side_effect=track_import),
        ):
            platform.apply_server_args_defaults(args)

        self.assertFalse(
            any(
                name == "torch_tpu" or name.startswith("torch_tpu.")
                for name in imported_modules
            )
        )

    def test_explicit_dtype_not_overridden(self):
        args = self._args(dtype="float32")
        self._apply(args)
        self.assertEqual(self._resolved(args, "dtype"), "float32")

    def test_explicit_mem_fraction_not_overridden(self):
        args = self._args(mem_fraction_static=0.60)
        self._apply(args)
        self.assertEqual(self._resolved(args, "mem_fraction_static"), 0.60)

    def test_explicit_sampling_backend_not_overridden(self):
        args = self._args(sampling_backend="pytorch")
        self._apply(args)
        self.assertEqual(self._resolved(args, "sampling_backend"), "pytorch")


class TestTpuPlatformDecodeContextParallel(CustomTestCase):
    """Decode context parallel is admitted only for the configuration the TPU
    batched RPA backend implements; every other combination is refused by name."""

    def _args(self, **overrides):
        defaults = dict(
            attention_backend="pallas_tpu_rpa_batched",
            tp_size=8,
            dcp_size=2,
            pp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            attn_cp_size=1,
            enable_prefill_cp=False,
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
            dcp_comm_backend="ag_rs",
            dcp_replicate_q_proj=False,
            enable_torch_compile=True,
            tpu_compile_tracer="dynamo",
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            speculative_use_rejection_sampling=False,
            dtype=None,
            sampling_backend=None,
            disable_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            page_size=None,
            chunked_prefill_size=None,
            kv_cache_dtype="auto",
            use_ray=False,
            hicache_io_backend=None,
            max_total_tokens=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _validate(self, args, env=None):
        environment = {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}
        environment.update(env or {})
        with patch.dict(os.environ, environment):
            _validate_decode_context_parallel(args)

    def _assert_refused(self, pattern, **overrides):
        with self.assertRaisesRegex(ValueError, pattern):
            self._validate(self._args(**overrides))

    def test_accepts_the_benchmark_configuration(self):
        # pallas_tpu_rpa_batched, tp 8, dcp 2, no DP attention, torch compile on.
        self._validate(self._args())

    def test_accepts_the_host_req_to_token_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TPU_REQ_TO_TOKEN_ON_HOST", None)
            _validate_decode_context_parallel(self._args())

    def test_dcp_disabled_admits_every_other_flag(self):
        self._validate(
            self._args(
                dcp_size=1,
                attention_backend="pallas_tpu_rpa_v3",
                pp_size=2,
                enable_dp_attention=True,
                enable_torch_compile=False,
            )
        )

    def test_refuses_another_attention_backend(self):
        for backend in ("pallas_tpu_rpa_v3", "pallas_tpu_flash", "torch_native"):
            with self.subTest(backend=backend):
                self._assert_refused(
                    "(?i)attention[-_ ]backend", attention_backend=backend
                )

    def test_refuses_a_tp_size_the_dcp_group_does_not_divide(self):
        self._assert_refused(
            "(?i)tp[-_ ]size|tensor[-_ ]parallel", tp_size=6, dcp_size=4
        )

    def test_refuses_pipeline_parallelism(self):
        self._assert_refused("(?i)pp[-_ ]size|pipeline", pp_size=2)

    def test_refuses_speculative_decoding(self):
        self._assert_refused("(?i)specul", speculative_algorithm="EAGLE3")

    def test_refuses_dp_attention(self):
        self._assert_refused(
            "(?i)dp[-_ ]attention", enable_dp_attention=True, dp_size=2
        )

    def test_refuses_prefill_context_parallelism(self):
        self._assert_refused(
            "(?i)attn[-_ ]cp[-_ ]size|context[-_ ]parallel", attn_cp_size=2
        )
        self._assert_refused("(?i)prefill[-_ ]cp", enable_prefill_cp=True)

    def test_refuses_pd_disaggregation(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                self._assert_refused("(?i)disaggregation", disaggregation_mode=mode)

    def test_refuses_hierarchical_cache(self):
        self._assert_refused("(?i)hierarchical", enable_hierarchical_cache=True)

    def test_refuses_another_dcp_communication_backend(self):
        for backend in ("a2a", "fi_a2a"):
            with self.subTest(backend=backend):
                self._assert_refused("(?i)dcp[-_ ]comm", dcp_comm_backend=backend)

    def test_refuses_a_replicated_q_projection(self):
        self._assert_refused("(?i)replicate[-_ ]q", dcp_replicate_q_proj=True)

    def test_refuses_eager_mode(self):
        self._assert_refused("(?i)torch[-_ ]compile", enable_torch_compile=False)

    def test_refuses_the_direct_tracer(self):
        self._assert_refused("(?i)tracer", tpu_compile_tracer="direct")

    def test_refuses_a_device_resident_req_to_token(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST"):
            self._validate(self._args(), env={"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "0"})

    def test_refuses_the_seq_along_lane_kv_layout(self):
        # The owner-masked KV write addresses the token-major fused cache.
        with self.assertRaisesRegex(ValueError, "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV"):
            self._validate(self._args(), env={"SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1"})

    def test_refuses_a_chunk_below_one_dcp_page(self):
        # A DCP page holds page_size * dcp_size tokens and prefill chunks end on
        # page boundaries, so a smaller chunk would leave every request waiting.
        self._assert_refused(
            "(?i)chunked[-_ ]prefill",
            dcp_size=8,
            page_size=256,
            chunked_prefill_size=2047,
        )
        for chunk in (2048, -1, None):
            with self.subTest(chunked_prefill_size=chunk):
                self._validate(
                    self._args(dcp_size=8, page_size=256, chunked_prefill_size=chunk)
                )

    def test_chunk_check_uses_the_resolved_page_size(self):
        # An unset page resolves to the batched kernel's 256-token floor first.
        args = self._args(dcp_size=8, page_size=None, chunked_prefill_size=1024)
        with (
            patch("sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"),
            patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}),
        ):
            with self.assertRaisesRegex(ValueError, r"256 x 8 = 2048"):
                TpuSRTPlatform().apply_server_args_defaults(args)

    def test_unset_backend_resolves_to_batched_rpa(self):
        # The platform default (RPA v3) is refused under DCP, so an unset
        # backend has to resolve to batched RPA and take its page floor.
        from sglang.srt.arg_groups.overrides import resolution_result

        args = self._args(attention_backend=None)
        with (
            patch("sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"),
            patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}),
        ):
            TpuSRTPlatform().apply_server_args_defaults(args)
        self.assertEqual(
            resolution_result(args, "attention_backend"), "pallas_tpu_rpa_batched"
        )
        self.assertEqual(resolution_result(args, "page_size"), 256)

    def test_platform_defaults_run_the_validator(self):
        # The gate has to be wired into apply_server_args_defaults, or a bad
        # configuration reaches the model runner.
        platform = TpuSRTPlatform()
        args = self._args(enable_dp_attention=True, dp_size=2, enable_dp_lm_head=True)
        with (
            patch("sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"),
            patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "1"}),
        ):
            with self.assertRaisesRegex(ValueError, "(?i)dp[-_ ]attention"):
                platform.apply_server_args_defaults(args)


class TestTpuPlatformFactoryMethods(CustomTestCase):
    """Factory methods return the expected classes."""

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_graph_runner_cls(self):
        cls = self.platform.get_graph_runner_cls()
        self.assertEqual(cls.__name__, "TpuGraphRunner")

    def test_paged_allocator_cls(self):
        cls = self.platform.get_paged_allocator_cls()
        self.assertEqual(cls.__name__, "TpuPagedTokenToKVPoolAllocator")

    def test_mha_kv_pool_cls(self):
        # get_mha_kv_pool_cls reads the exec bag; publish the backend.
        with get_context().override_server_args(attention_backend="torch_native"):
            cls = self.platform.get_mha_kv_pool_cls()
        self.assertEqual(cls.__name__, "MHATokenToKVPool")

        with get_context().override_server_args(attention_backend="pallas_tpu_rpa_v3"):
            cls = self.platform.get_mha_kv_pool_cls()
        self.assertEqual(cls.__name__, "TpuFusedMHATokenToKVPool")

    def test_nsa_raises(self):
        with self.assertRaises(NotImplementedError):
            self.platform.get_nsa_kv_pool_cls()

    def test_piecewise_backend_raises(self):
        with self.assertRaises(NotImplementedError):
            self.platform.get_piecewise_backend_cls()


class TestPrefillContextParallelValidation(TestTpuPlatformServerDefaults):
    """apply_server_args_defaults gates the prefill context-parallel path."""

    def _cp_args(self, **overrides):
        fields = dict(
            tp_size=8,
            attn_cp_size=8,
            enable_prefill_cp=True,
            cp_strategy="zigzag",
            moe_dp_size=8,
            enable_torch_compile=True,
            sampling_backend="pytorch",
        )
        fields.update(overrides)
        return self._args(**fields)

    def test_accepts_the_zigzag_batched_rpa_configuration(self):
        with patch.dict(os.environ, {"SGLANG_TPU_REQ_TO_TOKEN_ON_HOST": "0"}):
            args = self._cp_args(sampling_backend=None)
            self._apply(args)
            self.assertEqual(
                self._resolved(args, "attention_backend"), "pallas_tpu_rpa_batched"
            )
            self.assertEqual(self._resolved(args, "page_size"), 256)
            self.assertEqual(self._resolved(args, "sampling_backend"), "pytorch")
            self.assertTrue(envs.SGLANG_TPU_REQ_TO_TOKEN_ON_HOST.get())

    def test_turns_off_the_sparse_core_gathers(self):
        """On the current libtpu the context-parallel forward returns wrong tokens
        or halts with SparseCore gathers, even when the user turned them on."""
        with patch.dict(
            os.environ,
            {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "1", "LIBTPU_INIT_ARGS": ""},
        ):
            self._apply(self._cp_args())
            self.assertFalse(envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get())
            libtpu_args = os.environ["LIBTPU_INIT_ARGS"].split()
            self.assertIn("--xla_tpu_offload_gather_to_sparsecore=false", libtpu_args)
            self.assertIn(
                "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
                libtpu_args,
            )

    def test_attn_cp_size_one_is_untouched(self):
        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": ""}):
            os.environ.pop("SGLANG_TPU_FUSED_MOE_SC_GATHER", None)
            args = self._args(tp_size=8, attn_cp_size=1, enable_prefill_cp=False)
            self._apply(args)
            self.assertEqual(self._resolved(args, "sampling_backend"), "tpu")
            self.assertTrue(envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get())
            self.assertEqual(os.environ["LIBTPU_INIT_ARGS"], "")

    def test_rejections_name_the_missing_piece(self):
        cases = [
            (dict(enable_prefill_cp=False), "requires --enable-prefill-cp"),
            (dict(cp_strategy="interleave"), "requires --cp-strategy zigzag"),
            (
                dict(attention_backend="pallas_tpu_rpa_v3"),
                "requires --attention-backend pallas_tpu_rpa_batched",
            ),
            (dict(speculative_algorithm="EAGLE3"), "speculative decoding"),
            (dict(enable_dp_attention=True, dp_size=8), "DP attention"),
            (dict(moe_dp_size=1), "requires --moe-dp-size 8"),
            (dict(enable_torch_compile=False), "requires --enable-torch-compile"),
            (dict(tpu_compile_tracer="direct"), "requires --tpu-compile-tracer dynamo"),
            (dict(sampling_backend="tpu"), "requires --sampling-backend pytorch"),
        ]
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                patch.dict(os.environ, {}, clear=False),
            ):
                with self.assertRaisesRegex(NotImplementedError, message):
                    self._apply(self._cp_args(**overrides))

    def test_rejects_the_seq_along_lane_kv_layout(self):
        with (
            patch.dict(os.environ, {"SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1"}),
            self.assertRaisesRegex(
                NotImplementedError, "unset SGLANG_TPU_USE_SEQ_ALONG_LANE_KV"
            ),
        ):
            self._apply(self._cp_args())


if __name__ == "__main__":
    unittest.main()
