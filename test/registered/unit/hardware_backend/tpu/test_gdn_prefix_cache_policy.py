"""Only backends that write native TPU checkpoints may enable prefix reuse."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.arg_groups import mamba_hook, overrides
from sglang.srt.configs import Qwen3NextConfig, hybrid_arch
from sglang.srt.mem_cache import kv_cache_builder as builder
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.platform import TpuOOTPlatform
from sglang_tpu.tpu_platform import TpuSRTPlatform


class TestTpuPrefixCachePolicy(CustomTestCase):
    def test_lazy_strategy_is_rejected_only_on_tpu(self):
        for platform_cls in (TpuSRTPlatform, TpuOOTPlatform, SRTPlatform):
            tpu = platform_cls is not SRTPlatform
            for strategy in ("extra_buffer", "extra_buffer_lazy"):
                with self.subTest(platform=platform_cls.__name__, strategy=strategy):
                    platform = SimpleNamespace(
                        is_tpu=tpu,
                        is_cuda=not tpu,
                        is_musa=False,
                        is_npu=False,
                        is_hip=False,
                        is_xpu=False,
                    )
                    view = SimpleNamespace(
                        mamba_radix_cache_strategy=strategy,
                        enable_mixed_chunk=False,
                        disable_radix_cache=False,
                        enable_hierarchical_cache=False,
                        enable_int8_mamba_checkpoint=False,
                        linear_attn_backend="triton",
                        disaggregation_mode="null",
                        speculative_num_draft_tokens=None,
                        page_size=64,
                        mamba_track_interval=64,
                        chunked_prefill_size=256,
                    )
                    config = SimpleNamespace(architectures=["Qwen3NextForCausalLM"])
                    with (
                        patch.object(mamba_hook, "get_platform", return_value=platform),
                        patch.object(
                            mamba_hook,
                            "current_platform",
                            platform_cls(),
                        ),
                        patch.object(overrides, "get_platform", return_value=platform),
                        self.assertRaisesRegex(
                            ValueError,
                            "extra_buffer_lazy.*TPU.*"
                            "--mamba-radix-cache-strategy extra_buffer",
                        )
                        if tpu and strategy == "extra_buffer_lazy"
                        else contextlib.nullcontext(),
                    ):
                        mamba_hook.validate_mamba_extra_buffer(
                            view, config, mamba_cache_chunk_size_of=lambda: 64
                        )

    def test_mixed_chunk_prefix_cache_is_rejected_only_on_tpu(self):
        """Mixed merging drops the plan for a checkpoint the request still claims."""
        for platform_cls in (TpuSRTPlatform, TpuOOTPlatform, SRTPlatform):
            for mixed, disabled in ((True, False), (False, False), (True, True)):
                with self.subTest(
                    platform=platform_cls.__name__, mixed=mixed, disabled=disabled
                ):
                    view = SimpleNamespace(
                        enable_mixed_chunk=mixed,
                        disable_radix_cache=disabled,
                        mamba_radix_cache_strategy="extra_buffer",
                        enable_hierarchical_cache=False,
                        enable_int8_mamba_checkpoint=False,
                    )
                    rejects = platform_cls is not SRTPlatform and mixed and not disabled
                    with (
                        self.assertRaisesRegex(
                            ValueError,
                            "TPU GDN prefix caching.*--enable-mixed-chunk.*"
                            "--disable-radix-cache",
                        )
                        if rejects
                        else contextlib.nullcontext()
                    ):
                        platform_cls().validate_mamba_prefix_cache(view)

    def test_kimi_k3_config_reaches_kda_prefix_checkpoints(self):
        """Kimi-K3's composite config resolves to the KDA backend's checkpoints."""
        from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
        from sglang.srt.configs.kimi_k3 import KimiK3Config

        model = SimpleNamespace(hf_config=KimiK3Config(), is_draft_model=False)
        # Reached through KDA, not by being mistaken for a GDN model.
        self.assertIsNone(hybrid_gdn_config(model))
        for platform_cls in (TpuSRTPlatform, TpuOOTPlatform):
            with self.subTest(platform=platform_cls.__name__):
                self.assertTrue(platform_cls().supports_mamba_prefix_cache(model))

    def test_backend_guard_preserves_other_devices_and_explicit_disable(self):
        """Unsupported backends must not silently disable caching after pool sizing."""
        # (tpu, recurrent, gdn, kda, disabled, expected); None means "rejects".
        cases = (
            (True, True, False, False, False, None),  # Unsupported backend rejects
            (True, True, False, False, True, True),  # Explicit disable sizes pools
            (True, True, True, False, False, False),
            (True, True, True, False, True, True),
            (True, True, False, True, False, False),  # TPU KDA writes checkpoints
            (True, True, False, True, True, True),
            (True, False, False, False, False, False),  # Ordinary attention
            (False, True, False, False, False, False),  # GPU KDA remains supported
        )
        for platform_cls, recurrent, gdn, kda, disabled, expected in (
            (platform_cls, recurrent, gdn, kda, disabled, expected)
            for tpu, recurrent, gdn, kda, disabled, expected in cases
            for platform_cls in (
                (TpuSRTPlatform, TpuOOTPlatform) if tpu else (SRTPlatform,)
            )
        ):
            with self.subTest(
                platform=platform_cls.__name__,
                recurrent=recurrent,
                gdn=gdn,
                kda=kda,
                disabled=disabled,
            ):
                worker = MagicMock()
                worker.is_hybrid_swa = False
                worker.get_memory_pool.return_value = (MagicMock(), MagicMock())
                text_config = Qwen3NextConfig() if gdn else object()
                model = SimpleNamespace(
                    is_multimodal=False,
                    hf_config=SimpleNamespace(get_text_config=lambda: text_config),
                )
                memory = MagicMock(disable_radix_cache=disabled)
                parallel = MagicMock(dcp_enabled=False, enable_dp_attention=False)
                with contextlib.ExitStack() as stack:
                    for name, value in {
                        "get_memory": memory,
                        "get_parallel": parallel,
                        "get_dp_tp_group": MagicMock(),
                        "get_exec": MagicMock(),
                        "get_disagg": SimpleNamespace(
                            disaggregation_decode_enable_radix_cache=False
                        ),
                        "get_schedule": SimpleNamespace(chunked_prefill_size=256),
                        "get_resolved_model_impl": builder.ModelImpl.SGLANG,
                        "uses_ssm_state": recurrent,
                        "hybrid_gdn_config": object() if gdn else None,
                        "is_deepseek_dsa": False,
                        "resolve_decode_retraction_backup": "cpu_tensor",
                    }.items():
                        stack.enter_context(
                            patch.object(builder, name, return_value=value)
                        )
                    # The platform resolves KDA from the model config; a mock
                    # config has no real KimiLinearConfig to detect.
                    stack.enter_context(
                        patch.object(
                            hybrid_arch,
                            "hybrid_kda_config",
                            return_value=object() if kda else None,
                        )
                    )
                    stack.enter_context(
                        patch("sglang.srt.platforms.current_platform", platform_cls())
                    )
                    stack.enter_context(
                        patch.object(
                            builder,
                            "auto_size_hicache",
                            return_value=contextlib.nullcontext(),
                        )
                    )
                    create = stack.enter_context(
                        patch.object(builder, "create_tree_cache")
                    )
                    stack.enter_context(
                        patch.object(builder, "init_mm_embedding_cache")
                    )
                    if expected is None:
                        stack.enter_context(
                            self.assertRaisesRegex(
                                ValueError,
                                "prefix checkpoints.*--disable-radix-cache",
                            )
                        )
                    result = builder.build_kv_cache(
                        server_args=MagicMock(),
                        model_config=model,
                        tp_worker=worker,
                        page_size=256,
                        spec_algorithm=MagicMock(),
                        enable_metrics=False,
                        enable_kv_cache_events=False,
                        enable_hierarchical_cache=False,
                    )
                if expected is None:
                    continue
                self.assertEqual(result.disable_radix_cache, expected)
                self.assertEqual(create.call_args.args[0].params.disable, expected)


if __name__ == "__main__":
    unittest.main()
