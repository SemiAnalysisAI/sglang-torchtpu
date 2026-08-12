# SPDX-License-Identifier: Apache-2.0
"""A1: platform lifecycle hooks fire from the shared stage machinery,
so stages that override forward() inherit them."""

import weakref
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as base_mod
from sglang.multimodal_gen.runtime.utils import perf_logger


class _Recorder:
    def __init__(self):
        self.events = []

    @contextmanager
    def _ctx(self, name):
        self.events.append(f"enter:{name}")
        yield
        self.events.append(f"exit:{name}")

    def denoise_loop_context(self):
        return self._ctx("denoise")

    def vae_decode_context(self, pipeline):
        self.events.append(f"pipeline:{pipeline}")
        return self._ctx("decode")


def _stage_with_role(role):
    cls = type(
        "S",
        (base_mod.PipelineStage,),
        {
            "role_affinity": property(lambda s: role),
            "forward": lambda s, batch, server_args: batch,
        },
    )
    return object.__new__(cls)  # skip __init__: context needs no state


class TestPlatformStageContext:
    def test_denoiser_role_gets_denoise_context(self):
        rec = _Recorder()
        stage = _stage_with_role(RoleType.DENOISER)
        with patch.object(base_mod, "current_platform", rec):
            with stage._platform_stage_context():
                pass
        assert rec.events == ["enter:denoise", "exit:denoise"]

    def test_decoder_role_gets_decode_context_with_weakref_pipeline(self):
        rec = _Recorder()
        stage = _stage_with_role(RoleType.DECODER)
        sentinel = MagicMock()
        stage.pipeline = weakref.ref(sentinel)
        with patch.object(base_mod, "current_platform", rec):
            with stage._platform_stage_context():
                pass
        assert rec.events[0] == f"pipeline:{sentinel}"
        assert rec.events[1:] == ["enter:decode", "exit:decode"]

    def test_encoder_role_gets_no_context(self):
        rec = _Recorder()
        stage = _stage_with_role(RoleType.ENCODER)
        with patch.object(base_mod, "current_platform", rec):
            with stage._platform_stage_context():
                pass
        assert rec.events == []


class TestStepProfilerHook:
    def _profiler(self, name="x", **kw):
        return perf_logger.StageProfiler(name, logger=MagicMock(), metrics=None, **kw)

    def test_step_profiler_fires_platform_hook(self):
        platform = MagicMock()
        with patch.object(perf_logger, "current_platform", platform):
            with self._profiler(record_as_step=True):
                pass
        platform.on_denoise_step_end.assert_called_once_with()

    def test_denoising_step_name_fires_without_flag(self):
        platform = MagicMock()
        with patch.object(perf_logger, "current_platform", platform):
            with self._profiler("denoising_step_3"):
                pass
        platform.on_denoise_step_end.assert_called_once_with()

    def test_non_step_profiler_does_not_fire(self):
        platform = MagicMock()
        with patch.object(perf_logger, "current_platform", platform):
            with self._profiler():
                pass
        platform.on_denoise_step_end.assert_not_called()

    def test_exception_skips_hook(self):
        platform = MagicMock()
        with patch.object(perf_logger, "current_platform", platform):
            try:
                with self._profiler(record_as_step=True):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
        platform.on_denoise_step_end.assert_not_called()
