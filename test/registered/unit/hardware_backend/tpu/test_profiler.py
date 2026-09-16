"""Unit tests for the TPU xprof profiler wiring."""

import dataclasses
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.profiler import (
    _tpu_profile_options,
    start_tpu_profile,
    stop_tpu_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Pure-logic tests. torch_tpu is faked via sys.modules, so this runs on the
# CPU CI runner (no TPU required).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclasses.dataclass
class _BaseProfileOptions:
    host_tracer_level: int = 2


def _fake_torch_tpu(profile_options_cls=_BaseProfileOptions, calls=None):
    """Fake torch_tpu's private profiler layout.

    start_trace/stop_trace live in _impl; the package root exports only the
    torch.profiler-facing names. A fake that puts them at the root would let a
    regression to the wrong import path pass here.
    """
    impl = types.ModuleType("torch_tpu._internal.profiler._impl")
    if profile_options_cls is not None:
        impl.ProfileOptions = profile_options_cls
    if calls is not None:
        impl.start_trace = lambda d, o=None: calls.append(("start", d, o))
        impl.stop_trace = lambda: calls.append(("stop",))
    profiler = types.ModuleType("torch_tpu._internal.profiler")
    profiler._impl = impl
    internal = types.ModuleType("torch_tpu._internal")
    internal.profiler = profiler
    torch_tpu = types.ModuleType("torch_tpu")
    torch_tpu._internal = internal
    return {
        "torch_tpu": torch_tpu,
        "torch_tpu._internal": internal,
        "torch_tpu._internal.profiler": profiler,
        "torch_tpu._internal.profiler._impl": impl,
    }


class TestTpuProfileOptions(CustomTestCase):
    def _options(
        self, *, num_chips, num_sparse_cores, profile_options_cls=_BaseProfileOptions
    ):
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(num_chips),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(num_sparse_cores),
            patch.dict(sys.modules, _fake_torch_tpu(profile_options_cls)),
        ):
            return _tpu_profile_options()

    def test_limits_enabled(self):
        options = self._options(num_chips=1, num_sparse_cores=2)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_cores_to_trace, 2)
        # Tile cap only kicks in at a single SparseCore; here 2 leaves it off.
        self.assertIsNone(options.tpu_num_sparse_core_tiles_to_trace)
        # Base fields survive.
        self.assertEqual(options.host_tracer_level, 2)

    def test_single_sparse_core_narrows_to_one_tile(self):
        options = self._options(num_chips=1, num_sparse_cores=1)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_cores_to_trace, 1)
        self.assertEqual(options.tpu_num_sparse_core_tiles_to_trace, 1)

    def test_one_limit_disabled(self):
        options = self._options(num_chips=4, num_sparse_cores=0)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 4)
        self.assertIsNone(options.tpu_num_sparse_cores_to_trace)
        self.assertIsNone(options.tpu_num_sparse_core_tiles_to_trace)

    def test_both_limits_disabled_traces_everything(self):
        # None means "pass no options", i.e. every chip and SparseCore.
        self.assertIsNone(self._options(num_chips=0, num_sparse_cores=-1))

    def test_non_dataclass_profile_options_degrades(self):
        class _NotADataclass:
            pass

        # A torch_tpu that reshapes ProfileOptions must lose the narrowing, not
        # break capture altogether.
        self.assertIsNone(
            self._options(
                num_chips=1, num_sparse_cores=1, profile_options_cls=_NotADataclass
            )
        )

    def test_missing_profile_options_degrades(self):
        self.assertIsNone(
            self._options(num_chips=1, num_sparse_cores=1, profile_options_cls=None)
        )


class TestStartStopUsesLibtpuTrace(CustomTestCase):
    """The capture must not go through torch.profiler.

    torch.profiler reaches TPU device activity only via kineto, which needs
    libtpu registered through PyTorch's PrivateUse1ProfilerRegistry -- absent
    before torch 2.12, while the TPU stack pins 2.11. On that route the capture
    succeeds and contains host activity only, which is silent and easy to
    reintroduce. These assert the libtpu path is the one taken.
    """

    def test_start_calls_start_trace_with_output_dir(self):
        calls = []
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(0),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(0),
            patch.dict(sys.modules, _fake_torch_tpu(calls=calls)),
        ):
            trace = start_tpu_profile(output_dir=Path("/tmp/xprof-unit-test"))

        self.assertEqual([c[0] for c in calls], ["start"])
        self.assertEqual(calls[0][1], "/tmp/xprof-unit-test")
        self.assertEqual(trace.output_dir, Path("/tmp/xprof-unit-test"))

    def test_start_forwards_narrowing_options(self):
        calls = []
        with (
            envs.SGLANG_TPU_PROFILE_NUM_CHIPS.override(1),
            envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.override(1),
            patch.dict(sys.modules, _fake_torch_tpu(calls=calls)),
        ):
            start_tpu_profile(output_dir=Path("/tmp/xprof-unit-test"))

        options = calls[0][2]
        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)
        self.assertEqual(options.tpu_num_sparse_core_tiles_to_trace, 1)

    def test_stop_calls_stop_trace(self):
        calls = []
        with patch.dict(sys.modules, _fake_torch_tpu(calls=calls)):
            stop_tpu_profile(None)

        self.assertEqual([c[0] for c in calls], ["stop"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
