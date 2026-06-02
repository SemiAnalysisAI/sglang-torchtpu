"""Unit tests for TpuGraphRunner.

Mocks torch.compile and the tpu dynamo backend registration so the
tests run on any machine without TPU hardware.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.model_executor.tpu_graph_runner import TpuGraphRunner
from sglang.test.test_utils import CustomTestCase


def _model_with(**attrs):
    """Return a bare object with the given attributes.

    A bare object is preferable here over MagicMock because MagicMock
    auto-creates attributes — so getattr(m, "forward_split_prefill",
    None) returns a Mock instead of None, and we can't distinguish
    "method present" from "method absent."
    """

    class _M:
        pass

    m = _M()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class TestTpuGraphRunner(CustomTestCase):
    """TpuGraphRunner wraps model methods with torch.compile correctly."""

    def setUp(self):
        # Stub torch.compile so the test doesn't need the tpu dynamo backend.
        compile_patcher = patch(
            "sglang.srt.model_executor.tpu_graph_runner.torch.compile",
            side_effect=lambda fn, **kwargs: (fn, kwargs),
        )
        self.mock_compile = compile_patcher.start()
        self.addCleanup(compile_patcher.stop)
        # Stub the backend-registration import so the test doesn't need torch_tpu.
        register_patcher = patch(
            "sglang.srt.hardware_backend.tpu.patches._register_tpu_compile_backend"
        )
        self.mock_register = register_patcher.start()
        self.addCleanup(register_patcher.stop)

    def _runner_for(self, model, enable_torch_compile=True):
        return SimpleNamespace(
            model=model,
            server_args=SimpleNamespace(
                enable_torch_compile=enable_torch_compile,
            ),
        )

    def test_wraps_forward_when_enabled(self):
        original = MagicMock()
        model = _model_with(forward=original)
        TpuGraphRunner(self._runner_for(model))
        self.assertIsNot(model.forward, original)
        self.mock_compile.assert_called_once()
        _, kwargs = self.mock_compile.call_args
        # Pinned compile kwargs — these are correctness-critical for the
        # TPU backend (dynamic=True triggers recompiles per shape;
        # fullgraph=False allows torch.compile to fall back to eager on
        # unsupported ops, which silently breaks the XLA pipeline).
        self.assertEqual(kwargs["backend"], "tpu")
        self.assertTrue(kwargs["fullgraph"])
        self.assertFalse(kwargs["dynamic"])
        self.mock_register.assert_called_once()

    def test_wraps_both_entry_points_when_present(self):
        model = _model_with(
            forward=MagicMock(),
            forward_split_prefill=MagicMock(),
        )
        TpuGraphRunner(self._runner_for(model))
        self.assertEqual(self.mock_compile.call_count, 2)

    def test_skips_missing_split_prefill_silently(self):
        # Only `forward` exists — no warning, no error.
        model = _model_with(forward=MagicMock())
        TpuGraphRunner(self._runner_for(model))
        self.assertEqual(self.mock_compile.call_count, 1)

    def test_respects_can_torch_compile_false(self):
        original = MagicMock()
        model = _model_with(forward=original)
        model._can_torch_compile = False
        runner = self._runner_for(model)
        TpuGraphRunner(runner)
        # forward was not replaced; torch.compile and registration both skipped.
        self.assertIs(model.forward, original)
        self.mock_compile.assert_not_called()
        self.mock_register.assert_not_called()
        # The skip is reflected back so downstream warmup doesn't try
        # to specialize over compile-only buckets.
        self.assertFalse(runner.server_args.enable_torch_compile)

    def test_can_torch_compile_true_default(self):
        # Models that don't set _can_torch_compile should still be wrapped.
        model = _model_with(forward=MagicMock())
        TpuGraphRunner(self._runner_for(model))
        self.mock_compile.assert_called_once()

    def test_warns_when_no_targets_present(self):
        # Model exposes neither forward nor forward_split_prefill:
        # nothing to wrap. We log a warning and return without raising.
        model = _model_with()  # empty
        TpuGraphRunner(self._runner_for(model))
        self.mock_compile.assert_not_called()
        # Backend registration still happens — the gate is _can_torch_compile.
        self.mock_register.assert_called_once()

    def test_can_run_is_always_false(self):
        # ModelRunner.forward branches on `graph_runner.can_run(...)`
        # to decide whether to replay a captured graph. TPU has no
        # captured graph (PJRT caches HLO internally), so this must
        # be False for every forward mode so the replay path stays off.
        runner = TpuGraphRunner(self._runner_for(_model_with(forward=MagicMock())))
        self.assertFalse(runner.can_run(MagicMock()))


if __name__ == "__main__":
    unittest.main()
