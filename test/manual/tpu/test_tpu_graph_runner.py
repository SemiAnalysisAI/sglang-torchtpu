"""Unit tests for TpuGraphRunner.

Mocks torch.compile and the tpu dynamo backend registration so the
tests run on any machine without TPU hardware.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    TpuCompiledForwardShim,
)
from sglang.srt.model_executor.tpu_graph_runner import TpuGraphRunner
from sglang.test.test_utils import CustomTestCase

PRECOMPILE_ALL = (
    "sglang.srt.hardware_backend.tpu.compilation."
    "compilation_manager.CompilationManager.precompile_all"
)


def _model_with(**attrs):
    """Return a bare object with the given attributes."""

    class _M:
        pass

    m = _M()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class TestTpuGraphRunner(CustomTestCase):
    """TpuGraphRunner wraps model methods with torch.compile correctly."""

    def setUp(self):
        def _fake_compile(fn, **kwargs):
            m = MagicMock(name="compiled")
            m._wrapped_fn = fn
            m._wrapped_kwargs = kwargs
            return m

        compile_patcher = patch(
            "sglang.srt.model_executor.tpu_graph_runner.torch.compile",
            side_effect=_fake_compile,
        )
        self.mock_compile = compile_patcher.start()
        self.addCleanup(compile_patcher.stop)
        register_patcher = patch(
            "sglang.srt.hardware_backend.tpu.patches._register_tpu_compile_backend"
        )
        self.mock_register = register_patcher.start()
        self.addCleanup(register_patcher.stop)
        precompile_patcher = patch(PRECOMPILE_ALL)
        self.mock_precompile = precompile_patcher.start()
        self.addCleanup(precompile_patcher.stop)

    def _runner_for(self, model, enable_torch_compile=True):
        return SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(context_len=2048),
            max_running_requests=1,
            is_draft_worker=False,
            server_args=SimpleNamespace(
                enable_torch_compile=enable_torch_compile,
                tpu_compile_tokens=None,
                tpu_compile_bs=None,
                chunked_prefill_size=None,
                max_prefill_tokens=2048,
                speculative_num_steps=None,
                speculative_algorithm=None,
                speculative_eagle_topk=None,
            ),
        )

    def test_wraps_forward_when_enabled(self):
        def original(input_ids, positions, forward_batch, input_embeds=None):
            pass

        model = _model_with(forward=original)
        runner = self._runner_for(model)
        TpuGraphRunner(runner)
        self.assertIsNot(model.forward, original)
        self.assertIn("input_embeds", inspect.signature(model.forward).parameters)
        self.assertIsInstance(runner.compiled_forward_shim, TpuCompiledForwardShim)
        self.mock_compile.assert_called_once()
        _, kwargs = self.mock_compile.call_args
        self.assertEqual(kwargs["backend"], "tpu")
        self.assertTrue(kwargs["fullgraph"])
        self.assertFalse(kwargs["dynamic"])
        self.mock_register.assert_called_once()

    def test_unwraps_no_grad_decorator_for_dynamo_isolation(self):
        """torch.compile receives the unwrapped forward, not the
        @torch.no_grad() decorator wrapper, so that models with
        inherited forward() get distinct Dynamo graph caches."""
        import torch

        @torch.no_grad()
        def parent_forward(self, input_ids, positions, forward_batch):
            pass

        @torch.no_grad()
        def child_forward(self, input_ids, positions, forward_batch):
            return parent_forward(self, input_ids, positions, forward_batch)

        # Both share the same decorator __code__.
        self.assertIs(
            parent_forward.__code__,
            child_forward.__code__,
        )
        # But __wrapped__ has distinct __code__.
        self.assertIsNot(
            parent_forward.__wrapped__.__code__,
            child_forward.__wrapped__.__code__,
        )

        target_model = _model_with(forward=parent_forward)
        draft_model = _model_with(forward=child_forward)

        from sglang.srt.model_executor.tpu_graph_runner import (
            _wrap_with_tpu_compile,
        )

        _wrap_with_tpu_compile(target_model, "forward")
        _wrap_with_tpu_compile(draft_model, "forward")

        self.assertEqual(self.mock_compile.call_count, 2)

        # The raw functions passed to torch.compile should have
        # different __code__ (the actual forward, not the decorator).
        target_compiled_fn = self.mock_compile.call_args_list[0][0][0]
        draft_compiled_fn = self.mock_compile.call_args_list[1][0][0]
        self.assertIsNot(target_compiled_fn.__code__, draft_compiled_fn.__code__)

    def test_can_run_is_always_false(self):
        runner = TpuGraphRunner(self._runner_for(_model_with(forward=MagicMock())))
        self.assertFalse(runner.can_run(MagicMock()))


if __name__ == "__main__":
    unittest.main()
