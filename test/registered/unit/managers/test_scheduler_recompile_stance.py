import contextlib
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The scheduler delegates the stance to this TPU module, which logs it.
_STANCE_MOD = "sglang.srt.hardware_backend.tpu.compilation.recompilation_checker"


class TestSchedulerRecompileStance(unittest.TestCase):
    def setUp(self):
        # ExitStack rather than enterContext, which is 3.11+.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(envs.SGLANG_TPU_ENABLE_UNSAFE_GUARD_SKIP.override(False))

    def _make_scheduler(
        self, *, speculative_algorithm, enable_dp_attention=False, device="tpu"
    ):
        # _maybe_arm_fail_on_recompile reads the exec and spec bags; publish both.
        override = get_context().override_server_args(
            enable_torch_compile=True,
            speculative_algorithm=speculative_algorithm,
            enable_dp_attention=enable_dp_attention,
        )
        server_args = override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(device=device)
        )
        # Built with __new__: wire server_args to the published instance.
        scheduler.server_args = server_args
        return scheduler

    def test_spec_freezes_recompiles_by_default(self):
        scheduler = self._make_scheduler(speculative_algorithm="EAGLE3")

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(False),
            patch("torch.compiler.set_stance", autospec=True) as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")

    def test_env_allows_spec_recompiles(self):
        scheduler = self._make_scheduler(speculative_algorithm="EAGLE3")

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(True),
            patch("torch.compiler.set_stance", autospec=True) as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_not_called()

    def test_hybrid_dp_attention_freezes_spec_recompiles(self):
        scheduler = self._make_scheduler(
            speculative_algorithm="EAGLE3", enable_dp_attention=True
        )

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(False),
            patch("torch.compiler.set_stance", autospec=True) as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")

    def test_env_does_not_affect_non_spec(self):
        scheduler = self._make_scheduler(speculative_algorithm=None)

        with (
            envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(True),
            patch("torch.compiler.set_stance", autospec=True) as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")

    def test_unsafe_guard_skip_is_opt_in_and_logged(self):
        scheduler = self._make_scheduler(speculative_algorithm=None)

        with (
            envs.SGLANG_TPU_ENABLE_UNSAFE_GUARD_SKIP.override(True),
            patch("torch.compiler.set_stance", autospec=True) as set_stance,
            self.assertLogs(_STANCE_MOD, level="WARNING") as logs,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with(
            "fail_on_recompile", skip_guard_eval_unsafe=True
        )
        self.assertEqual(logs.records[-1].levelno, logging.WARNING)
        self.assertIn("skip_guard_eval_unsafe=True", logs.records[-1].getMessage())

    def test_unsafe_guard_skip_is_ignored_off_tpu(self):
        scheduler = self._make_scheduler(speculative_algorithm=None, device="cuda")

        with (
            envs.SGLANG_TPU_ENABLE_UNSAFE_GUARD_SKIP.override(True),
            patch("torch.compiler.set_stance", autospec=True) as set_stance,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_not_called()

    def test_unsafe_guard_skip_is_ignored_with_dp_attention(self):
        scheduler = self._make_scheduler(
            speculative_algorithm=None, enable_dp_attention=True
        )

        with (
            envs.SGLANG_TPU_ENABLE_UNSAFE_GUARD_SKIP.override(True),
            patch("torch.compiler.set_stance", autospec=True) as set_stance,
            self.assertLogs(_STANCE_MOD, level="INFO") as logs,
        ):
            scheduler._maybe_arm_fail_on_recompile()

        set_stance.assert_called_once_with("fail_on_recompile")
        warnings = [r for r in logs.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertIn("DP attention", warnings[0].getMessage())

    def test_unsafe_guard_skip_is_ignored_with_speculative_decoding(self):
        for allow_spec_recompile in (False, True):
            with self.subTest(allow_spec_recompile=allow_spec_recompile):
                scheduler = self._make_scheduler(speculative_algorithm="EAGLE3")
                with (
                    envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.override(allow_spec_recompile),
                    envs.SGLANG_TPU_ENABLE_UNSAFE_GUARD_SKIP.override(True),
                    patch("torch.compiler.set_stance", autospec=True) as set_stance,
                    self.assertLogs(_STANCE_MOD, level="INFO") as logs,
                ):
                    scheduler._maybe_arm_fail_on_recompile()

                if allow_spec_recompile:
                    set_stance.assert_not_called()
                else:
                    set_stance.assert_called_once_with("fail_on_recompile")
                warnings = [r for r in logs.records if r.levelno == logging.WARNING]
                self.assertEqual(len(warnings), 1)
                self.assertIn(
                    "Ignoring SGLANG_TPU_ENABLE_UNSAFE_GUARD_SKIP",
                    warnings[0].getMessage(),
                )


if __name__ == "__main__":
    unittest.main()
