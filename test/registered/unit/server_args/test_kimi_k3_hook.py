"""Kimi hybrid spec-decode server-arg hook."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.kimi_k3_hook import apply_kimi_k3_spec_backend_defaults
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**overrides):
    fields = dict(
        speculative_algorithm="EAGLE",
        linear_attn_verify_backend=None,
        speculative_draft_attention_backend=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestKimiK3SpecBackendDefaults(CustomTestCase):
    def test_non_tpu_pins_cutedsl_verify_backend(self):
        args = _args()
        with override_platform(is_tpu=False, is_sm100=False):
            apply_kimi_k3_spec_backend_defaults(args)
        self.assertEqual(
            resolution_result(args, "linear_attn_verify_backend"), "nv_cutedsl"
        )

    def test_tpu_rejects_speculative_decoding(self):
        """Spec decode on TPU fails at launch."""
        with override_platform(is_tpu=True, is_sm100=False):
            with self.assertRaisesRegex(ValueError, "not supported"):
                apply_kimi_k3_spec_backend_defaults(_args())

    def test_tpu_without_speculative_decoding_is_allowed(self):
        with override_platform(is_tpu=True, is_sm100=False):
            apply_kimi_k3_spec_backend_defaults(_args(speculative_algorithm=None))

    def test_kimi_k3_linear_arch_registered_in_mamba_cache_sets(self):
        from sglang.srt.arg_groups.overrides import (
            _MAMBA_EXTRA_BUFFER_ARCHS,
            _MAMBA_RADIX_CACHE_ARCHS,
        )

        for arch in ("KimiK3LinearForCausalLM", "KimiK3ForConditionalGeneration"):
            self.assertIn(arch, _MAMBA_RADIX_CACHE_ARCHS)
            self.assertIn(arch, _MAMBA_EXTRA_BUFFER_ARCHS)


if __name__ == "__main__":
    unittest.main()
