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


class TestKimiHybridArchitectures(CustomTestCase):
    def test_every_kimi_k3_entry_class_gets_the_hybrid_hooks(self):
        """The text-only entry runs the same KDA decoder as the multimodal one
        (and is what the expert-pack runtime serves), so it needs the same
        spec-decode and linear-attention defaults and refusals."""
        from sglang.srt.arg_groups.kimi_k3_hook import KIMI_HYBRID_ARCHITECTURES
        from sglang.srt.models.kimi_k3 import EntryClass

        for cls in EntryClass:
            self.assertIn(cls.__name__, KIMI_HYBRID_ARCHITECTURES)


if __name__ == "__main__":
    unittest.main()
