# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression test for :func:`_reset_cp_v2_state`.

ForwardBatch instances are pooled/reused across forwards. Leaving stale
CP-v2 mirrors from a prior PCP forward makes the next non-CP forward
wrongly enter the ``moe_cp_allgather`` branch or the PCP attention
dispatch — silent wrong outputs.

The helper resets 7 fields. This test asserts every one of them, so
adding a new mirror without also clearing it here becomes a loud CI
failure instead of a silent regression.

Pure Python; no TPU / JAX device required.
"""

from __future__ import annotations

import types
import unittest

from sglang.srt.model_executor.runner.eager_runner import _reset_cp_v2_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ResetCpV2StateTest(CustomTestCase):
    def _make_dirty_forward_batch(self) -> types.SimpleNamespace:
        """A ForwardBatch stand-in with every CP-v2 mirror set to a non-default."""
        return types.SimpleNamespace(
            attn_cp_metadata=object(),
            pcp_layout_plan=object(),
            pcp_attn_active=True,
            moe_cp_active=True,
            moe_cp_pad_max_tokens=42,
            moe_cp_local_tokens=37,
            pcp_cache_pages=8,
        )

    def test_clears_all_seven_cp_v2_fields(self) -> None:
        fb = self._make_dirty_forward_batch()
        _reset_cp_v2_state(fb)
        self.assertIsNone(fb.attn_cp_metadata)
        self.assertIsNone(fb.pcp_layout_plan)
        self.assertFalse(fb.pcp_attn_active)
        self.assertFalse(fb.moe_cp_active)
        self.assertEqual(fb.moe_cp_pad_max_tokens, 0)
        self.assertEqual(fb.moe_cp_local_tokens, 0)
        self.assertEqual(fb.pcp_cache_pages, 0)

    def test_idempotent(self) -> None:
        # Calling twice should leave the same defaults.
        fb = self._make_dirty_forward_batch()
        _reset_cp_v2_state(fb)
        _reset_cp_v2_state(fb)
        self.assertIsNone(fb.attn_cp_metadata)
        self.assertEqual(fb.pcp_cache_pages, 0)

    def test_does_not_touch_unrelated_fields(self) -> None:
        # Any field _reset_cp_v2_state doesn't know about must be preserved.
        fb = self._make_dirty_forward_batch()
        fb.some_unrelated_field = "keep me"
        fb.forward_mode = 42
        _reset_cp_v2_state(fb)
        self.assertEqual(fb.some_unrelated_field, "keep me")
        self.assertEqual(fb.forward_mode, 42)


if __name__ == "__main__":
    unittest.main()
