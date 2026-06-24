"""Unit tests for the TPU SpecV2 (EAGLE) V1/V2 routing gate.

These exercise the pure routing decision in
``sglang.srt.arg_groups.speculative_hook`` -- the flag x env x device x topk x
model matrix -- with lightweight ``SimpleNamespace`` stubs, so they run on any
machine with NO TPU hardware and NO model load.

Routing contract under test:
  - On TPU, ``--enable-spec-v2-tpu`` is the SOLE enabler of SpecV2 and overrides
    the global ``SGLANG_ENABLE_SPEC_V2`` env in both directions; the env alone
    never enables TPU SpecV2. An explicit ``--disable-overlap-schedule`` wins.
  - On non-TPU devices the flag has no effect on V1/V2 selection.
  - When TPU SpecV2 is active, scoped-out configs are rejected (not silently
    enabled).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import (
    _apply_eagle_overlap_gate,
    _reject_unsupported_tpu_spec_v2,
    is_tpu_spec_v2_active,
    reject_request_for_tpu_spec_v2,
)
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


def _server_args(**overrides):
    """Minimal stub carrying only the attributes the gate/reject read."""
    base = dict(
        device="tpu",
        enable_spec_v2_tpu=False,
        disable_overlap_schedule=False,
        speculative_eagle_topk=1,
        speculative_algorithm="EAGLE3",
        speculative_adaptive=False,
        enable_dp_attention=False,
        enable_return_hidden_states=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestTpuEagleOverlapGate(CustomTestCase):
    """flag x env x device x topk routing on the overlap gate."""

    # ---- TPU: the flag is the sole enabler ----

    def test_tpu_no_flag_env_default_routes_v1(self):
        # No flag, env defaults True: TPU EAGLE3 must stay V1.
        sa = _server_args()
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            reason = _apply_eagle_overlap_gate(sa)
        self.assertTrue(sa.disable_overlap_schedule)
        self.assertIsNotNone(reason)

    def test_tpu_no_flag_env_true_still_v1(self):
        # SGLANG_ENABLE_SPEC_V2=1 WITHOUT the flag keeps V1 on TPU.
        sa = _server_args()
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            _apply_eagle_overlap_gate(sa)
        self.assertTrue(sa.disable_overlap_schedule)

    def test_tpu_flag_routes_v2(self):
        sa = _server_args(enable_spec_v2_tpu=True)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            reason = _apply_eagle_overlap_gate(sa)
        self.assertFalse(sa.disable_overlap_schedule)
        self.assertIsNone(reason)

    def test_tpu_flag_overrides_env_false(self):
        # The flag enables V2 regardless of the global env value.
        sa = _server_args(enable_spec_v2_tpu=True)
        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            _apply_eagle_overlap_gate(sa)
        self.assertFalse(sa.disable_overlap_schedule)

    def test_tpu_no_flag_env_false_v1(self):
        sa = _server_args()
        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            _apply_eagle_overlap_gate(sa)
        self.assertTrue(sa.disable_overlap_schedule)

    def test_tpu_explicit_disable_overlap_wins(self):
        # User passed --disable-overlap-schedule: even with the flag, stay V1.
        sa = _server_args(enable_spec_v2_tpu=True, disable_overlap_schedule=True)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            _apply_eagle_overlap_gate(sa)
        self.assertTrue(sa.disable_overlap_schedule)

    def test_tpu_flag_topk_gt_1_routes_v1(self):
        # topk>1 is scoped out: route to V1 (clear message), never silent V2.
        sa = _server_args(enable_spec_v2_tpu=True, speculative_eagle_topk=4)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            reason = _apply_eagle_overlap_gate(sa)
        self.assertTrue(sa.disable_overlap_schedule)
        self.assertIn("topk", reason)

    # ---- Non-TPU: the flag does not change selection ----

    def test_non_tpu_flag_ignored_env_true_v2(self):
        sa = _server_args(device="cuda", enable_spec_v2_tpu=True)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            _apply_eagle_overlap_gate(sa)
        # Same as the no-flag baseline: V2 by the global default.
        self.assertFalse(sa.disable_overlap_schedule)

    def test_non_tpu_flag_does_not_enable_when_env_false(self):
        # On non-TPU, env=False forces V1 and the TPU flag cannot override it.
        sa_flag = _server_args(device="cuda", enable_spec_v2_tpu=True)
        sa_noflag = _server_args(device="cuda", enable_spec_v2_tpu=False)
        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            _apply_eagle_overlap_gate(sa_flag)
            _apply_eagle_overlap_gate(sa_noflag)
        self.assertTrue(sa_flag.disable_overlap_schedule)
        # Identical selection with/without the flag on a non-TPU path.
        self.assertEqual(
            sa_flag.disable_overlap_schedule, sa_noflag.disable_overlap_schedule
        )

    def test_non_tpu_no_flag_env_true_v2(self):
        sa = _server_args(device="cuda")
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            _apply_eagle_overlap_gate(sa)
        self.assertFalse(sa.disable_overlap_schedule)


class TestTpuSpecV2Rejection(CustomTestCase):
    """Scoped-out configs are rejected when TPU SpecV2 is active."""

    def test_clean_eagle3_no_raise(self):
        sa = _server_args(enable_spec_v2_tpu=True)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            _apply_eagle_overlap_gate(sa)  # must not raise
        self.assertFalse(sa.disable_overlap_schedule)

    def test_reject_dp_attention(self):
        sa = _server_args(enable_dp_attention=True)
        with self.assertRaises(ValueError):
            _reject_unsupported_tpu_spec_v2(sa)

    def test_reject_adaptive(self):
        sa = _server_args(speculative_adaptive=True)
        with self.assertRaises(ValueError):
            _reject_unsupported_tpu_spec_v2(sa)

    def test_reject_return_hidden_states(self):
        sa = _server_args(enable_return_hidden_states=True)
        with self.assertRaises(ValueError):
            _reject_unsupported_tpu_spec_v2(sa)

    def test_reject_topk_gt_1(self):
        sa = _server_args(speculative_eagle_topk=4)
        with self.assertRaises(ValueError):
            _reject_unsupported_tpu_spec_v2(sa)

    def test_reject_non_eagle3_algorithm(self):
        # First pass is EAGLE3-only: STANDALONE and plain EAGLE are rejected.
        for algo in ("STANDALONE", "EAGLE"):
            sa = _server_args(speculative_algorithm=algo)
            with self.assertRaises(ValueError, msg=algo):
                _reject_unsupported_tpu_spec_v2(sa)

    def test_active_gate_raises_on_dp_attention(self):
        # End-to-end: the gate calls the reject path when V2 is active on TPU.
        sa = _server_args(enable_spec_v2_tpu=True, enable_dp_attention=True)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            with self.assertRaises(ValueError):
                _apply_eagle_overlap_gate(sa)

    def test_eagle3_allowed(self):
        sa = _server_args(speculative_algorithm="EAGLE3", enable_spec_v2_tpu=True)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            _apply_eagle_overlap_gate(sa)  # EAGLE3 is the supported config
        self.assertFalse(sa.disable_overlap_schedule)


class TestTpuSpecV2RequestRejection(CustomTestCase):
    """Request-time grammar/logprob rejection when TPU SpecV2 is active."""

    def test_active_detection(self):
        active = _server_args(enable_spec_v2_tpu=True, disable_overlap_schedule=False)
        self.assertTrue(is_tpu_spec_v2_active(active))
        # Not active when overlap disabled, flag off, non-tpu, or non-EAGLE3.
        self.assertFalse(
            is_tpu_spec_v2_active(_server_args(disable_overlap_schedule=True))
        )
        self.assertFalse(is_tpu_spec_v2_active(_server_args(enable_spec_v2_tpu=False)))
        self.assertFalse(
            is_tpu_spec_v2_active(_server_args(device="cuda", enable_spec_v2_tpu=True))
        )

    def _active(self):
        return _server_args(enable_spec_v2_tpu=True, disable_overlap_schedule=False)

    def test_reject_return_logprob(self):
        with self.assertRaises(ValueError):
            reject_request_for_tpu_spec_v2(self._active(), {}, return_logprob=True)

    def test_reject_grammar_params(self):
        for key in ("json_schema", "regex", "ebnf", "structural_tag"):
            with self.assertRaises(ValueError, msg=key):
                reject_request_for_tpu_spec_v2(
                    self._active(), {key: "x"}, return_logprob=False
                )

    def test_plain_request_allowed(self):
        # No grammar, no logprob -> must not raise.
        reject_request_for_tpu_spec_v2(
            self._active(), {"temperature": 0.0}, return_logprob=False
        )

    def test_not_rejected_when_inactive(self):
        # V1 (overlap disabled): the same request must NOT be rejected.
        inactive = _server_args(disable_overlap_schedule=True)
        reject_request_for_tpu_spec_v2(
            inactive, {"json_schema": "x"}, return_logprob=True
        )


if __name__ == "__main__":
    unittest.main()
