"""CPU-side logic of the two-level Pallas all-reduce and Kimi-K3's use of it.

The kernel itself runs on TPU in test_twolevel_allreduce.py; these cover the
host split, the eligibility gate, the None fallbacks, registration, and the
Kimi-K3 construction-time switch and merge gate.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import twolevel_allreduce as tl

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORLD = 32


def _bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


class TestHostSplit(CustomTestCase):
    def test_derived_from_nnodes(self):
        self.assertEqual(tl.local_size(32, 4), 8)
        self.assertEqual(tl.local_size(8, 1), 8)
        self.assertEqual(tl.local_size(16, 2), 8)

    def test_no_split_when_ranks_do_not_divide(self):
        self.assertEqual(tl.local_size(30, 4), 0)
        self.assertEqual(tl.local_size(32, 0), 0)


class TestEligibility(CustomTestCase):
    def test_single_token_bf16_lane_aligned(self):
        self.assertTrue(tl.eligible(_bf16(1, 7168)))

    def test_rejects_others(self):
        for t in (
            _bf16(4, 7168),  # more than one token
            torch.zeros(1, 7168, dtype=torch.float32),
            _bf16(1, 7100),  # not a 128 multiple
            _bf16(7168),
            _bf16(1, tl.MAX_ELEMS + 128),
        ):
            with self.subTest(shape=tuple(t.shape), dtype=t.dtype):
                self.assertFalse(tl.eligible(t))


class TestPluginEntryPoints(CustomTestCase):
    def setUp(self):
        self.ops = {
            (WORLD, 1): mock.Mock(return_value="one"),
            (WORLD, 2): mock.Mock(return_value=("a", "b")),
        }
        patches = [
            mock.patch.object(tl, "_READY", True),
            mock.patch.object(tl, "_WORLD", WORLD),
            mock.patch.object(tl, "_OPS", self.ops),
            mock.patch(
                "sglang.srt.distributed.parallel_state."
                "get_tensor_model_parallel_world_size",
                return_value=WORLD,
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_tp_all_reduce_takes_single_tokens(self):
        self.assertEqual(tl.tp_all_reduce(_bf16(1, 7168)), "one")
        self.assertIsNone(tl.tp_all_reduce(_bf16(4, 7168)))

    def test_merged_takes_the_kernel(self):
        got = tl.merged_moe_all_reduce(_bf16(1, 3584), _bf16(1, 7168))
        self.assertEqual(got, ("a", "b"))

    def test_merged_returns_none_when_it_does_not_apply(self):
        with self.subTest("not registered"), mock.patch.object(tl, "_READY", False):
            self.assertIsNone(tl.merged_moe_all_reduce(_bf16(1, 3584), _bf16(1, 7168)))
        with self.subTest("several tokens"):
            self.assertIsNone(tl.merged_moe_all_reduce(_bf16(4, 3584), _bf16(4, 7168)))


class TestRegistration(CustomTestCase):
    def test_off_by_default(self):
        with (
            mock.patch.object(tl, "register") as reg,
            mock.patch.object(tl, "_READY", False),
        ):
            tl.register_from_parallel_state()
            self.assertFalse(tl.enabled())
        reg.assert_not_called()

    def _register(self, world, nnodes):
        from sglang_tpu.envs import envs as tpu_envs

        with (
            tpu_envs.SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE.override(True),
            mock.patch.object(tl, "register") as reg,
            mock.patch("torch.distributed.get_world_size", return_value=world),
            mock.patch(
                "sglang.srt.distributed.parallel_state."
                "get_tensor_model_parallel_world_size",
                return_value=world,
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=SimpleNamespace(nnodes=nnodes),
            ),
        ):
            tl.register_from_parallel_state()
        return reg

    def test_registers_with_the_host_split(self):
        self._register(WORLD, 4).assert_called_once_with(WORLD, 8)

    def test_no_host_split_registers_nothing(self):
        self._register(30, 4).assert_not_called()


class TestKimiK3Switch(CustomTestCase):
    """kimi_k3._tpu_twolevel_all_reduce: the plugin module only on TPU with the
    ops built, read once at construction."""

    def _switch(self, is_tpu, ready):
        from sglang.srt.models import kimi_k3

        with (
            mock.patch.object(kimi_k3, "_is_tpu", is_tpu),
            mock.patch.object(tl, "_READY", ready),
        ):
            return kimi_k3._tpu_twolevel_all_reduce()

    def test_on_when_built(self):
        self.assertIs(self._switch(True, True), tl)

    def test_off_otherwise(self):
        self.assertIsNone(self._switch(True, False))
        self.assertIsNone(self._switch(False, True))


class TestKimiK3MergeGate(CustomTestCase):
    def _moe(self, **overrides):
        attrs = dict(
            _merged_tp_all_reduce=tl.merged_moe_all_reduce,
            tp_size=WORLD,
            _routed_needs_reduce=True,
            _shared_experts_tp1=False,
            _shared_experts_tp_comm=False,
        )
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    def _can_merge(self, moe, shared="shared", event=None):
        from sglang.srt.models import kimi_k3

        return kimi_k3.KimiK3MoE._can_merge_moe_all_reduce(moe, shared, event)

    def test_both_need_a_tp_reduce(self):
        self.assertTrue(self._can_merge(self._moe()))

    def test_otherwise_keeps_two_reduces(self):
        cases = {
            "two-level off": (self._moe(_merged_tp_all_reduce=None), "shared", None),
            "no shared output": (self._moe(), None, None),
            "shared on a side stream": (self._moe(), "shared", "event"),
            "tp1": (self._moe(tp_size=1), "shared", None),
            "routed already reduced": (
                self._moe(_routed_needs_reduce=False),
                "shared",
                None,
            ),
            "tp1 shared experts": (self._moe(_shared_experts_tp1=True), "shared", None),
            "shared reduce fused": (
                self._moe(_shared_experts_tp_comm=True),
                "shared",
                None,
            ),
        }
        for name, (moe, shared, event) in cases.items():
            with self.subTest(name):
                self.assertFalse(self._can_merge(moe, shared, event))


if __name__ == "__main__":
    unittest.main()
