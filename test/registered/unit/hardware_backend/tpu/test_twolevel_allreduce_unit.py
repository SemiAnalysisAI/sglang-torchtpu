"""CPU-side logic of the two-level Pallas all-reduce and its core call sites.

The kernel itself runs on TPU in test_twolevel_allreduce.py; these cover the
host split, the eligibility gate, the None fallbacks, and how
GroupCoordinator.all_reduce and Kimi-K3's MoE tail use them.
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
        self.tp_group = mock.Mock(world_size=WORLD)
        self.ops = {
            (WORLD, 1): mock.Mock(return_value="one"),
            (WORLD, 2): mock.Mock(return_value=("a", "b")),
        }
        patches = [
            mock.patch.object(tl, "_READY", True),
            mock.patch.object(tl, "_WORLD", WORLD),
            mock.patch.object(tl, "_OPS", self.ops),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=self.tp_group,
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state."
                "get_tensor_model_parallel_world_size",
                return_value=WORLD,
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_all_reduce_takes_the_kernel(self):
        self.assertEqual(tl.all_reduce(self.tp_group, _bf16(1, 7168)), "one")

    def test_all_reduce_returns_none_when_it_does_not_apply(self):
        other_group = mock.Mock(world_size=WORLD)
        cases = {
            "not registered": (self.tp_group, _bf16(1, 7168), False),
            "not the TP group": (other_group, _bf16(1, 7168), True),
            "several tokens": (self.tp_group, _bf16(4, 7168), True),
        }
        for name, (group, x, ready) in cases.items():
            with self.subTest(name), mock.patch.object(tl, "_READY", ready):
                self.assertIsNone(tl.all_reduce(group, x))

    def test_merged_takes_the_kernel(self):
        got = tl.merged_moe_all_reduce(_bf16(1, 3584), _bf16(1, 7168))
        self.assertEqual(got, ("a", "b"))

    def test_merged_returns_none_when_it_does_not_apply(self):
        with self.subTest("not registered"), mock.patch.object(tl, "_READY", False):
            self.assertIsNone(tl.merged_moe_all_reduce(_bf16(1, 3584), _bf16(1, 7168)))
        with self.subTest("several tokens"):
            self.assertIsNone(tl.merged_moe_all_reduce(_bf16(4, 3584), _bf16(4, 7168)))

    def test_registration_off_by_default(self):
        with mock.patch.object(tl, "register") as reg:
            tl.register_from_parallel_state()
        reg.assert_not_called()


class TestGroupCoordinatorAllReduce(CustomTestCase):
    """GroupCoordinator.all_reduce on TPU: the plugin's result when it has one,
    otherwise the in-place custom op as before."""

    def _call(self, plugin_result):
        from sglang.srt.distributed import parallel_state as ps

        group = SimpleNamespace(world_size=WORLD, unique_name="tp:0")
        x = mock.Mock(is_cpu=False)
        with (
            mock.patch.object(ps, "_is_tpu", True),
            mock.patch.object(
                ps, "_tpu_twolevel_all_reduce", return_value=plugin_result
            ) as plugin,
            mock.patch.object(ps, "inplace_all_reduce") as inplace,
        ):
            out = ps.GroupCoordinator.all_reduce(group, x)
        plugin.assert_called_once_with(group, x)
        return out, x, inplace

    def test_uses_the_plugin_result(self):
        out, _, inplace = self._call("reduced")
        self.assertEqual(out, "reduced")
        inplace.assert_not_called()

    def test_falls_back_to_the_custom_op(self):
        out, x, inplace = self._call(None)
        self.assertIs(out, x)
        inplace.assert_called_once_with(x, group_name="tp:0")


class TestKimiK3MergeGate(CustomTestCase):
    def _moe(self, **overrides):
        attrs = dict(
            tp_size=WORLD,
            _routed_needs_reduce=True,
            _shared_experts_tp1=False,
            _shared_experts_tp_comm=False,
        )
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    def _can_merge(self, moe, shared="shared", event=None, is_tpu=True):
        from sglang.srt.models import kimi_k3

        with mock.patch.object(kimi_k3, "_is_tpu", is_tpu):
            return kimi_k3.KimiK3MoE._can_merge_moe_all_reduce(moe, shared, event)

    def test_both_need_a_tp_reduce(self):
        self.assertTrue(self._can_merge(self._moe()))

    def test_otherwise_keeps_two_reduces(self):
        cases = {
            "not TPU": (self._moe(), "shared", None, False),
            "no shared output": (self._moe(), None, None, True),
            "shared on a side stream": (self._moe(), "shared", "event", True),
            "tp1": (self._moe(tp_size=1), "shared", None, True),
            "routed already reduced": (
                self._moe(_routed_needs_reduce=False),
                "shared",
                None,
                True,
            ),
            "tp1 shared experts": (
                self._moe(_shared_experts_tp1=True),
                "shared",
                None,
                True,
            ),
            "shared reduce fused": (
                self._moe(_shared_experts_tp_comm=True),
                "shared",
                None,
                True,
            ),
        }
        for name, (moe, shared, event, is_tpu) in cases.items():
            with self.subTest(name):
                self.assertFalse(self._can_merge(moe, shared, event, is_tpu))


if __name__ == "__main__":
    unittest.main()
