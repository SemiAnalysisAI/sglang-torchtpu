"""Gating of the Kimi-K3 TPU fused router (SGLANG_TPU_K3_FUSED_GATE).
Kernel numerics: test/registered/tpu/unit/test_kimi_k3_router_topk.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import topk
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.test_utils import CustomTestCase


def _owner(config_ok=True, dtype=torch.bfloat16, hidden=256):
    bias = torch.randn(16)
    cfg = SimpleNamespace(top_k=4, correction_bias=bias)
    return SimpleNamespace(
        gate=SimpleNamespace(
            weight=torch.nn.Parameter(torch.randn(16, hidden).to(dtype)),
            e_score_correction_bias=torch.nn.Parameter(bias, requires_grad=False),
        ),
        topk=SimpleNamespace(topk_config=cfg),
        _routing_config_ok=config_ok,
        _tpu_route_kwargs=dict(renormalize=True, scale=2.5, eps=1e-20),
        _tpu_gate_kn=None,
        layer_idx=0,
    )


class TestK3FusedRouterGating(CustomTestCase):
    def setUp(self):
        # Stand-in for the JAX kernel module.
        self.mod = types.ModuleType("sglang.srt.layers.moe.pallas.router_topk")
        self.mod.GATE_ROUTE_MAX_ROWS = 64
        self.mod.gate_route_supported = lambda k_dim, n: k_dim % 128 == 0
        self.mod.get_gate_route_op = MagicMock()
        self.mod.gate_route_tpu = MagicMock(return_value=(None, None))
        patcher = patch.dict(sys.modules, {self.mod.__name__: self.mod})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _bind(self, owner, flag=True):
        with envs.SGLANG_TPU_K3_FUSED_GATE.override(flag):
            KimiK3MoE._bind_tpu_router(owner)
        return owner

    def test_unbound_when_off_or_unsupported(self):
        for owner, flag in (
            (_owner(), False),
            (_owner(config_ok=False), True),
            (_owner(dtype=torch.float32), True),
            (_owner(hidden=200), True),
        ):
            self.assertIsNone(self._bind(owner, flag)._tpu_gate_kn)
        self.mod.get_gate_route_op.assert_not_called()

    def test_bind_shares_gate_buffer(self):
        owner = _owner()
        before = owner.gate.weight.clone()
        self._bind(owner)
        self.assertEqual(tuple(owner._tpu_gate_kn.shape), (256, 16))
        self.assertTrue(owner._tpu_gate_kn.is_contiguous())
        torch.testing.assert_close(owner.gate.weight, before, rtol=0, atol=0)
        self.assertEqual(owner.gate.weight.data_ptr(), owner._tpu_gate_kn.data_ptr())
        self.assertEqual(owner._tpu_route_bias.shape, (1, 16))
        # Taken from the gate parameter, as its own storage (not a view).
        torch.testing.assert_close(
            owner._tpu_route_bias[0], owner.gate.e_score_correction_bias.float()
        )
        self.assertIsNone(owner._tpu_route_bias._base)
        self.mod.get_gate_route_op.assert_called_once_with(
            4, renormalize=True, scale=2.5, eps=1e-20
        )

    def test_route_covers_decode_rows_only(self):
        owner = self._bind(_owner())
        for rows, dtype in (
            (0, torch.bfloat16),
            (65, torch.bfloat16),
            (4, torch.float32),
        ):
            self.assertIsNone(
                KimiK3MoE._tpu_gate_route(owner, torch.zeros(rows, 256, dtype=dtype))
            )
        self.mod.gate_route_tpu.assert_not_called()
        with patch("sglang.srt.models.kimi_k3.build_precomputed_topk_output") as build:
            KimiK3MoE._tpu_gate_route(owner, torch.zeros(64, 256, dtype=torch.bfloat16))
        self.assertIs(build.call_args.args[2], owner.topk.topk_config)

    def test_precomputed_postprocess_noop_on_tpu(self):
        cfg = SimpleNamespace(num_fused_shared_experts=0)
        with patch.object(topk, "_is_cuda", False), patch.object(topk, "_is_tpu", True):
            self.assertTrue(topk.precomputed_topk_postprocess_is_noop(cfg))
            # A padding mask still needs select_experts.
            self.assertFalse(
                topk.precomputed_topk_postprocess_is_noop(cfg, torch.tensor(1))
            )


if __name__ == "__main__":
    unittest.main()
