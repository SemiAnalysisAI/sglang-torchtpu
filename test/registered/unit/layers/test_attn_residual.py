import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import attn_residual
from sglang.srt.layers.attn_residual import (
    _aggregate,
    _aggregate_fused_add,
    _supports_attn_res_tma,
    aggregate_stream_torch,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_KERNEL_MODULE = "sglang.srt.hardware_backend.tpu.kimi_attn_residual"


class TestAttnResidual(unittest.TestCase):
    def test_tma_capability_gate(self):
        self.assertFalse(_supports_attn_res_tma((9, 0)))
        self.assertTrue(_supports_attn_res_tma((10, 0)))
        self.assertTrue(_supports_attn_res_tma((10, 3)))
        self.assertTrue(_supports_attn_res_tma((11, 0)))
        self.assertFalse(_supports_attn_res_tma((12, 0)))
        self.assertTrue(_supports_attn_res_tma((13, 0)))

    def test_aggregate_on_tpu_matches_reference(self):
        """TPU uses the eager reference."""
        torch.manual_seed(0)
        T, H, nvb = 3, 1024, 2
        score_proj = ReplicatedLinear(H, 1, bias=False)
        score_norm, out_norm = RMSNorm(H), RMSNorm(H)
        with torch.no_grad():
            score_proj.weight.normal_()
        prefix_sum, bank = torch.randn(T, H), torch.randn(T, nvb + 1, H)
        with patch.object(attn_residual, "_is_tpu", True):
            out = _aggregate(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)
        expected = out_norm(
            aggregate_stream_torch(prefix_sum, bank, nvb, score_proj, score_norm)
        )
        torch.testing.assert_close(out, expected)

    def test_tpu_decode_kernel_routing(self):
        """The opt-in TPU decode kernel: off TPU or with the flag unset the gate
        never imports the kernel module (it needs JAX); on TPU with the flag the
        kernel's shape gate decides, and both aggregation entry points route to
        the kernel when it holds."""
        torch.manual_seed(0)
        T, H, nvb = 4, 1024, 2
        score_proj = ReplicatedLinear(H, 1, bias=False)
        score_norm, out_norm = RMSNorm(H), RMSNorm(H)
        with torch.no_grad():
            score_proj.weight.normal_()
        prefix, bank = torch.randn(T, H), torch.randn(T, nvb + 1, H)

        # A None entry in sys.modules makes importing the kernel module raise.
        with patch.dict(sys.modules, {_KERNEL_MODULE: None}):
            for tpu, flag in ((True, False), (False, True)):
                with (
                    patch.object(attn_residual, "_is_tpu", tpu),
                    envs.SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL.override(flag),
                ):
                    self.assertFalse(
                        attn_residual._use_tpu_decode(
                            prefix, None, bank, score_norm, out_norm
                        )
                    )

        kernel = types.ModuleType(_KERNEL_MODULE)
        gate = kernel.supports_attn_res_decode = MagicMock(return_value=False)
        with (
            patch.dict(sys.modules, {_KERNEL_MODULE: kernel}),
            patch.object(attn_residual, "_is_tpu", True),
            patch.object(
                attn_residual,
                "_aggregate_tpu_decode",
                side_effect=lambda a, b, *_: (a, a if b is None else a + b),
            ) as decode,
            envs.SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL.override(True),
        ):
            # Shape gate closed: the eager reference.
            out = _aggregate(prefix, bank, nvb, score_proj, score_norm, out_norm)
            decode.assert_not_called()
            expected = out_norm(
                aggregate_stream_torch(prefix, bank, nvb, score_proj, score_norm)
            )
            torch.testing.assert_close(out, expected)

            gate.return_value = True
            _aggregate(prefix, bank, nvb, score_proj, score_norm, out_norm)
            _aggregate_fused_add(
                prefix, prefix, bank, nvb, score_proj, score_norm, out_norm
            )
            with self.assertRaisesRegex(AssertionError, "TPU decode kernel"):
                _aggregate(
                    prefix,
                    bank,
                    nvb,
                    score_proj,
                    score_norm,
                    out_norm,
                    write_bank_row=True,
                )
        self.assertEqual(gate.call_count, 4)
        a, b, bk, sn, on = gate.call_args_list[0].args
        self.assertTrue(a is prefix and b is None and bk is bank)
        self.assertTrue(sn is score_norm and on is out_norm)
        # The pending addend reaches the shape gate.
        self.assertIs(gate.call_args_list[2].args[1], prefix)
        self.assertEqual(decode.call_count, 2)
        self.assertIsNone(decode.call_args_list[0].args[1])
        self.assertIs(decode.call_args_list[1].args[1], prefix)


if __name__ == "__main__":
    unittest.main()
