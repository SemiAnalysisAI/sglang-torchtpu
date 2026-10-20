"""MoEGate on torch_tpu: bf16 router weights and activations give fp32 logits
(bf16 x bf16 -> fp32, as linear_bf16_fp32 on CUDA), so Kimi-K3's 896-expert
routing matches an fp32 router."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase

try:
    import torch
    import torch_tpu  # noqa: F401

    # torch.tpu.is_available() reports False under pytest even with chips
    # attached; a tensor on the device is the test that matters.
    torch.empty(1, device="tpu")
    _HAS_TPU = True
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

E, H, K = 896, 7168, 8


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestMoEGateFp32Logits(CustomTestCase):
    def _gate(self):
        import sglang.srt.models.deepseek_v2 as dv2

        config = SimpleNamespace(
            n_routed_experts=E, hidden_size=H, topk_method="greedy"
        )
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            gate = dv2.MoEGate(config=config, quant_config=None)
        finally:
            torch.set_default_dtype(default_dtype)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            gate.weight.copy_(
                (torch.randn(E, H, generator=generator) * 0.02).to(torch.bfloat16)
            )
        return gate.to("tpu")

    def test_logits_are_fp32_and_route_like_fp32(self):
        import sglang.srt.models.deepseek_v2 as dv2

        gate = self._gate()
        self.assertEqual(gate.weight.dtype, torch.bfloat16)
        generator = torch.Generator().manual_seed(1)
        hidden = torch.randn(256, H, generator=generator).to(torch.bfloat16)
        bias = torch.randn(E, generator=generator) * 0.05
        exec_cfg = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        )
        with patch.object(dv2, "get_exec", return_value=exec_cfg):
            logits = gate(hidden.to("tpu")).cpu()
        self.assertEqual(logits.dtype, torch.float32)

        reference = hidden.double() @ gate.weight.cpu().double().t()
        self.assertLess(float((logits.double() - reference).abs().max()), 1e-4)

        def experts(x):
            return torch.topk(torch.sigmoid(x.float()) + bias, K).indices.sort().values

        torch.testing.assert_close(experts(logits), experts(reference))


if __name__ == "__main__":
    unittest.main()
