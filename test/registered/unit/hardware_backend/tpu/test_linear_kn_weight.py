"""TPU KN linear weight: outputs match the [out, in] path on device."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit")


@unittest.skipUnless(
    hasattr(torch, "accelerator") and torch.accelerator.is_available(), "requires TPU"
)
class TestLinearKnWeightTpu(unittest.TestCase):
    def _layer(self, k, n, bias):
        torch.manual_seed(0)
        layer = ReplicatedLinear(k, n, bias=bias, params_dtype=torch.bfloat16)
        layer = layer.to("tpu")
        with torch.no_grad():
            layer.weight.copy_((torch.randn(n, k) / k**0.5).to("tpu", torch.bfloat16))
            if bias:
                layer.bias.copy_(torch.randn(n).to("tpu", torch.bfloat16))
        layer.allow_kn_weight = True
        return layer

    @torch.no_grad()
    def test_matches_nk_path(self):
        # Decode GEMV, a prefill-sized batch, and a K3-like latent shape.
        for t, k, n, bias in [
            (1, 512, 768, False),
            (64, 512, 768, True),
            (8, 896, 256, False),
        ]:
            with self.subTest(t=t, k=k, n=n, bias=bias):
                layer = self._layer(k, n, bias)
                x = torch.randn(t, k).to("tpu", torch.bfloat16)
                ref, _ = layer(x)
                ref = ref.cpu()
                with envs.SGLANG_TPU_ENABLE_LINEAR_KN_WEIGHT.override(True):
                    layer.quant_method.process_weights_after_loading(layer)
                self.assertIsNotNone(layer.weight_kn)
                self.assertEqual(tuple(layer.weight.shape), (n, k))
                out, _ = layer(x)
                torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)

    @torch.no_grad()
    def test_reload_writes_through_on_device(self):
        layer = self._layer(512, 768, False)
        with envs.SGLANG_TPU_ENABLE_LINEAR_KN_WEIGHT.override(True):
            layer.quant_method.process_weights_after_loading(layer)
        new_w = torch.randn(768, 512).to(torch.bfloat16)
        layer.weight_loader(layer.weight, new_w.to("tpu"))
        x = torch.randn(4, 512).to(torch.bfloat16)
        out, _ = layer(x.to("tpu"))
        ref = F.linear(x.to("tpu"), new_w.to("tpu")).cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
