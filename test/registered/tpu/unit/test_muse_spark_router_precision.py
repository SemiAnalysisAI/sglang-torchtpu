"""The Muse Spark router keeps fp32 accuracy on TPU, where an fp32 matmul rounds
its operands to bf16."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _detect_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1).to("tpu")
        return True
    except Exception:
        return False


@unittest.skipUnless(_detect_tpu(), "requires a TPU backend")
class TestMuseSparkRouterPrecision(CustomTestCase):
    def test_split_precision_router_matches_fp64(self):
        """Router logits within fp32 error of an fp64 product; a plain fp32
        matmul on TPU is off by about 2^-9, enough to flip near-tie experts."""
        from sglang_tpu.models.muse_spark import _split_precision_linear

        generator = torch.Generator().manual_seed(0)
        hidden = torch.randn(64, 2048, generator=generator).bfloat16()
        weight = torch.randn(256, 2048, generator=generator) * 2048**-0.5
        exact = hidden.double() @ weight.double().T

        # Compiled as in serving: the two products must not be folded into one.
        compiled = torch.compile(_split_precision_linear, fullgraph=True, dynamic=False)
        for fn in (_split_precision_linear, compiled):
            logits = fn(hidden.to("tpu"), weight.to("tpu"))
            error = (logits.cpu().double() - exact).abs().max() / exact.abs().max()
            self.assertLess(error.item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
