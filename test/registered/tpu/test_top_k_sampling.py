"""Correctness tests for the exact TPU top-k sampling primitive."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax

    _HAS_TPU = any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover
    _HAS_TPU = False


@unittest.skipUnless(_HAS_TPU, "Requires JAX and a visible TPU backend")
class TestTpuTopK(CustomTestCase):
    def test_exact_top_k_50_at_serving_vocab(self):
        batch, vocab, k = 2, 202_048, 50
        generator = torch.Generator().manual_seed(7)
        values_cpu = torch.randn(batch, vocab, generator=generator)
        expected_values, expected_indices = torch.topk(values_cpu, k, dim=-1)

        from sglang.srt.hardware_backend.tpu.top_k import tpu_top_k

        actual_values, actual_indices = tpu_top_k(values_cpu.to("tpu"), k)
        actual_values = actual_values.cpu()
        actual_indices = actual_indices.cpu().to(torch.int64)

        torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual_indices, expected_indices))


if __name__ == "__main__":
    unittest.main()
