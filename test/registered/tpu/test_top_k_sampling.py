"""Correctness tests for the exact TPU top-k sampling primitive."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestTpuTopK(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_tpu  # noqa: F401

            from sglang.srt.hardware_backend.tpu.env_override import (
                apply_tpu_env_overrides,
            )

            # CustomTestCase intentionally keeps JAX on CPU for exported-op
            # tracing. Probe the registered TorchTPU device instead: checking
            # jax.devices() here would therefore skip on every TPU CI worker.
            apply_tpu_env_overrides()
            torch.zeros(1, device="tpu")
        except Exception as exc:
            raise unittest.SkipTest(f"TorchTPU device not available: {exc}")

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
