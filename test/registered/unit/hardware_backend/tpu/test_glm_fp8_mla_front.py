"""CPU boundary and registration tests for the opt-in GLM front kernel."""

import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.tpu import glm_fp8_mla_front as backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlmFp8MlaFrontBoundary(unittest.TestCase):
    def test_wrapper_prepares_a_layout_and_position_dtype(self):
        # Small tensors test the boundary transformation without allocating
        # checkpoint weights. Shape qualification belongs to the dispatcher.
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        a_scale = torch.ones(3, 1)
        q_norm, kv_norm = torch.ones(2), torch.ones(1)
        b, b_scale = torch.ones(2, 4), torch.ones(4, 1)
        kc = torch.ones(1, 2, 3)
        positions = torch.tensor([[137]], dtype=torch.int64)
        cache = torch.ones(256, 64, dtype=torch.bfloat16)
        outputs = tuple(torch.empty(shape) for shape in (
            (1, 4, 512), (1, 4, 64), (1, 1, 512), (1, 1, 64)
        ))
        op = Mock(return_value=outputs)
        with (
            patch.object(backend, "_get_front_op", return_value=op),
            patch.object(backend.envs.SGLANG_TPU_GLM_FP8_MLA_FLAT_SCALES, "get", return_value=False),
        ):
            actual = backend.glm_fp8_mla_front(
                x, a, a_scale, q_norm, kv_norm, b, b_scale, kc,
                positions, cache, 1e-5, 2e-5,
            )
        self.assertIs(actual, outputs)
        passed = op.call_args.args
        self.assertIs(passed[0], x)
        self.assertTrue(passed[1].is_contiguous())
        torch.testing.assert_close(passed[1], a.t(), rtol=0, atol=0)
        for index, original in enumerate(
            (a_scale, q_norm, kv_norm, b, b_scale, kc), start=2
        ):
            self.assertIs(passed[index], original)
        self.assertEqual(passed[8].shape, (1,))
        self.assertEqual(passed[8].dtype, torch.int32)
        self.assertEqual(passed[8].tolist(), [137])
        self.assertIs(passed[9], cache)
        self.assertEqual(passed[10:], (1e-5, 2e-5, False))

    def test_logical_kc_view_keeps_storage_and_sets_static_argument(self):
        physical = torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3)
        logical = physical.transpose(-1, -2)
        arguments = [
            torch.ones(1, 2), torch.ones(2, 3), torch.ones(3, 1),
            torch.ones(2), torch.ones(1), torch.ones(2, 4), torch.ones(4, 1),
            logical, torch.tensor([0]), torch.ones(8, 64), 1e-5, 1e-5,
        ]
        op = Mock()
        with patch.object(backend, "_get_front_op", return_value=op):
            backend.glm_fp8_mla_front(*arguments, kc_logical=True)
        self.assertIs(op.call_args.args[7], logical)
        self.assertEqual(op.call_args.args[7].data_ptr(), physical.data_ptr())
        self.assertFalse(op.call_args.args[7].is_contiguous())
        self.assertIs(op.call_args.args[12], True)

    def test_flat_scale_views_preserve_values_and_storage(self):
        a_scale = torch.arange(12, dtype=torch.float32).reshape(12, 1)
        b_scale = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        arguments = [
            torch.ones(1, 2), torch.ones(2, 12), a_scale,
            torch.ones(2), torch.ones(1), torch.ones(2, 8), b_scale,
            torch.ones(1, 2, 3), torch.tensor([0]), torch.ones(8, 64),
            1e-5, 1e-5,
        ]
        op = Mock()
        with (
            patch.object(backend, "_get_front_op", return_value=op),
            patch.object(backend.envs.SGLANG_TPU_GLM_FP8_MLA_FLAT_SCALES, "get", return_value=True),
        ):
            backend.glm_fp8_mla_front(*arguments)
        for index, source in ((2, a_scale), (6, b_scale)):
            value = op.call_args.args[index]
            self.assertEqual(value.shape, (source.numel(),))
            self.assertEqual(value.dtype, source.dtype)
            self.assertEqual(value.data_ptr(), source.data_ptr())
            self.assertTrue(torch.equal(value, source.flatten()))

    def test_lazy_custom_op_registration_is_once_per_process(self):
        fake_jax = ModuleType("jax")
        fake_jax.Array = object
        adapter_name = "sglang.srt.hardware_backend.tpu.pallas_jax_op"
        fake_adapter = ModuleType(adapter_name)
        sentinel = Mock()

        def register(name, function):
            # Encourage simultaneous callers so the test exercises the lock.
            time.sleep(0.01)
            import inspect

            parameter = inspect.signature(function).parameters["kc_logical"]
            self.assertIs(parameter.annotation, bool)
            self.assertIs(parameter.default, False)
            return sentinel

        fake_adapter.cpu_export_jax_op = Mock(side_effect=register)
        with (
            patch.object(backend, "_FRONT_OP", None),
            patch.dict(sys.modules, {"jax": fake_jax, adapter_name: fake_adapter}),
            ThreadPoolExecutor(max_workers=8) as workers,
        ):
            outputs = list(workers.map(lambda _: backend._get_front_op(), range(16)))
            self.assertTrue(all(output is sentinel for output in outputs))
            fake_adapter.cpu_export_jax_op.assert_called_once()
            self.assertEqual(
                fake_adapter.cpu_export_jax_op.call_args.args[0],
                "sglang_tpu::glm_fp8_mla_front",
            )
            sentinel.register_fake.assert_called_once()
            fake = sentinel.register_fake.call_args.args[0]
            meta_x = torch.empty(1, 6144, dtype=torch.bfloat16, device="meta")
            fake_outputs = fake(meta_x, *([None] * 11))
            self.assertEqual(
                [tuple(value.shape) for value in fake_outputs],
                [(1, 4, 512), (1, 4, 64), (1, 1, 512), (1, 1, 64)],
            )
            self.assertTrue(all(value.dtype == torch.bfloat16 for value in fake_outputs))


if __name__ == "__main__":
    unittest.main()
