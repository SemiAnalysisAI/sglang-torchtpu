"""CPU guard/dispatch/registration checks; TPU arithmetic qualified separately."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.tpu import glm_routing_gather as backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlmRoutingGather(unittest.TestCase):
    def test_metadata_guard(self):
        scores = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), shape=(1, 256), dtype=torch.float32
        )
        bias = torch.empty(256, dtype=torch.float32)
        kwargs = dict(
            topk=8, renormalize=True, num_expert_group=1, topk_group=1,
            num_fused_shared_experts=0, apply_routed_scaling_factor_on_output=False,
        )
        self.assertTrue(backend.can_use_glm_routing_gather(scores, bias, **kwargs))
        for key, value in (
            ("topk", 4), ("renormalize", False), ("num_expert_group", 2),
            ("topk_group", 2), ("num_fused_shared_experts", 1),
            ("apply_routed_scaling_factor_on_output", True),
        ):
            with self.subTest(key=key):
                self.assertFalse(backend.can_use_glm_routing_gather(
                    scores, bias, **(kwargs | {key: value})
                ))
        for name, value in (("shape", (8, 256)), ("shape", (1, 128)),
                            ("dtype", torch.bfloat16),
                            ("device", SimpleNamespace(type="cuda"))):
            modified = SimpleNamespace(**(vars(scores) | {name: value}))
            self.assertFalse(
                backend.can_use_glm_routing_gather(modified, bias, **kwargs)
            )
        for invalid_bias in (bias.bfloat16(), torch.empty(128)):
            self.assertFalse(
                backend.can_use_glm_routing_gather(scores, invalid_bias, **kwargs)
            )

    def test_topk_dispatch_preserves_ids_and_normalizes_once(self):
        # Execute the real helper body without importing GPU-specific kernels.
        source = Path(backend.__file__).parents[2] / "layers/moe/topk.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "biased_grouped_topk_impl")
        function.decorator_list = []
        flag = Mock(return_value=True)
        scope = dict(torch=torch, Optional=Optional, _is_tpu=False,
                     _RENORMALIZE_SUM_EPSILON=1e-20,
                     envs=SimpleNamespace(
                         SGLANG_TPU_GLM_ROUTING_GATHER=SimpleNamespace(get=flag)
                     ))
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(source), "exec"), scope)
        fn = scope["biased_grouped_topk_impl"]
        logits = torch.randn(1, 256, generator=torch.Generator().manual_seed(52))
        bias = torch.randn(256, generator=torch.Generator().manual_seed(53)) * 0.01
        args = (logits, logits, bias, 8, True, 1, 1)
        expected = fn(*args)

        def gather(scores, ids):
            selected = scores.gather(1, ids)
            return selected / (selected.sum(-1, keepdim=True) + 1e-20)

        scope["_is_tpu"] = True
        with (
            patch.object(backend, "can_use_glm_routing_gather", return_value=True),
            patch.object(
                backend, "glm_routing_gather_normalize", side_effect=gather
            ) as op,
        ):
            actual = fn(*args)
            op.assert_called_once()
            for value, target in zip(actual, expected):
                torch.testing.assert_close(value, target, rtol=0, atol=0)
            flag.return_value = False
            op.reset_mock()
            fallback = fn(*args)
            op.assert_not_called()
            for value, target in zip(fallback, expected):
                torch.testing.assert_close(value, target, rtol=0, atol=0)

    def test_registration_and_wrapper(self):
        fake_jax = ModuleType("jax")
        fake_jax.Array = object
        adapter_name = "sglang.srt.hardware_backend.tpu.pallas_jax_op"
        fake_adapter = ModuleType(adapter_name)
        sentinel = Mock()
        fake_adapter.cpu_export_jax_op = Mock(return_value=sentinel)
        with (patch.object(backend, "_GATHER_OP", None),
              patch.dict(sys.modules, {"jax": fake_jax, adapter_name: fake_adapter})):
            backend.register_glm_routing_gather_op()
            backend.register_glm_routing_gather_op()
            fake_adapter.cpu_export_jax_op.assert_called_once()
            self.assertEqual(fake_adapter.cpu_export_jax_op.call_args.args[0],
                             "sglang_tpu::glm_routing_gather_normalize")
            scores = torch.empty(1, 256, device="meta")
            ids = torch.empty(1, 8, dtype=torch.int64, device="meta")
            fake = sentinel.register_fake.call_args.args[0]
            output = fake(scores, ids)
            self.assertEqual(output.shape, (1, 8))
            self.assertEqual(output.dtype, torch.float32)
            backend.glm_routing_gather_normalize(scores, ids)
            self.assertIs(sentinel.call_args.args[0], scores)
            self.assertEqual(sentinel.call_args.args[1].dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
