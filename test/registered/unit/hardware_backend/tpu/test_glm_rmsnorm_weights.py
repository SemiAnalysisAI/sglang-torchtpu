"""CPU arithmetic, buffer lifetime and weight-reload coverage for GLM norms."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.tpu import glm_rmsnorm_weights as cache
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlmRmsnormWeights(unittest.TestCase):
    def norm(self, width=6144):
        norm = RMSNorm(width, eps=1e-5, weight_dtype=torch.bfloat16)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(318)
            norm.weight.copy_(1 + torch.randn(width, generator=generator) * 0.2)
        return norm

    def test_native_arithmetic_and_original_parameter_are_preserved(self):
        norm = self.norm()
        original = norm.weight
        generator = torch.Generator().manual_seed(121)
        pairs = [
            (torch.randn(rows, 6144, generator=generator) * scale).bfloat16()
            for rows in (1, 3)
            for scale in (1e-14, 1.0, 1000.0)
        ]
        expected = []
        for x in pairs:
            expected.extend((norm.forward_native(x), norm.forward_native(x, -x)))
        cache._refresh_norm_weight(norm)
        actual = []
        for x in pairs:
            actual.extend((norm.forward_native(x), norm.forward_native(x, -x)))
        for target, result in zip(expected, actual):
            torch.testing.assert_close(result, target, rtol=0, atol=0)
        self.assertIs(norm.weight, original)
        self.assertEqual(norm.weight.dtype, torch.bfloat16)
        self.assertEqual(norm._tpu_weight_fp32.dtype, torch.float32)
        self.assertEqual(list(norm.state_dict()), ["weight"])
        self.assertEqual(list(dict(norm.named_buffers())), ["_tpu_weight_fp32"])

    def test_cached_operand_only_for_rank_two_single_row_bf16(self):
        norm = self.norm()
        cases = [
            torch.ones((1, 6144), dtype=torch.bfloat16),
            torch.ones((3, 6144), dtype=torch.bfloat16),
            torch.ones((1, 3, 6144), dtype=torch.bfloat16),
            torch.ones((1, 1, 6144), dtype=torch.bfloat16),
            torch.ones((6144,), dtype=torch.bfloat16),
            torch.ones((1, 6144), dtype=torch.float32),
        ]
        expected = [norm.forward_native(x) for x in cases]
        cache._refresh_norm_weight(norm)
        # Deliberately different values reveal which operand was selected.
        # Supported loaders never leave the real cache different from weight.
        norm._tpu_weight_fp32.zero_()
        torch.testing.assert_close(
            norm.forward_native(cases[0]), torch.zeros_like(expected[0]), rtol=0, atol=0
        )
        for x, target in zip(cases[1:], expected[1:]):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                torch.testing.assert_close(norm.forward_native(x), target, rtol=0, atol=0)

    def test_refresh_keeps_compiled_buffer_reference_and_reload_hook(self):
        norm = self.norm()
        cache._refresh_norm_weight(norm)
        compiled_reference = norm._tpu_weight_fp32
        pointer = compiled_reference.data_ptr()
        with torch.no_grad():
            norm.weight.add_(0.125)
        cache._refresh_norm_weight(norm)
        self.assertIs(norm._tpu_weight_fp32, compiled_reference)
        self.assertEqual(norm._tpu_weight_fp32.data_ptr(), pointer)
        torch.testing.assert_close(
            compiled_reference, norm.weight.float(), rtol=0, atol=0
        )

        norm.register_load_state_dict_post_hook(cache._refresh_after_load_state_dict)
        replacement = torch.full((6144,), 0.75, dtype=torch.bfloat16)
        norm.load_state_dict({"weight": replacement})
        self.assertIs(norm._tpu_weight_fp32, compiled_reference)
        torch.testing.assert_close(
            compiled_reference, replacement.float(), rtol=0, atol=0
        )
        self.assertEqual(norm.weight.dtype, torch.bfloat16)

    def test_wrong_cache_metadata_requires_rebuild(self):
        norm = self.norm()
        norm.register_buffer("_tpu_weight_fp32", torch.empty(4), persistent=False)
        with self.assertRaisesRegex(RuntimeError, "rebuild the model"):
            cache._refresh_norm_weight(norm)

    def test_tpu_metadata_selection_skips_qa_kv_and_unsupported_norms(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            torch.utils.rename_privateuse1_backend("tpu")
        mode = FakeTensorMode()
        stub = SimpleNamespace(is_initialized=lambda: False, current_device=lambda: 0)

        def fake_norm(width=6144):
            norm = self.norm(width)
            tensor = FakeTensor(
                mode,
                torch.empty(width, dtype=torch.bfloat16, device="meta"),
                torch.device("tpu"),
            )
            norm.weight = torch.nn.Parameter(tensor)
            return norm

        with patch.object(torch, "tpu", stub, create=True):
            layer = torch.nn.Module()
            layer.input_layernorm = fake_norm()
            layer.post_attention_layernorm = fake_norm()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_a_layernorm = fake_norm(2048)
            layer.self_attn.kv_a_layernorm = fake_norm(512)
            inner = torch.nn.Module()
            inner.layers = torch.nn.ModuleList([layer])
            inner.norm = fake_norm()
            model = SimpleNamespace(model=inner, _tpu_glm_rmsnorm_fp32_weights=True)
            with mode:
                self.assertEqual(cache.refresh_glm_rmsnorm_weights(model), 3)
            buffers = dict(inner.named_buffers())
            self.assertEqual(len(buffers), 3)
            self.assertTrue(
                all(value.dtype == torch.float32 for value in buffers.values())
            )
            self.assertFalse(hasattr(layer.self_attn.q_a_layernorm, "_tpu_weight_fp32"))
            self.assertFalse(
                hasattr(layer.self_attn.kv_a_layernorm, "_tpu_weight_fp32")
            )
            hooks = len(layer.input_layernorm._load_state_dict_post_hooks)
            # CPU-only FakeTensor has no TPU copy_ device implementation. The
            # real copy/identity behavior is covered above with CPU tensors.
            with patch.object(cache, "_refresh_norm_weight") as refresh:
                self.assertEqual(cache.refresh_glm_rmsnorm_weights(model), 3)
                self.assertEqual(refresh.call_count, 3)
            self.assertEqual(
                len(layer.input_layernorm._load_state_dict_post_hooks), hooks
            )
            model._tpu_glm_rmsnorm_fp32_weights = False
            self.assertEqual(cache.refresh_glm_rmsnorm_weights(model), 0)

        # Device guard keeps a CPU model unchanged even with a manually set marker.
        cpu_inner = torch.nn.Module()
        cpu_inner.layers = torch.nn.ModuleList([])
        cpu_inner.norm = self.norm()
        cpu_model = SimpleNamespace(model=cpu_inner, _tpu_glm_rmsnorm_fp32_weights=True)
        self.assertEqual(cache.refresh_glm_rmsnorm_weights(cpu_model), 0)
        self.assertFalse(hasattr(cpu_inner.norm, "_tpu_weight_fp32"))

    def test_direct_and_custom_tensor_updates_refresh_after_mutation(self):
        # Execute the actual update method with I/O mocked. This verifies the
        # bypass loaders invoke only the lightweight hook, not full post-load.
        source = (
            Path(cache.__file__).parents[2]
            / "model_executor/model_runner_components/weight_updater.py"
        )
        tree = ast.parse(source.read_text())
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "WeightUpdater"
        )
        update = copy.deepcopy(
            next(
                node
                for node in owner.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "update_weights_from_tensor"
            )
        )
        refresh = copy.deepcopy(
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_refresh_derived_weight_caches"
            )
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                update,
                refresh,
            ],
            type_ignores=[],
        )
        events = []
        model = SimpleNamespace(
            refresh_derived_weight_caches=lambda: events.append("refresh")
        )
        scope = dict(
            _unsupported_derived_weight_cache_error=lambda: None,
            monkey_patch_torch_reductions=lambda: None,
            torch=SimpleNamespace(
                get_device_module=lambda _: SimpleNamespace(current_device=lambda: 0)
            ),
            _unwrap_tensor=lambda value, **kwargs: value,
            _model_load_weights_direct=lambda *args: events.append("direct"),
            dynamic_import=lambda _: lambda *args: events.append("custom"),
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
        updater = SimpleNamespace(
            _assert_weight_cache_inactive=lambda _: None,
            device="tpu",
            tp_rank=0,
            custom_weight_loaders={"custom": True},
            get_model=lambda: model,
        )
        for kind in ("direct", "custom"):
            events.clear()
            self.assertEqual(
                scope["update_weights_from_tensor"](
                    updater, [("weight", object())], kind
                ),
                (True, "Success"),
            )
            self.assertEqual(events, [kind, "refresh"])
        scope["_refresh_derived_weight_caches"](SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
