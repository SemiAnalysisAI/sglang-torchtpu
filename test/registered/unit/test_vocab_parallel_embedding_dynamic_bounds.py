# SPDX-License-Identifier: Apache-2.0

"""CPU/meta tests without importing SGLang's distributed/platform bootstrap."""

import ast
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/layers/vocab_parallel_embedding.py"


def load_embedding_methods():
    # Execute the actual methods with real CPU Torch, without platform imports,
    # torch.compile decorators, distributed setup, or a TPU backend probe.
    tree = ast.parse(SOURCE.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_masked_input_and_mask"
    )
    function.decorator_list = []
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VocabParallelEmbedding"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.AnnAssign)
        or (
            isinstance(n, ast.FunctionDef)
            and n.name in {"prepare_tpu_runtime_vocab_bounds", "_embed_local_shard"}
        )
    ]
    namespace = {
        "torch": torch,
        "Tuple": tuple,
        "_is_tpu": True,
        "use_symmetric_memory": lambda *a, **kw: nullcontext(),
        "get_parallel": lambda: SimpleNamespace(tp_group=None),
        "is_allocation_symmetric": lambda: False,
    }
    module = ast.Module(body=[function, cls], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace


@unittest.skipIf(torch is None, "CPU Torch is not installed")
class RuntimeBoundsTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_embedding_methods()
        self.embedding = self.ns["VocabParallelEmbedding"]
        self.mask = self.ns["get_masked_input_and_mask"]

    def layer(self, bounds=(4, 8, 2, 12, 15), device="cpu", tp_size=2):
        layer = self.embedding()
        layer.register_buffer("_tpu_runtime_vocab_bounds", None, persistent=False)
        layer.weight = torch.nn.Parameter(torch.empty(16, 3, device=device))
        layer.tp_size, layer.output_dtype = tp_size, None
        layer.shard_indices = SimpleNamespace(
            **dict(
                zip(
                    (
                        "org_vocab_start_index",
                        "org_vocab_end_index",
                        "num_org_vocab_padding",
                        "added_vocab_start_index",
                        "added_vocab_end_index",
                    ),
                    bounds,
                )
            )
        )
        layer._use_triton_embedding = lambda _: False
        layer.quant_method = SimpleNamespace(
            embedding=lambda owner, ids: torch.nn.functional.embedding(
                ids, owner.weight
            )
        )
        return layer

    def test_int64_mask_parity(self):
        for bounds in (
            (0, 4, 0, 8, 8),
            (4, 8, 0, 12, 15),
            (4, 8, 2, 12, 15),
            (4, 4, 2, 12, 12),
            (189420, 202048, 0, 202048, 202048),
        ):
            with self.subTest(bounds=bounds):
                ids = torch.tensor(
                    sorted(
                        {
                            -1,
                            0,
                            4095,
                            *[x + delta for x in bounds for delta in (-1, 0, 1)],
                        }
                    ),
                    dtype=torch.int64,
                )
                buffer = torch.tensor(bounds, dtype=torch.int64)
                actual = self.mask(ids, *buffer.unbind())
                expected = self.mask(ids, *bounds)
                self.assertEqual(actual[0].dtype, torch.int64)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_default_off_and_tp1(self):
        layer = self.layer()
        self.assertIsNone(layer._tpu_runtime_vocab_bounds)
        self.assertEqual(list(layer.named_buffers()), [])
        self.assertNotIn("_tpu_runtime_vocab_bounds", layer.state_dict())
        layer._embed_local_shard(torch.tensor([4, 8, 12], dtype=torch.int64))
        self.assertIsNone(layer._tpu_runtime_vocab_bounds)
        single = self.layer(tp_size=1)
        single.prepare_tpu_runtime_vocab_bounds()
        self.assertIsNone(single._tpu_runtime_vocab_bounds)
        self.assertEqual(list(single.named_buffers()), [])

    def test_prepare_nonpersistent_repeat_and_reload(self):
        layer = self.layer()
        layer.prepare_tpu_runtime_vocab_bounds()
        buffer = layer._tpu_runtime_vocab_bounds
        self.assertEqual(buffer.dtype, torch.int64)
        self.assertEqual(tuple(buffer.shape), (5,))
        self.assertEqual(buffer.device, layer.weight.device)
        self.assertNotIn("_tpu_runtime_vocab_bounds", layer.state_dict())
        layer.prepare_tpu_runtime_vocab_bounds()
        self.assertIs(layer._tpu_runtime_vocab_bounds, buffer)
        layer.shard_indices.org_vocab_start_index = 3
        with self.assertRaisesRegex(RuntimeError, "stale"):
            layer._embed_local_shard(torch.tensor([4], dtype=torch.int64))
        layer.weight = torch.nn.Parameter(torch.ones(16, 3))
        layer.prepare_tpu_runtime_vocab_bounds()
        self.assertIs(layer._tpu_runtime_vocab_bounds, buffer)
        self.assertEqual(buffer.tolist(), [3, 8, 2, 12, 15])

    def test_meta_to_empty_and_residency(self):
        layer = self.layer(device="meta")
        self.assertIsNone(layer._tpu_runtime_vocab_bounds)
        with self.assertRaisesRegex(RuntimeError, "materialized"):
            layer.prepare_tpu_runtime_vocab_bounds()
        layer.to_empty(device="cpu")
        layer.prepare_tpu_runtime_vocab_bounds()
        torch.testing.assert_close(
            layer._tpu_runtime_vocab_bounds, torch.tensor([4, 8, 2, 12, 15])
        )
        layer.to("meta")
        with self.assertRaisesRegex(RuntimeError, "wrong device"):
            layer._embed_local_shard(torch.tensor([4], dtype=torch.int64))
        layer.to_empty(device="cpu")
        layer._tpu_runtime_vocab_bounds.fill_(-100)
        layer.prepare_tpu_runtime_vocab_bounds()
        self.assertEqual(layer._tpu_runtime_vocab_bounds.tolist(), [4, 8, 2, 12, 15])
        layer._tpu_runtime_vocab_bounds = torch.empty(
            5, device="meta", dtype=torch.int64
        )
        layer.prepare_tpu_runtime_vocab_bounds()
        self.assertEqual(layer._tpu_runtime_vocab_bounds.device.type, "cpu")

    def test_post_load_staging(self):
        path = ROOT / "python/sglang/srt/model_loader/post_load.py"
        spec = importlib.util.spec_from_file_location("_bounds_post_load_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)
            layer = self.layer()
            for _ in range(2):
                with module.stage_module_for_post_load(layer, torch.device("cpu")):
                    layer.prepare_tpu_runtime_vocab_bounds()
                self.assertEqual(layer._tpu_runtime_vocab_bounds.device.type, "cpu")
                self.assertNotIn("_tpu_runtime_vocab_bounds", layer.state_dict())
                self.assertEqual(
                    layer._tpu_runtime_vocab_bounds.tolist(), [4, 8, 2, 12, 15]
                )

    def test_missing_buffer_and_lifted_attribute(self):
        layer = self.layer()
        layer.weight.data.copy_(torch.arange(48).reshape(16, 3))
        ids = torch.tensor([-1, 4, 7, 8, 12, 14, 15], dtype=torch.int64)
        expected = layer._embed_local_shard(ids)
        layer._tpu_runtime_vocab_bounds_required = True
        with self.assertRaisesRegex(RuntimeError, "missing"):
            layer._embed_local_shard(ids)
        layer.prepare_tpu_runtime_vocab_bounds()
        layer.output_dtype = torch.float64
        buffer = layer._tpu_runtime_vocab_bounds.clone()
        # Like direct_compile, replace the attribute without changing metadata.
        layer._tpu_runtime_vocab_bounds = buffer
        observed = Mock(wraps=self.mask)
        self.ns["get_masked_input_and_mask"] = observed
        actual = layer._embed_local_shard(ids)
        torch.testing.assert_close(actual, expected.double())
        self.assertTrue(
            all(isinstance(x, torch.Tensor) for x in observed.call_args.args[1:])
        )
        self.assertEqual(actual.dtype, torch.float64)
        self.ns["_is_tpu"] = False
        layer._embed_local_shard(ids)
        self.assertTrue(all(type(x) is int for x in observed.call_args.args[1:]))


class RuntimeBoundsDeclarationTest(unittest.TestCase):
    def test_explicit_state_and_empty_constructor_slot(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "VocabParallelEmbedding"
        )
        declarations = {
            n.target.id: n for n in cls.body if isinstance(n, ast.AnnAssign)
        }
        for name in (
            "_tpu_runtime_vocab_bounds",
            "_tpu_runtime_vocab_bounds_required",
            "_tpu_runtime_vocab_bounds_values",
        ):
            self.assertIn(name, declarations)
        self.assertFalse(declarations["_tpu_runtime_vocab_bounds_required"].value.value)
        self.assertIsNone(declarations["_tpu_runtime_vocab_bounds_values"].value.value)
        constructor = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        registrations = [
            n
            for n in ast.walk(constructor)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "register_buffer"
            and n.args[0].value == "_tpu_runtime_vocab_bounds"
        ]
        self.assertEqual(len(registrations), 1)
        self.assertIsNone(registrations[0].args[1].value)
        self.assertFalse(
            next(
                k.value.value
                for k in registrations[0].keywords
                if k.arg == "persistent"
            )
        )


if __name__ == "__main__":
    unittest.main()
