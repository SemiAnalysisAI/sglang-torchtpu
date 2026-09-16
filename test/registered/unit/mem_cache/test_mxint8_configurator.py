"""CPU tests for the optional MXINT8 backend integration boundary."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_validation():
    # Load the actual method without initializing accelerator-only registries.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/kv_cache_configurator.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheConfigurator"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_mxint8"
    )
    scope = {
        "torch": torch,
        "resolving_view": lambda args: args,
        "get_parallel": lambda: SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
        "is_minimax_sparse": lambda _: False,
        "is_deepseek_v4": lambda _: False,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


class TestMXINT8Configurator(unittest.TestCase):
    def setUp(self):
        self.validate = _load_validation()
        self.backend = ModuleType(
            "sglang_torchtpu_meta.srt.backends.attention.kv_mxint8"
        )
        self.backend.validate_mxint8_config = Mock()
        self.backend.validate_mxint8_features = Mock()
        self.kvc = SimpleNamespace(
            kv_cache_dtype_str="mxint8",
            kv_cache_dtype=torch.int8,
            kv_group_size=16,
            model_dtype=torch.bfloat16,
            device="tpu",
            pool_page_size=128,
            is_draft_worker=True,
            speculative_draft_attention_backend="runner_backend",
            server_args=SimpleNamespace(
                speculative_draft_attention_backend="global_backend",
                speculative_draft_kv_cache_dtype="mxint8",
                prefill_attention_backend="target_prefill",
                decode_attention_backend="target_decode",
                attention_backend=None,
                tpu_mxint8_group_size=64,
            ),
            model_config=SimpleNamespace(
                get_num_kv_heads=lambda *_: 3,
                get_swa_num_kv_heads=lambda *_: 5,
                head_dim=64,
                v_head_dim=64,
                swa_head_dim=128,
                swa_v_head_dim=128,
                hf_config=SimpleNamespace(),
            ),
            spec_algorithm=SimpleNamespace(
                is_eagle=lambda: True, is_standalone=lambda: False
            ),
            is_hybrid_swa=True,
            use_mla_backend=False,
            post_capture_kv_active=False,
            is_hybrid_swa_compress=False,
        )

    def test_draft_validation_uses_runner_backend_for_both_geometries(self):
        with patch.dict(sys.modules, {self.backend.__name__: self.backend}):
            self.validate(self.kvc)
        calls = self.backend.validate_mxint8_config.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.kwargs["head_num"] for call in calls], [3, 5])
        for call in calls:
            self.assertEqual(call.kwargs["prefill_backend"], "runner_backend")
            self.assertEqual(call.kwargs["decode_backend"], "runner_backend")
            self.assertEqual(call.kwargs["group_size"], 16)

    def test_target_uses_configured_draft_backend_and_group_for_shared_draft(self):
        self.kvc.is_draft_worker = False
        self.kvc.speculative_draft_attention_backend = None
        self.kvc.kv_cache_dtype_str = "bf16"
        self.kvc.kv_cache_dtype = torch.bfloat16
        self.kvc.kv_group_size = None
        with patch.dict(sys.modules, {self.backend.__name__: self.backend}):
            self.validate(self.kvc)
        kwargs = self.backend.validate_mxint8_config.call_args.kwargs
        self.assertEqual(kwargs["prefill_backend"], "global_backend")
        self.assertEqual(kwargs["decode_backend"], "global_backend")
        self.assertEqual(kwargs["group_size"], 64)
        self.assertEqual(kwargs["kv_cache_dtype"], torch.int8)

    def test_ordinary_formats_do_not_import_private_backend(self):
        self.kvc.server_args.speculative_draft_kv_cache_dtype = None
        with patch.dict(sys.modules, {self.backend.__name__: None}):
            for name, dtype in (
                ("auto", torch.bfloat16),
                ("bf16", torch.bfloat16),
                ("fp8_e4m3", torch.float8_e4m3fn),
                ("fp8_e5m2", torch.float8_e5m2),
            ):
                with self.subTest(format=name):
                    self.kvc.kv_cache_dtype_str = name
                    self.kvc.kv_cache_dtype = dtype
                    self.assertIsNone(self.validate(self.kvc))

    def test_bare_int8_fails_before_importing_backend(self):
        self.kvc.kv_cache_dtype_str = "auto"
        with patch.dict(sys.modules, {self.backend.__name__: None}):
            with self.assertRaisesRegex(ValueError, "explicit mxint8"):
                self.validate(self.kvc)


if __name__ == "__main__":
    unittest.main()
