"""CPU-only metadata and early-dispatch coverage; numerical tests require TPU."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
DISPATCH = ROOT / "python/sglang/srt/hardware_backend/tpu/glm_fp8_mla_front_dispatch.py"
FORWARD = (
    ROOT
    / "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py"
)
DEVICE = SimpleNamespace(type="tpu")
TORCH = SimpleNamespace(
    bfloat16="bf16", float32="fp32", float8_e4m3fn="fp8", int32="i32", int64="i64"
)


class TensorMetadata:
    def __init__(
        self, shape, dtype="bf16", *, contiguous=True, transpose_contiguous=False
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = DEVICE
        self.contiguous = contiguous
        self.transpose_contiguous = transpose_contiguous

    def is_contiguous(self):
        return self.contiguous

    def transpose(self, first, second):
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        return TensorMetadata(
            tuple(shape), self.dtype, contiguous=self.transpose_contiguous
        )


RMSNorm = type("RMSNorm", (SimpleNamespace,), {})
RotaryEmbedding = type("RotaryEmbedding", (SimpleNamespace,), {})
ReplicatedLinear = type("ReplicatedLinear", (SimpleNamespace,), {})
ColumnParallelLinear = type("ColumnParallelLinear", (SimpleNamespace,), {})
Fp8LinearMethod = type("Fp8LinearMethod", (SimpleNamespace,), {})


class DropImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None

    def visit_Import(self, node):
        return None


def load_functions(nodes, namespace):
    # Execute the production functions unchanged, replacing only heavy imports
    # with metadata stand-ins. No tensor payload or TPU arithmetic is simulated.
    nodes = [DropImports().visit(copy.deepcopy(node)) for node in nodes]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(DISPATCH), "exec"), namespace)


def make_namespace():
    namespace = dict(
        torch=TORCH,
        RMSNorm=RMSNorm,
        RotaryEmbedding=RotaryEmbedding,
        ReplicatedLinear=ReplicatedLinear,
        ColumnParallelLinear=ColumnParallelLinear,
        Fp8LinearMethod=Fp8LinearMethod,
        envs=SimpleNamespace(
            SGLANG_TPU_GLM_FP8_MLA_KC_LOGICAL=SimpleNamespace(get=lambda: False)
        ),
    )
    load_functions(
        [
            n
            for n in ast.parse(DISPATCH.read_text()).body
            if isinstance(n, ast.FunctionDef)
        ],
        namespace,
    )
    return namespace


def make_inputs():
    def linear(cls, k, n):
        return cls(
            weight=TensorMetadata((k, n), "fp8"),
            weight_scale=TensorMetadata((n, 1), "fp32"),
            quant_method=Fp8LinearMethod(use_marlin=False, use_mxfp8=False),
            weight_block_size=None,
            input_scale=None,
            bias=None,
            gather_output=False,
        )

    def norm(width):
        return RMSNorm(
            hidden_size=width,
            variance_size_override=None,
            override_orig_dtype=None,
            cast_x_before_out_mul=False,
            fp32_residual=False,
            variance_epsilon=1e-5,
            weight=TensorMetadata((width,)),
        )

    attn = SimpleNamespace(
        _tpu_glm_dense_dsa=True,
        use_dsa=False,
        is_nextn=False,
        use_deep_gemm_bmm=False,
        current_attention_backend="pallas_tpu_mla",
        q_lora_rank=2048,
        kv_lora_rank=512,
        num_local_heads=4,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        use_deepseek_yarn_rope=False,
        fused_qkv_a_proj_with_mqa=linear(ReplicatedLinear, 6144, 2624),
        q_b_proj=linear(ColumnParallelLinear, 2048, 1024),
        q_a_layernorm=norm(2048),
        kv_a_layernorm=norm(512),
        rotary_emb=RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            is_neox_style=False,
            position_start=0,
            cos_sin_cache=TensorMetadata((1048576, 64)),
        ),
        w_kc=TensorMetadata((4, 192, 512), contiguous=False, transpose_contiguous=True),
    )
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
    options = dict(
        parallel=SimpleNamespace(
            dcp_enabled=False,
            dcp_replicate_q_proj=False,
            attn_cp_size=1,
            attn_dp_size=1,
        ),
        input_scattered=False,
        lora_enabled=False,
        experimental_lora=False,
        llama_4_scaling=None,
    )
    return attn, TensorMetadata((1, 6144)), TensorMetadata((1,), "i64"), batch, options


def set_path(root, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        root = getattr(root, part)
    setattr(root, parts[-1], value)


class TestGlmFp8MlaFrontDispatch(unittest.TestCase):
    def setUp(self):
        self.namespace = make_namespace()

    def eligible(self, inputs):
        attn, hidden, positions, batch, options = inputs
        return self.namespace["can_use_glm_fp8_mla_front"](
            attn, hidden, positions, batch, **options
        )

    def test_actual_metadata_contract_and_int32_positions(self):
        inputs = make_inputs()
        self.assertTrue(self.eligible(inputs))
        inputs[2].dtype = "i32"
        inputs[0].fused_qkv_a_proj_with_mqa.weight_scale.shape = (2624,)
        self.assertTrue(self.eligible(inputs))

    def test_real_torch_classes_and_tensor_metadata(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        from sglang.srt.hardware_backend.tpu.glm_fp8_mla_front_dispatch import (
            can_use_glm_fp8_mla_front,
        )
        from sglang.srt.layers.layernorm import RMSNorm as RealRMSNorm
        from sglang.srt.layers.linear import (
            ColumnParallelLinear as RealColumnParallelLinear,
        )
        from sglang.srt.layers.linear import ReplicatedLinear as RealReplicatedLinear
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod as RealFp8Method
        from sglang.srt.layers.rotary_embedding.base import (
            RotaryEmbedding as RealRotaryEmbedding,
        )

        classes = {
            RMSNorm: RealRMSNorm,
            RotaryEmbedding: RealRotaryEmbedding,
            ReplicatedLinear: RealReplicatedLinear,
            ColumnParallelLinear: RealColumnParallelLinear,
            Fp8LinearMethod: RealFp8Method,
        }
        dtypes = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp8": torch.float8_e4m3fn,
            "i32": torch.int32,
            "i64": torch.int64,
        }
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            torch.utils.rename_privateuse1_backend("tpu")
        mode = FakeTensorMode()
        device = torch.device("tpu")

        def materialize(value):
            if isinstance(value, TensorMetadata):
                meta = torch.empty(
                    value.shape, dtype=dtypes[value.dtype], device="meta"
                )
                if value.transpose_contiguous:
                    meta = torch.empty(
                        (4, 512, 192), dtype=dtypes[value.dtype], device="meta"
                    ).transpose(-1, -2)
                return FakeTensor(mode, meta, device)
            if type(value) in classes:
                cls = classes[type(value)]
                result = cls.__new__(cls)
                if isinstance(result, torch.nn.Module):
                    torch.nn.Module.__init__(result)
                for key, item in vars(value).items():
                    setattr(result, key, materialize(item))
                return result
            if isinstance(value, SimpleNamespace):
                return SimpleNamespace(
                    **{key: materialize(item) for key, item in vars(value).items()}
                )
            return value

        # FakeTensor uses real dtype/shape/stride logic, with metadata tensors
        # only. The stub prevents querying or initializing a TPU runtime.
        device_stub = SimpleNamespace(
            is_initialized=lambda: False, current_device=lambda: 0
        )
        with patch.object(torch, "tpu", device_stub, create=True):
            attn, hidden, positions, batch, options = make_inputs()
            attn, hidden, positions = map(materialize, (attn, hidden, positions))
            self.assertTrue(
                can_use_glm_fp8_mla_front(attn, hidden, positions, batch, **options)
            )
            self.assertFalse(attn.w_kc.is_contiguous())
            self.assertTrue(attn.w_kc.transpose(-1, -2).is_contiguous())
            attn.q_b_proj.gather_output = True
            self.assertFalse(
                can_use_glm_fp8_mla_front(attn, hidden, positions, batch, **options)
            )

    def test_unsupported_model_modes_and_metadata_fall_back(self):
        cases = {
            "_tpu_glm_dense_dsa": False,
            "use_dsa": True,
            "is_nextn": True,
            "use_deep_gemm_bmm": True,
            "current_attention_backend": "flashmla",
            "q_lora_rank": 1024,
            "num_local_heads": 8,
            "kv_lora_rank": 256,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 32,
            "use_deepseek_yarn_rope": True,
            "prepare_attention_output_gate": object(),
            "learnable_sink_param": object(),
            "_kimi_split_gguf_kv_b": True,
            "_use_min_latency_fused_a_gemm": True,
            "_use_min_latency_q_b_gemm": True,
            "q_a_proj": object(),
            "fused_qkv_a_proj_with_mqa.bias": object(),
            "fused_qkv_a_proj_with_mqa.input_scale": object(),
            "fused_qkv_a_proj_with_mqa.weight_block_size": [128, 128],
            "fused_qkv_a_proj_with_mqa.weight.dtype": "bf16",
            "fused_qkv_a_proj_with_mqa.weight.contiguous": False,
            "fused_qkv_a_proj_with_mqa.weight_scale.shape": (21, 48),
            "fused_qkv_a_proj_with_mqa.weight_scale.dtype": "bf16",
            "fused_qkv_a_proj_with_mqa.quant_method.use_mxfp8": True,
            "q_b_proj.quant_method.use_marlin": True,
            "q_b_proj.set_lora": True,
            "q_b_proj.gather_output": True,
            "q_b_proj.weight.shape": (2048, 2048),
            "q_a_layernorm.cast_x_before_out_mul": True,
            "q_a_layernorm.variance_size_override": 128,
            "q_a_layernorm.override_orig_dtype": "fp32",
            "q_a_layernorm.fp32_residual": True,
            "kv_a_layernorm.variance_epsilon": 1e-6,
            "kv_a_layernorm.weight.dtype": "fp32",
            "rotary_emb.is_neox_style": True,
            "rotary_emb.position_start": 8,
            "rotary_emb.sin_cos_cache": object(),
            "rotary_emb.rotary_dim": 32,
            "rotary_emb.cos_sin_cache.dtype": "fp32",
            "rotary_emb.cos_sin_cache.shape": (2047, 64),
            "rotary_emb.cos_sin_cache.contiguous": False,
            "w_kc.dtype": "fp8",
            "w_kc.transpose_contiguous": False,
        }
        for path, value in cases.items():
            with self.subTest(path=path):
                inputs = make_inputs()
                set_path(inputs[0], path, value)
                self.assertFalse(self.eligible(inputs))
        for index, attribute, value in (
            (1, "shape", (8, 6144)),
            (1, "dtype", "fp32"),
            (1, "device", SimpleNamespace(type="cpu")),
            (1, "contiguous", False),
            (2, "shape", (2,)),
            (2, "dtype", "fp32"),
            (2, "contiguous", False),
        ):
            with self.subTest(index=index, attribute=attribute):
                inputs = make_inputs()
                setattr(inputs[index], attribute, value)
                self.assertFalse(self.eligible(inputs))

    def test_non_decode_parallel_and_lora_fall_back(self):
        inputs = make_inputs()
        inputs[3].forward_mode.is_decode = lambda: False
        self.assertFalse(self.eligible(inputs))
        for key in (
            "input_scattered",
            "lora_enabled",
            "experimental_lora",
            "llama_4_scaling",
        ):
            inputs = make_inputs()
            inputs[4][key] = True
            self.assertFalse(self.eligible(inputs))
        for key, value in (
            ("dcp_enabled", True),
            ("dcp_replicate_q_proj", True),
            ("attn_cp_size", 2),
            ("attn_dp_size", 2),
        ):
            inputs = make_inputs()
            setattr(inputs[4]["parallel"], key, value)
            self.assertFalse(self.eligible(inputs))

    def test_module_subclasses_fall_back(self):
        for field in (
            "q_a_layernorm",
            "rotary_emb",
            "fused_qkv_a_proj_with_mqa",
            "q_b_proj",
        ):
            inputs = make_inputs()
            value = getattr(inputs[0], field)
            value.__class__ = type("UntestedSubclass", (type(value),), {})
            self.assertFalse(self.eligible(inputs))

    def test_early_dispatch_preserves_kv_owner_and_prepare_tuple(self):
        attn, hidden, positions, batch, options = make_inputs()
        ns = self.namespace
        cls = next(
            n
            for n in ast.parse(FORWARD.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekMLAForwardMixin"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_absorb_prepare"
        )
        outputs = tuple(object() for _ in range(4))
        prepared_kv = (object(), object())
        kernel_calls, kv_calls = [], []

        def kernel(*args):
            kernel_calls.append(args)
            return outputs

        def prepare(*args, **kwargs):
            kv_calls.append((args, kwargs))
            return prepared_kv

        ns.update(
            _is_tpu=True,
            envs=SimpleNamespace(
                SGLANG_TPU_GLM_FP8_MLA_FRONT=SimpleNamespace(get=lambda: True)
            ),
            get_parallel=lambda: options["parallel"],
            get_attn_tp_context=lambda: SimpleNamespace(input_scattered=False),
            get_lora=lambda: SimpleNamespace(enable_lora=False),
            _SGLANG_EXPERIMENTAL_LORA_OPTI=False,
            run_glm_fp8_mla_front=kernel,
            prepare_kv_for_attention=prepare,
        )
        load_functions([method], ns)
        allocator = object()
        actual = ns["forward_absorb_prepare"](attn, positions, hidden, batch, allocator)
        self.assertEqual(kernel_calls, [(attn, hidden, positions)])
        self.assertEqual(
            kv_calls,
            [((attn, batch, outputs[2], outputs[3]), {"defer_materialization": False})],
        )
        self.assertEqual(
            actual,
            (
                outputs[1],
                prepared_kv[1],
                outputs[0],
                prepared_kv[0],
                batch,
                allocator,
                positions,
                None,
                None,
                None,
            ),
        )
        self.assertFalse(attn._q8kv8_qprep_overlap_pending)

        class NativePathReached(Exception):
            pass

        def native_path(*args):
            raise NativePathReached

        attn._can_fuse_bmm_into_attention = native_path
        for tpu, enabled in ((False, True), (True, False)):
            with self.subTest(tpu=tpu, enabled=enabled):
                ns["_is_tpu"] = tpu
                ns["envs"].SGLANG_TPU_GLM_FP8_MLA_FRONT.get = lambda: enabled
                with self.assertRaises(NativePathReached):
                    ns["forward_absorb_prepare"](
                        attn, positions, hidden, batch, allocator
                    )
        ns["_is_tpu"] = True
        ns["envs"].SGLANG_TPU_GLM_FP8_MLA_FRONT.get = lambda: True
        attn.use_dsa = True
        with self.assertRaises(NativePathReached):
            ns["forward_absorb_prepare"](attn, positions, hidden, batch, allocator)
        self.assertEqual(len(kernel_calls), 1)

    def test_constructor_registration_requires_opt_in_bounded_glm_tpu(self):
        path = ROOT / "python/sglang/srt/models/deepseek_v2.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2AttentionMLA"
        )
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        index = next(
            i
            for i, n in enumerate(init.body)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_tpu_glm_dense_dsa"
                for t in n.targets
            )
        )
        statements = [
            DropImports().visit(copy.deepcopy(n)) for n in init.body[index : index + 3]
        ]
        module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
        for tpu, enabled, model_type, dense_limit, dsa, nextn, expected in (
            (True, True, "glm_moe_dsa", 2048, False, False, True),
            (True, False, "glm_moe_dsa", 2048, False, False, False),
            (False, True, "glm_moe_dsa", 2048, False, False, False),
            (True, True, "deepseek_v3", 2048, False, False, False),
            (True, True, "glm_moe_dsa", None, False, False, False),
            (True, True, "glm_moe_dsa", 2048, True, False, False),
            (True, True, "glm_moe_dsa", 2048, False, True, False),
        ):
            with self.subTest(
                tpu=tpu, enabled=enabled, model=model_type, dsa=dsa, nextn=nextn
            ):
                calls = []
                ns = dict(
                    self=SimpleNamespace(use_dsa=dsa),
                    _is_tpu=tpu,
                    is_nextn=nextn,
                    config=SimpleNamespace(
                        model_type=model_type, tpu_dense_dsa_max_context=dense_limit
                    ),
                    envs=SimpleNamespace(
                        SGLANG_TPU_GLM_FP8_MLA_FRONT=SimpleNamespace(
                            get=lambda: enabled
                        )
                    ),
                    register_glm_fp8_mla_front_op=lambda: calls.append(True),
                )
                exec(compile(module, str(path), "exec"), ns)
                self.assertEqual(bool(calls), expected)
                self.assertEqual(ns["self"]._tpu_glm_dense_dsa_max_context, dense_limit)

    def test_kernel_argument_aliases_and_kc_layout(self):
        attn, hidden, positions, _, _ = make_inputs()
        for enabled in (False, True):
            with self.subTest(kc_logical=enabled):
                calls = []
                self.namespace["envs"].SGLANG_TPU_GLM_FP8_MLA_KC_LOGICAL.get = (
                    lambda: enabled
                )
                self.namespace["glm_fp8_mla_front"] = (
                    lambda *args, **kwargs: calls.append((args, kwargs))
                )
                self.namespace["run_glm_fp8_mla_front"](attn, hidden, positions)
                arguments, keywords = calls[0]
                self.assertIs(arguments[0], hidden)
                self.assertIs(arguments[1], attn.fused_qkv_a_proj_with_mqa.weight)
                self.assertIs(arguments[5], attn.q_b_proj.weight)
                if enabled:
                    self.assertIs(arguments[7], attn.w_kc)
                    self.assertEqual(arguments[7].shape, (4, 192, 512))
                else:
                    self.assertEqual(arguments[7].shape, (4, 512, 192))
                    self.assertTrue(arguments[7].is_contiguous())
                self.assertIs(arguments[8], positions)
                self.assertIs(arguments[9], attn.rotary_emb.cos_sin_cache)
                self.assertEqual(arguments[10:], (1e-5, 1e-5))
                self.assertEqual(keywords, {"kc_logical": enabled})

    def test_bounded_cache_uses_existing_values_and_keeps_dma_alignment(self):
        import torch

        attn, hidden, positions, _, _ = make_inputs()
        original = torch.arange(4096 * 64, dtype=torch.float32).reshape(4096, 64)
        original = original.to(torch.bfloat16)
        attn.rotary_emb.cos_sin_cache = original
        for bound, rows in ((2048, 2048), (2039, 2040), (1, 8), (None, 4096)):
            with self.subTest(bound=bound):
                attn._tpu_glm_dense_dsa_max_context = bound
                calls = []
                self.namespace["glm_fp8_mla_front"] = (
                    lambda *args, **kwargs: calls.append(args)
                )
                self.namespace["run_glm_fp8_mla_front"](attn, hidden, positions)
                cache = calls[0][9]
                self.assertEqual(tuple(cache.shape), (rows, 64))
                self.assertTrue(torch.equal(cache, original[:rows]))
                self.assertEqual(cache.data_ptr(), original.data_ptr())
                self.assertIs(attn.rotary_emb.cos_sin_cache, original)


if __name__ == "__main__":
    unittest.main()
