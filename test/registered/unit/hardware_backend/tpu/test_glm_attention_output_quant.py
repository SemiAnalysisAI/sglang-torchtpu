"""CPU contracts for GLM attention-output quantization dispatch.

FakeTensor supplies device metadata without a TPU allocation. The operation is
mocked only at the wrapper boundary; TPU tests establish its numerical parity.
"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from sglang.srt.hardware_backend.tpu import fp8_linear
from sglang.srt.hardware_backend.tpu import glm_attention_output_quant as backend
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGlmAttentionOutputQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            torch.utils.rename_privateuse1_backend("tpu")
        if not hasattr(torch, "tpu"):
            torch._register_device_module(
                "tpu",
                SimpleNamespace(is_initialized=lambda: False, current_device=lambda: 0),
            )

    def setUp(self):
        super().setUp()
        self.mode = FakeTensorMode()
        self.attention = self.tensor((1, 4, 512))
        projection = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(projection)
        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.use_marlin = False
        method.use_mxfp8 = False
        method.block_quant = True
        projection.quant_method = method
        projection.tp_size = 16
        projection.input_is_parallel = True
        projection.reduce_results = False
        projection.use_decode_attn_tp = False
        projection.use_dp_attention_reduce = False
        projection.weight_block_size = None
        projection.input_scale = None
        projection.bias = None
        projection.weight = self.tensor((1024, 6144), torch.float8_e4m3fn)
        projection.weight_scale = self.tensor((6144, 1), torch.float32)
        self.attn = SimpleNamespace(
            _tpu_glm_attention_output_quant=True,
            _tpu_glm_dense_dsa=True,
            is_nextn=False,
            use_dsa=False,
            current_attention_backend="pallas_tpu_mla",
            use_deep_gemm_bmm=False,
            num_local_heads=4,
            kv_lora_rank=512,
            v_head_dim=256,
            w_vc=self.tensor((4, 256, 512)).transpose(-1, -2),
            o_proj=projection,
            kv_b_proj=SimpleNamespace(set_lora=False),
        )
        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            spec_info=None,
            spec_algorithm=None,
        )
        self.options = dict(
            parallel=SimpleNamespace(
                tp_size=16,
                attn_tp_size=16,
                attn_dp_size=1,
                attn_cp_size=1,
                dcp_enabled=False,
                dcp_replicate_q_proj=False,
            ),
            sp_active=False,
            input_scattered=False,
            lora_enabled=False,
            experimental_lora=False,
            kv_b_lora_active=False,
            attention_output_gate=None,
            llama_4_scaling=None,
            fusion_plan=None,
        )

    def tensor(self, shape, dtype=torch.bfloat16, *, device="tpu:0"):
        return FakeTensor(
            self.mode,
            torch.empty(shape, device="meta", dtype=dtype),
            torch.device(device),
        )

    def eligible(self):
        return backend.can_use_glm_attention_output_quant(
            self.attn, self.attention, self.batch, **self.options
        )

    def test_prepared_channelwise_projection_is_eligible(self):
        self.assertFalse(self.attn.w_vc.is_contiguous())
        self.assertTrue(self.attn.w_vc.transpose(-1, -2).is_contiguous())
        self.assertTrue(self.eligible())
        self.attn.o_proj.weight_scale = self.tensor((6144,), torch.float32)
        self.attn.o_proj.quant_method.block_quant = False
        self.batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.assertTrue(self.eligible())

    def test_only_opted_in_ordinary_glm_decode_is_eligible(self):
        for owner, name, value in (
            (self.attn, "_tpu_glm_attention_output_quant", False),
            (self.attn, "_tpu_glm_dense_dsa", False),
            (self.attn, "is_nextn", True),
            (self.attn, "use_dsa", True),
            (self.attn, "current_attention_backend", "flashinfer"),
            (self.batch.forward_mode, "is_decode", lambda: False),
            (self.batch, "spec_info", object()),
            (self.batch, "spec_algorithm", SimpleNamespace(is_none=lambda: False)),
        ):
            with self.subTest(name=name), patch.object(owner, name, value):
                self.assertFalse(self.eligible())
        with patch.object(self, "batch", None):
            self.assertFalse(self.eligible())

    def test_collective_and_partition_variants_fall_back(self):
        parallel = self.options["parallel"]
        for name, value in (
            ("tp_size", 8),
            ("attn_tp_size", 8),
            ("attn_dp_size", 2),
            ("attn_cp_size", 2),
            ("dcp_enabled", True),
        ):
            with self.subTest(name=name), patch.object(parallel, name, value):
                self.assertFalse(self.eligible())
        for name in ("sp_active", "input_scattered"):
            with self.subTest(name=name), patch.dict(self.options, {name: True}):
                self.assertFalse(self.eligible())
        projection = self.attn.o_proj
        for name, value in (
            ("input_is_parallel", False),
            ("reduce_results", True),
            ("use_decode_attn_tp", True),
            ("use_dp_attention_reduce", True),
        ):
            with self.subTest(name=name), patch.object(projection, name, value):
                self.assertFalse(self.eligible())

    def test_paths_that_modify_values_after_bmm_fall_back(self):
        for name in ("lora_enabled", "experimental_lora", "kv_b_lora_active"):
            with self.subTest(name=name), patch.dict(self.options, {name: True}):
                self.assertFalse(self.eligible())
        for name in ("attention_output_gate", "llama_4_scaling", "fusion_plan"):
            with self.subTest(name=name), patch.dict(self.options, {name: object()}):
                self.assertFalse(self.eligible())
        with patch.object(self.attn.kv_b_proj, "set_lora", True):
            self.assertFalse(self.eligible())
        with patch.object(self.attn, "use_deep_gemm_bmm", True):
            self.assertFalse(self.eligible())

    def test_unqualified_dtype_shape_or_storage_falls_back(self):
        for value in (
            self.tensor((2, 4, 512)),
            self.tensor((1, 4, 512), torch.float32),
            self.tensor((1, 512, 4)).transpose(1, 2),
            torch.empty((1, 4, 512), dtype=torch.bfloat16),
        ):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with patch.object(self, "attention", value):
                    self.assertFalse(self.eligible())
        for value in (
            self.tensor((4, 512, 256)),  # Logical-contiguous is different storage.
            self.tensor((4, 256, 512), torch.float8_e4m3fn).transpose(1, 2),
            self.tensor((4, 128, 512)).transpose(1, 2),
            self.tensor((4, 256, 512), device="tpu:1").transpose(1, 2),
        ):
            with self.subTest(weight_shape=value.shape, weight_dtype=value.dtype):
                with patch.object(self.attn, "w_vc", value):
                    self.assertFalse(self.eligible())

    def test_non_channelwise_projection_or_bias_falls_back(self):
        projection = self.attn.o_proj
        for name, value in (
            ("weight_block_size", [128, 128]),
            ("input_scale", self.tensor((1,), torch.float32)),
            ("bias", self.tensor((6144,))),
            ("weight_scale", self.tensor((1,), torch.float32)),
            ("weight_scale", self.tensor((6144,), torch.bfloat16)),
            ("weight", self.tensor((1024, 6144), torch.bfloat16)),
        ):
            with self.subTest(name=name), patch.object(projection, name, value):
                self.assertFalse(self.eligible())
        for name in ("use_marlin", "use_mxfp8"):
            with self.subTest(name=name):
                with patch.object(projection.quant_method, name, True):
                    self.assertFalse(self.eligible())

    def test_wrapper_preserves_storage_and_returns_call_local_carrier(self):
        physical = torch.randn((4, 256, 512)).bfloat16()
        attn = SimpleNamespace(w_vc=physical.transpose(1, 2))
        attention = torch.randn((1, 4, 512)).bfloat16()
        outputs = [
            (
                torch.zeros((16, 1024), dtype=torch.float8_e4m3fn),
                torch.arange(16, dtype=torch.float32),
            )
            for _ in range(2)
        ]
        original_attributes = vars(attn).copy()
        with patch.object(backend, "_WV_QUANT_OP", side_effect=outputs) as operation:
            carriers = [
                backend.prepare_glm_attention_output_quant(attn, attention)
                for _ in outputs
            ]
        for call, carrier, (quantized, scales) in zip(
            operation.call_args_list, carriers, outputs
        ):
            self.assertIs(call.args[0], attention)
            supplied_weight = call.args[1]
            self.assertEqual(supplied_weight.shape, (4, 256, 512))
            self.assertTrue(supplied_weight.is_contiguous())
            self.assertEqual(supplied_weight.data_ptr(), physical.data_ptr())
            self.assertIsInstance(carrier, fp8_linear.TpuFp8PrequantizedInput)
            self.assertIs(carrier.q, quantized)
            self.assertEqual(carrier.scale.shape, (16, 1))
            self.assertEqual(carrier.scale.data_ptr(), scales.data_ptr())
            self.assertTrue(torch.equal(carrier.scale[:, 0], scales))
            self.assertEqual(carrier.logical_shape, (1, 1024))
            self.assertEqual(carrier.output_dtype, torch.bfloat16)
        self.assertIsNot(carriers[0].q, carriers[1].q)
        self.assertEqual(vars(attn).keys(), original_attributes.keys())
        self.assertIs(attn.w_vc, original_attributes["w_vc"])

    def test_registration_is_eager_once_and_required_by_wrapper(self):
        kernel_name = "sglang.srt.hardware_backend.tpu.glm_attention_output_quant_kernel"
        module = ModuleType(kernel_name)
        operation = Mock()
        module.register_wv_quant_op = Mock(return_value=operation)
        with patch.object(backend, "_WV_QUANT_OP", None):
            with self.assertRaisesRegex(RuntimeError, "register before tracing"):
                backend.prepare_glm_attention_output_quant(SimpleNamespace(), None)
            with patch.dict(sys.modules, {kernel_name: module}):
                backend.register_glm_attention_output_quant_op()
                backend.register_glm_attention_output_quant_op()
            module.register_wv_quant_op.assert_called_once_with()
            self.assertIs(backend._WV_QUANT_OP, operation)

    def test_carrier_reuses_native_dense_consumer_without_requantization(self):
        generator = torch.Generator().manual_seed(128)
        expanded = torch.randn((1, 1024), generator=generator).bfloat16()
        padded, _ = fp8_linear._pad_rows_to_scaled_mm_alignment(expanded)
        q, scales = fp8_linear._quantize_activation_per_row(
            padded, torch.float8_e4m3fn
        )
        carrier = fp8_linear.TpuFp8PrequantizedInput(
            q, scales, (1, 1024), torch.bfloat16
        )
        weight = torch.zeros((1024, 6144), dtype=torch.float8_e4m3fn)
        weight_scale = torch.linspace(0.002, 0.018, 6144).reshape(-1, 1)
        raw_dot = torch.randn((16, 6144), generator=generator)
        with patch.object(torch, "_scaled_mm", return_value=raw_dot):
            expected = fp8_linear.apply_tpu_fp8_linear(expanded, weight, weight_scale)
        with (
            patch.object(torch, "_scaled_mm", return_value=raw_dot) as dot,
            patch.object(
                fp8_linear,
                "_quantize_activation_per_row",
                side_effect=AssertionError("Carrier must not be requantized"),
            ),
        ):
            actual = fp8_linear.apply_tpu_fp8_linear(carrier, weight, weight_scale)
        self.assertIs(dot.call_args.args[0], q)
        self.assertIs(dot.call_args.args[1], weight)
        self.assertEqual(dot.call_args.kwargs["out_dtype"], torch.float32)
        self.assertEqual(actual.shape, (1, 6144))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
        )


if __name__ == "__main__":
    unittest.main()
