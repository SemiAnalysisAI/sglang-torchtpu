"""Host contracts for the opt-in GLM norm/shared-input quantization path.

FakeTensor metadata exercises real module types without initializing TPU.
Arithmetic tests keep native collectives mocked and the RMS prefix real;
separate TPU qualification establishes the Pallas epilogue's equivalence.
"""

from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from sglang.srt.hardware_backend.tpu import glm_norm_shared_quant as backend
from sglang.srt.layers import communicator as comm
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.models import deepseek_v2 as model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def module_without_constructor(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def decode_batch():
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: True, is_decode_or_idle=lambda: True
        )
    )


class TestGlmNormSharedQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Metadata only: no torch_tpu import, PJRT client or physical device.
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            torch.utils.rename_privateuse1_backend("tpu")
        if not hasattr(torch, "tpu"):
            torch._register_device_module(
                "tpu",
                SimpleNamespace(is_initialized=lambda: False, current_device=lambda: 0),
            )

    def make_fixture(self):
        fake_mode = FakeTensorMode()

        def tensor(shape, dtype):
            return FakeTensor(
                fake_mode, torch.empty(shape, dtype=dtype, device="meta"),
                torch.device("tpu"),
            )

        norm = module_without_constructor(RMSNorm)
        norm.hidden_size = 6144
        norm.variance_size_override = None
        norm.override_orig_dtype = None
        norm.cast_x_before_out_mul = False
        norm.fp32_residual = False
        norm.variance_epsilon = 1e-5
        norm.weight = tensor((6144,), torch.bfloat16)
        norm._tpu_weight_fp32 = tensor((6144,), torch.float32)

        shared = module_without_constructor(model.DeepseekV2MLP)
        shared.swiglu_limit = None
        for name, cls, shape in (
            ("gate_up_proj", MergedColumnParallelLinear, (6144, 256)),
            ("down_proj", RowParallelLinear, (128, 6144)),
        ):
            linear = module_without_constructor(cls)
            linear.tp_size = 16
            linear.gather_output = False
            linear.input_is_parallel = True
            linear.reduce_results = False
            linear.use_decode_attn_tp = False
            linear.use_dp_attention_reduce = False
            linear.weight = tensor(shape, torch.float8_e4m3fn)
            linear.weight_scale = tensor((shape[1], 1), torch.float32)
            linear.weight_block_size = None
            linear.bias = None
            linear.input_scale = None
            method = object.__new__(Fp8LinearMethod)
            method.use_marlin = False
            method.use_mxfp8 = False
            method.block_quant = True
            linear.quant_method = method
            setattr(shared, name, linear)

        moe = module_without_constructor(model.DeepseekV2MoE)
        moe.is_nextn = False
        moe._enable_a2a_moe = False
        moe._fuse_shared_experts_inside_sbo = False
        moe._shared_expert_tp1 = False
        moe.num_fused_shared_experts = 0
        moe.alt_stream = None
        moe._moe_quant_once = False
        moe.shared_experts = shared

        communicator = object.__new__(comm.LayerCommunicator)
        communicator._sp_variant = None
        communicator._context = SimpleNamespace(
            tp_size=16, attn_tp_size=16, attn_dp_size=1, attn_cp_size=1,
            cache=None,
        )
        communicator.post_attention_layernorm = norm
        communicator._communicate_with_all_reduce_and_layer_norm_fn = partial(
            comm.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
            residual_input_mode=comm.ScatterMode.TP_ATTN_FULL,
        )
        decoder = module_without_constructor(model.DeepseekV2DecoderLayer)
        decoder._tpu_glm_norm_shared_quant = True
        decoder.is_nextn = False
        decoder.is_layer_sparse = True
        decoder.speculative_algorithm = SimpleNamespace(is_none=lambda: True)
        decoder.post_attention_layernorm = norm
        decoder.layer_communicator = communicator
        decoder.mlp = moe
        options = dict(
            parallel=SimpleNamespace(
                tp_size=16, attn_tp_size=16, attn_dp_size=1, attn_cp_size=1,
                moe_ep_size=16, dcp_enabled=False, dcp_replicate_q_proj=False,
            ),
            sp_active=False, input_scattered=False, lora_enabled=False,
            moe_a2a_none=True,
        )
        return (
            decoder, tensor((1, 6144), torch.bfloat16),
            tensor((1, 6144), torch.bfloat16), decode_batch(), options,
        )

    def test_real_module_metadata_eligible(self):
        decoder, hidden, residual, batch, options = self.make_fixture()
        self.assertTrue(
            backend.can_use_glm_norm_shared_quant(
                decoder, hidden, residual, batch, **options
            )
        )
        # Both prepared channel-scale views are supported without conversion.
        for linear in (
            decoder.mlp.shared_experts.gate_up_proj,
            decoder.mlp.shared_experts.down_proj,
        ):
            linear.weight_scale = linear.weight_scale.reshape(-1)
        self.assertTrue(
            backend.can_use_glm_norm_shared_quant(
                decoder, hidden, residual, batch, **options
            )
        )

    def test_metadata_guards_fall_back(self):
        mutations = (
            ("_tpu_glm_norm_shared_quant", False),
            ("is_nextn", True),
            ("is_layer_sparse", False),
            ("speculative_algorithm", SimpleNamespace(is_none=lambda: False)),
            ("post_attention_layernorm.variance_epsilon", 1e-6),
            ("post_attention_layernorm.cast_x_before_out_mul", True),
            ("post_attention_layernorm.fp32_residual", True),
            ("post_attention_layernorm._tpu_weight_fp32", None),
            ("layer_communicator._context.attn_dp_size", 2),
            ("layer_communicator._context.cache", object()),
            ("layer_communicator._communicate_with_all_reduce_and_layer_norm_fn", Mock()),
            ("mlp._enable_a2a_moe", True),
            ("mlp._fuse_shared_experts_inside_sbo", True),
            ("mlp._shared_expert_tp1", True),
            ("mlp.num_fused_shared_experts", 1),
            ("mlp.alt_stream", object()),
            ("mlp._moe_quant_once", True),
            ("mlp.shared_experts.swiglu_limit", 7.0),
            ("mlp.shared_experts.gate_up_proj.gather_output", True),
            ("mlp.shared_experts.down_proj.reduce_results", True),
            ("mlp.shared_experts.down_proj.use_decode_attn_tp", True),
            ("mlp.shared_experts.gate_up_proj.quant_method.use_mxfp8", True),
            ("mlp.shared_experts.gate_up_proj.weight_block_size", [128, 128]),
            ("mlp.shared_experts.gate_up_proj.input_scale", torch.ones(1)),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                decoder, hidden, residual, batch, options = self.make_fixture()
                target = decoder
                *attributes, name = path.split(".")
                for attribute in attributes:
                    target = getattr(target, attribute)
                setattr(target, name, value)
                self.assertFalse(backend.can_use_glm_norm_shared_quant(
                    decoder, hidden, residual, batch, **options
                ))
        for name, value in (
            ("sp_active", True), ("input_scattered", True),
            ("lora_enabled", True), ("moe_a2a_none", False),
        ):
            with self.subTest(option=name):
                decoder, hidden, residual, batch, options = self.make_fixture()
                options[name] = value
                self.assertFalse(backend.can_use_glm_norm_shared_quant(
                    decoder, hidden, residual, batch, **options
                ))
        decoder, hidden, residual, batch, options = self.make_fixture()
        for invalid in (None, hidden.expand(2, -1), hidden.reshape(1, 1, -1), hidden.float()):
            self.assertFalse(backend.can_use_glm_norm_shared_quant(
                decoder, invalid, residual, batch, **options
            ))
        batch.forward_mode.is_decode = lambda: False
        self.assertFalse(backend.can_use_glm_norm_shared_quant(
            decoder, hidden, residual, batch, **options
        ))

    def test_native_prefix_uses_unrounded_residual_and_flat_scale_view(self):
        norm = module_without_constructor(RMSNorm)
        norm.hidden_size = 6144
        norm.override_orig_dtype = None
        norm.variance_size_override = None
        norm.cast_x_before_out_mul = False
        norm.fp32_residual = False
        norm.variance_epsilon = 1e-5
        norm.weight = torch.linspace(0.5, 1.5, 6144).bfloat16()
        norm._tpu_weight_fp32 = norm.weight.float()
        hidden = torch.ones((1, 6144), dtype=torch.bfloat16)
        residual = torch.full_like(hidden, 2**-8)
        z = hidden.float() + residual.float()
        expected_inverse = torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
        normalized, native_residual = norm.forward_native(hidden, residual)
        q = torch.zeros((16, 6144), dtype=torch.float8_e4m3fn)
        scales = torch.arange(16, dtype=torch.float32)
        op = Mock(return_value=(normalized, q, scales))
        with patch.object(backend, "_EPILOGUE_OP", op):
            (actual, carrier), residual_out = backend._norm_with_shared_quant(
                norm, hidden, residual
            )
        passed_z, passed_inverse, passed_weight = op.call_args.args
        self.assertTrue(torch.equal(passed_z, z))
        self.assertFalse(torch.equal(passed_z, residual_out.float()))
        self.assertTrue(torch.equal(passed_inverse, expected_inverse))
        self.assertIs(passed_weight, norm._tpu_weight_fp32)
        self.assertIs(actual, normalized)
        self.assertTrue(torch.equal(residual_out, native_residual))
        self.assertIs(carrier.q, q)
        self.assertEqual(carrier.logical_shape, (1, 6144))
        self.assertEqual(carrier.output_dtype, torch.bfloat16)
        self.assertEqual(carrier.scale.shape, (16, 1))
        self.assertEqual(carrier.scale.data_ptr(), scales.data_ptr())
        with patch.object(backend, "_EPILOGUE_OP", None):
            with self.assertRaisesRegex(RuntimeError, "before tracing"):
                backend._norm_with_shared_quant(norm, hidden, residual)

    def test_native_collective_precedes_override_and_default_is_unchanged(self):
        communicator = object.__new__(comm.LayerCommunicator)
        communicator._sp_variant = None
        communicator._context = SimpleNamespace(attn_dp_size=1, cache=None)
        communicator._communicate_with_all_reduce_and_layer_norm_fn = partial(
            comm.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
            residual_input_mode=comm.ScatterMode.TP_ATTN_FULL,
        )
        hidden, residual = torch.ones(1, 8), torch.full((1, 8), 2.0)
        reduced = hidden * 16
        for override in (False, True):
            events = []

            def all_reduce(value):
                self.assertIs(value, hidden)
                events.append("all_reduce")
                return reduced

            def norm(value, skip):
                self.assertIs(value, reduced)
                self.assertIs(skip, residual)
                events.append("norm")
                return value + skip, skip

            communicator.post_attention_layernorm = (
                Mock(side_effect=AssertionError("wrong norm")) if override else norm
            )
            kwargs = {"norm_override": norm} if override else {}
            with (
                patch.object(comm, "get_attn_tp_context", return_value=SimpleNamespace(input_scattered=False)),
                patch.object(comm, "apply_aiter_all_reduce_fusion", return_value=False),
                patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=False),
                patch.object(comm, "attention_tensor_model_parallel_all_reduce", all_reduce),
            ):
                actual, skip = communicator.prepare_mlp(
                    hidden, residual, decode_batch(), **kwargs
                )
            self.assertEqual(events, ["all_reduce", "norm"])
            self.assertTrue(torch.equal(actual, reduced + residual))
            self.assertIs(skip, residual)

    def test_carrier_is_shared_only_in_both_execution_orders(self):
        hidden = torch.ones((1, 8), dtype=torch.bfloat16)
        carrier = object()
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                moe = module_without_constructor(model.DeepseekV2MoE)
                shared = Mock(return_value=torch.full_like(hidden, 2))
                moe.shared_experts = shared
                moe.num_fused_shared_experts = 0
                moe._fuse_shared_experts_inside_sbo = False
                moe._shared_expert_tp1 = False
                moe.is_nextn = False
                moe.routed_scaling_factor = 1.0
                moe.tp_size = 1
                moe.gate = Mock(return_value=torch.ones(1, 4))
                topk_result = object()
                moe.topk = Mock(return_value=topk_result)
                moe.experts = Mock(return_value=torch.full_like(hidden, 3))
                moe.experts.moe_runner_config = SimpleNamespace(inplace=inplace)
                moe.experts.quant_method = object()
                moe._maybe_quant_moe_input_once = Mock(return_value=None)
                with (
                    patch.object(model, "use_intel_amx_backend", return_value=False),
                    patch.object(model, "get_exec", return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False))),
                    patch.object(model, "maybe_fuse_routed_scale_and_shared_add", side_effect=lambda _, routed, shared, scale: routed + shared),
                ):
                    result = moe.forward_normal(
                        hidden, shared_gateup_pre_quant=carrier
                    )
                shared.assert_called_once_with(hidden, gateup_pre_quant=carrier)
                moe.experts.assert_called_once_with(hidden, topk_result)
                self.assertIs(moe.gate.call_args.args[0], hidden)
                self.assertTrue(torch.equal(result, torch.full_like(hidden, 5)))


if __name__ == "__main__":
    unittest.main()
