"""CPU contracts for reusing FP8 activations in the native TPU dense consumer.

Only the raw scaled-mm is replaced by an FP32 CPU reference. These tests cover
metadata, padding, operand reuse, scaling order, and output/bias boundaries;
they do not establish TPU kernel or compiler equivalence.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu import fp8_linear
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuFp8PrequantizedInput(CustomTestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(123)
        self.weight = torch.randn(32, 16, generator=generator).to(
            torch.float8_e4m3fn
        )

    @staticmethod
    def _scaled_mm(q, weight, *, scale_a, scale_b, out_dtype):
        assert scale_a.dtype == scale_b.dtype == torch.float32
        assert scale_a.shape == scale_b.shape == (1,)
        assert torch.equal(scale_a, torch.ones_like(scale_a))
        assert torch.equal(scale_b, torch.ones_like(scale_b))
        assert out_dtype == torch.float32
        return q.float() @ weight.float()

    @staticmethod
    def _prequantize(x):
        rows = x.reshape(-1, x.shape[-1])
        padded, _ = fp8_linear._pad_rows_to_scaled_mm_alignment(rows)
        q, scale = fp8_linear._quantize_activation_per_row(
            padded, torch.float8_e4m3fn
        )
        return fp8_linear.TpuFp8PrequantizedInput(
            q, scale, tuple(x.shape), x.dtype
        )

    def test_matches_tensor_input_for_logical_shapes_dtypes_and_scales(self):
        generator = torch.Generator().manual_seed(82)
        shapes = ((32,), (1, 32), (16, 32), (17, 32), (2, 3, 32))
        scales = (
            torch.tensor(0.017),
            torch.linspace(0.003, 0.07, 16),
            torch.linspace(0.003, 0.07, 16).reshape(-1, 1),
        )
        for shape in shapes:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                x = torch.randn(shape, generator=generator).to(dtype)
                prequantized = self._prequantize(x)
                for weight_scale in scales:
                    with self.subTest(
                        shape=shape, dtype=dtype, scale=weight_scale.shape
                    ):
                        with patch.object(torch, "_scaled_mm", self._scaled_mm):
                            expected = fp8_linear.apply_tpu_fp8_linear(
                                x, self.weight, weight_scale
                            )
                            actual = fp8_linear.apply_tpu_fp8_linear(
                                prequantized, self.weight, weight_scale
                            )
                        self.assertEqual(actual.shape, (*shape[:-1], 16))
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(torch.equal(actual, expected))

    def test_reuses_noncontiguous_q_and_scale_without_requantizing(self):
        q = torch.ones(32, 16).to(torch.float8_e4m3fn).t()
        scale = torch.full((16, 2), 0.25)[:, 1:]
        self.assertFalse(q.is_contiguous())
        self.assertFalse(scale.is_contiguous())
        value = fp8_linear.TpuFp8PrequantizedInput(
            q, scale, (1, 32), torch.bfloat16
        )
        with (
            patch.object(torch, "_scaled_mm", side_effect=self._scaled_mm) as dot,
            patch.object(
                fp8_linear,
                "_quantize_activation_per_row",
                side_effect=AssertionError("prequantized input was requantized"),
            ),
            patch.object(
                fp8_linear,
                "_pad_rows_to_scaled_mm_alignment",
                side_effect=AssertionError("prequantized input was padded again"),
            ),
        ):
            result = fp8_linear.apply_tpu_fp8_linear(
                value, self.weight, torch.ones(16)
            )
        self.assertIs(dot.call_args.args[0], q)
        self.assertIs(dot.call_args.args[1], self.weight)
        self.assertEqual(result.shape, (1, 16))
        self.assertIs(value.scale, scale)

    def test_scaling_remains_row_then_channel_in_fp32(self):
        # Reassociation changes the last FP32 bit for this constructed triple.
        raw = 6.430325508117676
        row_scale = 0.34760648012161255
        channel_scale = 2.8227901458740234
        value = self._prequantize(torch.ones(1, 32))
        value = value._replace(scale=torch.full((16, 1), row_scale))
        with patch.object(
            torch, "_scaled_mm", return_value=torch.full((16, 16), raw)
        ):
            result = fp8_linear.apply_tpu_fp8_linear(
                value, self.weight, torch.full((16,), channel_scale)
            )
        expected = torch.full((1, 16), 6.30956506729126)
        reassociated = torch.full((1, 16), 6.309564590454102)
        self.assertTrue(torch.equal(result, expected))
        self.assertFalse(torch.equal(result, reassociated))

    def test_cast_precedes_bias_and_preserves_bias_promotion(self):
        value = self._prequantize(torch.ones(1, 32, dtype=torch.bfloat16))
        value = value._replace(scale=torch.ones(16, 1))
        for bias_dtype, expected_value in (
            (torch.bfloat16, 1.0),
            (torch.float32, 1.00390625),
        ):
            with self.subTest(bias_dtype=bias_dtype):
                bias = torch.full((16,), 0.00390625, dtype=bias_dtype)
                with patch.object(
                    torch,
                    "_scaled_mm",
                    return_value=torch.full((16, 16), 1.00390625),
                ):
                    result = fp8_linear.apply_tpu_fp8_linear(
                        value, self.weight, torch.ones(16), bias
                    )
                self.assertEqual(result.dtype, bias_dtype)
                self.assertTrue(
                    torch.equal(result, torch.full_like(result, expected_value))
                )

    def test_invalid_metadata_is_rejected_before_scaled_mm(self):
        good = self._prequantize(torch.ones(1, 32, dtype=torch.bfloat16))
        invalid = {
            "no_logical_dimensions": good._replace(logical_shape=()),
            "wrong_hidden_width": good._replace(logical_shape=(1, 16)),
            "insufficient_padded_rows": good._replace(q=good.q[:1]),
            "wrong_quantized_width": good._replace(q=good.q[:, :16]),
            "wrong_quantized_dtype": good._replace(q=good.q.float()),
            "wrong_scale_shape": good._replace(scale=good.scale.reshape(-1)),
            "wrong_scale_dtype": good._replace(scale=good.scale.bfloat16()),
            "wrong_output_dtype": good._replace(output_dtype=torch.int32),
            "wrong_q_device": good._replace(q=good.q.to("meta")),
            "wrong_scale_device": good._replace(scale=good.scale.to("meta")),
            "wrong_logical_row_count": good._replace(logical_shape=(17, 32)),
        }
        for name, value in invalid.items():
            with self.subTest(case=name):
                with patch.object(torch, "_scaled_mm") as dot:
                    with self.assertRaises(ValueError):
                        fp8_linear.apply_tpu_fp8_linear(
                            value, self.weight, torch.ones(16)
                        )
                dot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
