"""Static-input-scale fp8 GEMV for RMSNorm-fed weights (TPU decode).

`quantize_rmsnorm_fed_weight` fixes the activation scale at
sqrt(K) * max|g| / fp8_max instead of reducing an amax every step. That is
only safe because an RMSNorm output is bounded: y = x / rms(x) * g gives
|y_i| <= sqrt(K) * |g_i|. These tests pin the bound (the fp8 cast does not
saturate, so an over-range value becomes NaN), the accuracy against a bf16
reference, and that every bound tensor is a base tensor (torch_tpu
re-materializes a view graph input on every call).

They run on CPU and cover the arithmetic; TPU lowering is covered by the
Kimi-K3 serving run.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.fp8_linear import (
    apply_tpu_static_fp8_linear,
    quantize_rmsnorm_fed_weight,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
K, N = 512, 256


def _rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g.float()).to(
        torch.bfloat16
    )


def _inputs(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    w = (torch.randn(K, N, generator=gen) * 0.02).to(torch.bfloat16)
    g = (torch.rand(K, generator=gen) * 0.5 + 0.05).to(torch.bfloat16)
    return w, g, gen


class TestStaticFp8Bound(CustomTestCase):
    def test_worst_case_rmsnorm_output_stays_in_range(self):
        """All energy in the element with the largest |g|: |y| hits sqrt(K)*g_max."""
        w, g, _ = _inputs()
        packed = quantize_rmsnorm_fed_weight(w, g)
        x = torch.zeros(1, K)
        x[0, int(g.float().abs().argmax())] = 1.0
        y = _rmsnorm(x, g)
        self.assertLessEqual(
            float((y.float() * packed.inv_x_scale).abs().max()), FP8_MAX
        )
        out = apply_tpu_static_fp8_linear(y, packed)
        self.assertTrue(torch.isfinite(out).all())

    def test_heavy_tailed_inputs_are_finite_and_accurate(self):
        w, g, gen = _inputs(1)
        packed = quantize_rmsnorm_fed_weight(w, g)
        x = torch.distributions.StudentT(3.0).sample((8, K))
        x[:, :4] *= 50  # a few massive channels, as in a residual stream
        y = _rmsnorm(x, g)
        ref = y.float() @ w.float()
        out = apply_tpu_static_fp8_linear(y, packed).float()
        self.assertTrue(torch.isfinite(out).all())
        rel = float((out - ref).norm() / ref.norm())
        self.assertLess(rel, 0.06)


class TestStaticFp8Apply(CustomTestCase):
    def test_matches_bf16_reference(self):
        w, g, gen = _inputs(2)
        packed = quantize_rmsnorm_fed_weight(w, g)
        y = _rmsnorm(torch.randn(4, K, generator=gen), g)
        ref = y.float() @ w.float()
        out = apply_tpu_static_fp8_linear(y, packed)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (4, N))
        rel = float((out.float() - ref).norm() / ref.norm())
        self.assertLess(rel, 0.06)

    def test_row_padding_does_not_change_real_rows(self):
        """M is padded to 16 for _scaled_mm; each row is independent of it."""
        w, g, gen = _inputs(3)
        packed = quantize_rmsnorm_fed_weight(w, g)
        y = _rmsnorm(torch.randn(17, K, generator=gen), g)
        full = apply_tpu_static_fp8_linear(y, packed)
        self.assertEqual(tuple(full.shape), (17, N))
        for m in (1, 5, 16):
            torch.testing.assert_close(
                apply_tpu_static_fp8_linear(y[:m], packed), full[:m], rtol=0, atol=0
            )

    def test_bound_weight_is_callable(self):
        """Models call the bound weight directly instead of importing the helper."""
        w, g, gen = _inputs(5)
        packed = quantize_rmsnorm_fed_weight(w, g)
        y = _rmsnorm(torch.randn(3, K, generator=gen), g)
        torch.testing.assert_close(
            packed(y), apply_tpu_static_fp8_linear(y, packed), rtol=0, atol=0
        )


class TestStaticFp8Pack(CustomTestCase):
    def test_bound_tensors_are_contiguous_base_tensors(self):
        """A transposed [N, K] parameter must not leave views in the pack."""
        w, g, _ = _inputs(4)
        w_nk = w.t().contiguous()  # how nn.Linear stores it
        packed = quantize_rmsnorm_fed_weight(w_nk.t(), g)
        for name in ("weight", "out_scale", "one"):
            t = getattr(packed, name)
            with self.subTest(name=name):
                self.assertIsNone(t._base)
                self.assertTrue(t.is_contiguous())
        self.assertEqual(packed.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(packed.out_scale.shape), (N,))

    def test_rejects_mismatched_norm_width(self):
        w, g, _ = _inputs()
        with self.assertRaises(ValueError):
            quantize_rmsnorm_fed_weight(w, g[:-1])

    def test_rejects_n_not_multiple_of_16(self):
        w, g, _ = _inputs()
        with self.assertRaises(ValueError):
            quantize_rmsnorm_fed_weight(w[:, :250], g)


class TestStaticFp8PallasGemvHelpers(CustomTestCase):
    """CPU-side pieces of the Pallas GEMV path: N padding and the K block choice.

    The kernel itself is covered on TPU by test_static_fp8_gemv_pallas.py.
    """

    def test_padded_n_matches_unpadded(self):
        """N=2112 (Kimi-K3 MLA q_a) padded to 2176 must not change the output."""
        from sglang_tpu.static_fp8_gemv_pallas import pad_n

        gen = torch.Generator().manual_seed(6)
        w = (torch.randn(K, 2112, generator=gen) * 0.02).to(torch.bfloat16)
        g = (torch.rand(K, generator=gen) * 0.5 + 0.05).to(torch.bfloat16)
        packed = quantize_rmsnorm_fed_weight(w, g)
        padded = pad_n(packed, force=True)
        self.assertEqual(tuple(padded.weight.shape), (K, 2176))
        self.assertEqual(padded.n_out, 2112)
        self.assertIsNone(padded.weight._base)
        self.assertIsNone(padded.out_scale._base)
        y = _rmsnorm(torch.randn(3, K, generator=gen), g)
        torch.testing.assert_close(
            apply_tpu_static_fp8_linear(y, padded),
            apply_tpu_static_fp8_linear(y, packed),
            rtol=0,
            atol=0,
        )

    def test_aligned_n_is_left_alone(self):
        from sglang_tpu.static_fp8_gemv_pallas import pad_n

        w, g, _ = _inputs(7)
        packed = quantize_rmsnorm_fed_weight(w, g)
        self.assertIs(pad_n(packed, force=True), packed)
        self.assertIs(pad_n(packed), packed)  # kernel not registered on CPU

    def test_inv_x_tensor_matches_scalar(self):
        w, g, _ = _inputs(8)
        packed = quantize_rmsnorm_fed_weight(w, g)
        # f32 tensor for the kernel, Python float for the XLA path: equal
        # up to the f32 rounding.
        self.assertEqual(
            float(packed.inv_x),
            float(torch.tensor(packed.inv_x_scale, dtype=torch.float32)),
        )

    def test_block_k_divides_k_and_fits_vmem(self):
        from sglang_tpu.static_fp8_gemv_pallas import block_k

        # Kimi-K3 TP32 shapes: KDA qkvg, MoE front, latent up.
        for k, n in ((7168, 1536), (7168, 3968), (3584, 7168)):
            with self.subTest(k=k, n=n):
                bk = block_k(k, n)
                self.assertGreater(bk, 0)
                self.assertEqual(k % bk, 0)
                self.assertEqual(bk % 128, 0)
                self.assertLessEqual(bk, 1792)
        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8_GEMV_BLOCK_K.override(1024):
            self.assertEqual(block_k(7168, 1536), 1024)
        # A weight too wide for any K block falls back to the XLA path.
        self.assertEqual(block_k(7168, 1 << 20), 0)


class TestKimiK3StaticFp8Selector(CustomTestCase):
    def test_parts_and_all(self):
        from sglang_tpu.kimi_k3_static_fp8 import PARTS, selected_parts

        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("moe_front,kda_qkvg"):
            self.assertEqual(selected_parts(), {"moe_front", "kda_qkvg"})
        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("all"):
            self.assertEqual(selected_parts(), PARTS)
        self.assertEqual(selected_parts(), frozenset())  # unset: everything bf16

    def test_unknown_part_raises(self):
        from sglang_tpu.kimi_k3_static_fp8 import selected_parts

        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("moe_frnt"):
            with self.assertRaises(ValueError):
                selected_parts()


if __name__ == "__main__":
    unittest.main()
