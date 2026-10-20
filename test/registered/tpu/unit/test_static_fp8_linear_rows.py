"""Kimi-K3's static fp8 linear through the registered torch op, at every row
count a decode or speculative-verify batch can bring.

test_static_fp8_gemv_pallas.py checks the Pallas kernel alone at a few row
counts. This goes through the path the model runs:
- apply_tpu_static_fp8_linear;
- the eligibility gate;
- the cpu_export torch op, which takes the Pallas GEMV up to 32 rows;
- the XLA fp8 dot past that (with its row padding);
- the n_out narrow of a padded N.

It runs every M from 1 to 33, plus 64, eager and under torch.compile, and
compares against an fp64 host reference built from the same quantized
operands. It also checks strided split views of the output, which the MoE
front hands to the next kernels.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import torch
    import torch_tpu  # noqa: F401

    # torch.tpu.is_available() reports False under pytest even with chips
    # attached; a tensor on the device is the test that matters.
    torch.empty(1, device="tpu")
    _HAS_TPU = True
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

# Kimi-K3 per-TP32-rank shapes: MoE front (latent down | shared gate_up),
# latent up, and MLA q_a (N=2112, padded to 2176 with n_out).
SHAPES = ((7168, 1152), (2048, 7168), (7168, 2112))
ROWS = tuple(range(1, 34)) + (64,)


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestStaticFp8LinearRows(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu import fp8_linear
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.patches import _register_tpu_compile_backend

        cls.fl = fp8_linear
        with tpu_envs.SGLANG_TPU_ENABLE_K3_STATIC_FP8_PALLAS_GEMV.override(True):
            fp8_linear.register_static_fp8_gemv_op()
        _register_tpu_compile_backend()
        # In a tuple: a function stored on the class would bind as a method.
        cls.compiled = (
            torch.compile(
                fp8_linear.apply_tpu_static_fp8_linear, backend="tpu", dynamic=False
            ),
        )

    def _weight(self, k, n, seed):
        g = torch.Generator().manual_seed(seed)
        w = (torch.randn(k, n, generator=g) * 0.02).to(torch.bfloat16).to("tpu")
        norm = (torch.rand(k, generator=g) + 0.5).to(torch.bfloat16).to("tpu")
        return self.fl.pad_static_fp8_n(self.fl.quantize_rmsnorm_fed_weight(w, norm))

    @staticmethod
    def _reference(x, sw):
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        xq = (x.float().cpu() * sw.inv_x_scale).clamp(-fp8_max, fp8_max)
        out = xq.to(torch.float8_e4m3fn).double() @ sw.weight.cpu().float().double()
        out = out * sw.out_scale.cpu().double()
        return out[:, : sw.n_out] if sw.n_out else out

    @staticmethod
    def _rel(got, want):
        got, want = got.double().cpu(), want.double().cpu()
        return float((got - want).abs().max() / want.abs().max())

    def test_every_row_count_matches_the_fp64_reference(self):
        apply = self.fl.apply_tpu_static_fp8_linear
        for k, n in SHAPES:
            sw = self._weight(k, n, k + n)
            for m in ROWS:
                with self.subTest(k=k, n=n, m=m):
                    g = torch.Generator().manual_seed(m)
                    x = (torch.randn(m, k, generator=g) * 0.5).to(torch.bfloat16)
                    x = x.to("tpu")
                    want = self._reference(x, sw)
                    for got in (apply(x, sw), self.compiled[0](x, sw)):
                        self.assertEqual(tuple(got.shape), tuple(want.shape))
                        # bf16 output rounding of an fp32-accumulated dot.
                        self.assertLess(self._rel(got, want), 1e-2)

    def test_split_views_of_the_output_read_back_exactly(self):
        apply = self.fl.apply_tpu_static_fp8_linear
        k, n = SHAPES[0]
        front = self._weight(k, n, 1)
        nxt = self._weight(512, 256, 2)

        def chain(x, contiguous):
            a, b = torch.split(apply(x, front), [512, n - 512], dim=-1)
            if contiguous:
                a, b = a.contiguous(), b.contiguous()
            return apply(a, nxt), b

        compiled_chain = torch.compile(chain, backend="tpu", dynamic=False)
        for m in (1, 2, 4, 8, 16, 32):
            with self.subTest(m=m):
                x = (torch.randn(m, k) * 0.5).to(torch.bfloat16).to("tpu")
                want_a, want_b = chain(x, True)
                for got_a, got_b in (chain(x, False), compiled_chain(x, False)):
                    torch.testing.assert_close(
                        got_a.cpu(), want_a.cpu(), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        got_b.cpu(), want_b.cpu(), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
