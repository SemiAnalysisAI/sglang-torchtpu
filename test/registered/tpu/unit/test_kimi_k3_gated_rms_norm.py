"""Kimi-K3 Pallas KDA o_norm (SGLANG_TPU_ENABLE_K3_FUSED_ONORM) vs the XLA chain.

The reference is the flag-off path on TPU: ``FusedRMSNormGated`` (sigmoid)
``forward_native`` over the TPU KDA backend's ``[1, T, H, D]`` output and the g
split of the fused ``[q, k, v, g]`` projection, then ``squeeze(0).flatten(-2)``.
The kernel reads g as column block 3 of the whole projection output and the
bf16 norm weight itself. It runs the same fp32 math, but Mosaic and XLA reduce
and evaluate rsqrt/sigmoid differently, so a bf16 rounding can move by one ulp
(about 1e-5 of the elements). Requires torch_tpu and a TPU device.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import importlib.util
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

_HEAD_DIM, _EPS = 128, 1e-5
# Kimi-K3 KDA heads per rank (96 heads) at TP32, TP16 and TP8.
_HEADS = (3, 6, 12)
_ROWS = (1, 4, 16, 64)


def _bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 representable values (bit patterns in ordered form)."""

    def ordered(t):
        bits = t.cpu().contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    return (ordered(a) - ordered(b)).abs()


def _tpu_available() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1).to("tpu")
        return True
    except Exception:
        return False


@unittest.skipUnless(_tpu_available(), "requires torch_tpu and a TPU device")
class TestK3GatedRmsNorm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.fla.fused_norm_gate import (
            FusedRMSNormGated,
        )
        from sglang.srt.hardware_backend.tpu.gated_rms_norm import (
            get_gated_rms_norm_op,
        )

        cls.gen = torch.Generator().manual_seed(0)
        norm = FusedRMSNormGated(_HEAD_DIM, eps=_EPS, activation="sigmoid")
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn(_HEAD_DIM, generator=cls.gen))
        # bf16 like the served parameter; the kernel upcasts it.
        cls.norm = norm.to(torch.bfloat16).to("tpu")
        # Registered before any trace, as _prepare_tpu_onorm does post-load.
        get_gated_rms_norm_op()

    def _assert_within_one_ulp(self, out, ref):
        self.assertEqual(out.dtype, ref.dtype)
        self.assertEqual(out.shape, ref.shape)
        dist = _bf16_ulp_distance(out, ref)
        self.assertLessEqual(int(dist.max()), 1)
        self.assertLessEqual((dist != 0).float().mean().item(), 1e-3)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen).bfloat16().to("tpu")

    def _reference(self, core, gate):
        gate = gate.unflatten(-1, (-1, _HEAD_DIM))
        return self.norm.forward_native(core, gate).squeeze(0).flatten(-2)

    def _kernel(self, core, g, g_block):
        from sglang.srt.hardware_backend.tpu.gated_rms_norm import (
            gated_rms_norm_tpu,
        )

        rows = g.shape[0]
        return gated_rms_norm_tpu(
            core.reshape(rows, -1, _HEAD_DIM), g, self.norm.weight, _EPS, g_block
        )

    def test_matches_forward_native_at_k3_head_counts(self):
        for heads in _HEADS:
            for rows in _ROWS:
                with self.subTest(heads=heads, rows=rows):
                    width = heads * _HEAD_DIM
                    core = self._randn(1, rows, heads, _HEAD_DIM)
                    qkvg = self._randn(rows, 4 * width)
                    ref = self._reference(core, qkvg[:, 3 * width :])
                    out = self._kernel(core, qkvg, g_block=3)
                    self.assertEqual(out.dtype, torch.bfloat16)
                    self.assertEqual(tuple(out.shape), (rows, width))
                    self._assert_within_one_ulp(out, ref)

    def test_rows_off_the_sublane_tile_and_head_dim_256(self):
        heads = 12
        width = heads * _HEAD_DIM
        for rows in (3, 5, 12):
            with self.subTest(rows=rows):
                core = self._randn(1, rows, heads, _HEAD_DIM)
                qkvg = self._randn(rows, 4 * width)
                ref = self._reference(core, qkvg[:, 3 * width :])
                self._assert_within_one_ulp(self._kernel(core, qkvg, g_block=3), ref)
        from sglang.kernels.ops.attention.fla.fused_norm_gate import (
            FusedRMSNormGated,
        )
        from sglang.srt.hardware_backend.tpu.gated_rms_norm import (
            gated_rms_norm_tpu,
        )

        head_dim, heads, rows = 256, 3, 4
        width = heads * head_dim
        norm = FusedRMSNormGated(head_dim, eps=_EPS, activation="sigmoid")
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn(head_dim, generator=self.gen))
        norm = norm.to(torch.bfloat16).to("tpu")
        core = self._randn(1, rows, heads, head_dim)
        qkvg = self._randn(rows, 4 * width)
        gate = qkvg[:, 3 * width :].unflatten(-1, (-1, head_dim))
        ref = norm.forward_native(core, gate).squeeze(0).flatten(-2)
        out = gated_rms_norm_tpu(
            core.reshape(rows, -1, head_dim), qkvg, norm.weight, _EPS, 3
        )
        self._assert_within_one_ulp(out, ref)

    def test_g_block_addresses_one_column_block(self):
        """Every block of a [T, 4W] buffer and a standalone [T, W] gate."""
        heads, rows = 3, 4
        width = heads * _HEAD_DIM
        core = self._randn(1, rows, heads, _HEAD_DIM)
        qkvg = self._randn(rows, 4 * width)
        for g_block in range(4):
            with self.subTest(g_block=g_block):
                gate = qkvg[:, g_block * width : (g_block + 1) * width]
                ref = self._reference(core, gate)
                out = self._kernel(core, qkvg, g_block=g_block)
                self._assert_within_one_ulp(out, ref)
        gate = self._randn(rows, width)
        out = self._kernel(core, gate, g_block=0)
        self._assert_within_one_ulp(out, self._reference(core, gate))

    def test_compiled_with_in_graph_projection(self):
        """Inside torch.compile with [q, k, v, g] produced in the graph, against
        the compiled flag-off chain."""
        import torch._dynamo

        from sglang.srt.hardware_backend.tpu.gated_rms_norm import (
            gated_rms_norm_tpu,
        )

        norm, hidden = self.norm, 512

        def xla_chain(h, w, core):
            qkvg = h @ w
            width = core.shape[-2] * _HEAD_DIM
            gate = qkvg[:, 3 * width :].unflatten(-1, (-1, _HEAD_DIM))
            return norm.forward_native(core, gate).squeeze(0).flatten(-2)

        def pallas(h, w, core):
            qkvg = h @ w
            x = core.reshape(qkvg.shape[0], -1, _HEAD_DIM)
            return gated_rms_norm_tpu(x, qkvg, norm.weight, _EPS, 3)

        for heads in _HEADS:
            for rows in (1, 4, 64):
                with self.subTest(heads=heads, rows=rows):
                    width = heads * _HEAD_DIM
                    h = self._randn(rows, hidden)
                    w = self._randn(hidden, 4 * width) * hidden**-0.5
                    core = self._randn(1, rows, heads, _HEAD_DIM)
                    torch._dynamo.reset()
                    # Serving runs without autograd; the op has no backward.
                    with torch.no_grad():
                        ref, out = (
                            torch.compile(
                                fn, backend="tpu", dynamic=False, fullgraph=True
                            )(h, w, core)
                            for fn in (xla_chain, pallas)
                        )
                    self._assert_within_one_ulp(out, ref)


if __name__ == "__main__":
    unittest.main()
