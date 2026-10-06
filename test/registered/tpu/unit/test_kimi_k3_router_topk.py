"""Kimi-K3 fused TPU router (SGLANG_TPU_K3_FUSED_GATE) vs the path it replaces:
a bf16 ``F.linear`` router plus ``biased_grouped_topk_impl``.

Ids match exactly when the top-k is unique. On an exact tie across the top-k
boundary the kernel picks the lowest ids (jax.lax.top_k semantics); XLA's TPU
``torch.topk`` does not (a run of ties from id 100 yields [120..123, 100..103]),
so there both must select the same scores. Requires torch_tpu and a TPU.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import importlib.util
import unittest

import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase


def _tpu_available() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1).to("tpu")
        return True
    except Exception:
        return False


def _exact_inputs(rows, hidden, n, seed):
    # Small integers times powers of two: the GEMV is exact in fp32, so the
    # logits do not depend on accumulation order.
    g = torch.Generator().manual_seed(seed)
    x = (torch.randint(-3, 4, (rows, hidden), generator=g) * 0.25).bfloat16()
    w = (torch.randint(-4, 5, (n, hidden), generator=g) * 2.0**-9).bfloat16()
    return x, w


@unittest.skipUnless(_tpu_available(), "requires torch_tpu and a TPU device")
class TestK3FusedRouterVsUnfused(CustomTestCase):
    def _check(self, x, w_nk, bias, k, renormalize=True):
        """Returns (fused ids sorted per row, rows with a tie at the boundary)."""
        from sglang.srt.layers.moe.pallas.router_topk import gate_route_tpu
        from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

        x, w_nk, bias = x.to("tpu"), w_nk.to("tpu"), bias.to("tpu")
        logits = F.linear(x, w_nk)
        ref_w, ref_ids = biased_grouped_topk_impl(
            x, logits, bias, k, renormalize, 1, 1, routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )  # fmt: skip
        scale = 2.5 if renormalize else 1.0
        w, ids = gate_route_tpu(
            x, w_nk.t().contiguous(), bias.reshape(1, -1), k, renormalize, scale, 1e-20
        )
        self.assertEqual((ids.dtype, w.dtype), (torch.int32, torch.float32))
        # The reference's selection scores, and its lowest-id top-k.
        biased = logits.sigmoid().float().cpu() + bias.cpu()
        lowest = biased.sort(dim=1, descending=True, stable=True).indices[:, :k]
        unique = (biased >= biased.gather(1, lowest[:, -1:])).sum(dim=1) == k
        ids = ids.cpu().long().sort(1).values
        ref_ids = ref_ids.cpu().long().sort(1).values

        torch.testing.assert_close(ids, lowest.sort(1).values, rtol=0, atol=0)
        torch.testing.assert_close(ids[unique], ref_ids[unique], rtol=0, atol=0)
        torch.testing.assert_close(
            biased.gather(1, ids).sort(1).values,
            biased.gather(1, ref_ids).sort(1).values,
            rtol=0,
            atol=0,
        )
        # Tied experts have equal scores, hence equal weights.
        torch.testing.assert_close(
            w.cpu().sort(1).values, ref_w.cpu().sort(1).values, rtol=2e-6, atol=1e-7
        )
        return ids, int((~unique).sum())

    def test_k3_shape_fp32_bias_matches_exactly(self):
        bias = torch.randn(896, generator=torch.Generator().manual_seed(7)) * 0.05
        for rows in (1, 8, 64):
            with self.subTest(rows=rows):
                x, w = _exact_inputs(rows, 7168, 896, seed=rows)
                _, ties = self._check(x, w, bias, k=16)
                self.assertEqual(ties, 0)

    def test_ties(self):
        n, k = 384, 8
        x, w = _exact_inputs(32, 2048, n, seed=11)
        # Coarse bf16 sigmoid scores and no bias tie across the boundary.
        _, ties = self._check(x, w, torch.zeros(n), k)
        self.assertGreater(ties, 0)
        self._check(x, w, torch.zeros(n), k, renormalize=False)
        # All scores exactly 0.5: the repeated bias values alone rank.
        zx = torch.zeros(4, 2048, dtype=torch.bfloat16)
        ids, _ = self._check(zx, w, (torch.arange(n) % 5).float() * 0.01, k)
        self.assertEqual(ids[0].tolist(), list(range(4, 40, 5)))
        ids, _ = self._check(zx, w, torch.zeros(n), k)
        self.assertEqual(ids[0].tolist(), list(range(k)))


if __name__ == "__main__":
    unittest.main()
