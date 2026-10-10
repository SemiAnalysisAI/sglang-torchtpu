# SPDX-License-Identifier: Apache-2.0
"""Fused GLM-5.3-Flash mHC site ops vs the torch reference hc_pre / hc_post.

The reference runs in fp32 on CPU: torch_tpu evaluates its fp32 projection at
bf16 precision (gate errors ~3e-3), while the kernel keeps fp32 (~1e-6).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.glm5_next_tpu import hc_post, hc_pre
from sglang.test.ci.ci_register import register_tpu_ci

try:
    import torch_tpu  # noqa: F401

    from sglang_tpu.glm_mhc import (
        glm_mhc_ops,
        register_glm_mhc_ops,
    )
    from sglang_tpu.kernels.glm_mhc import VEC_WIDTH
except ImportError:
    torch_tpu = None

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")

HC, HIDDEN = 4, 4096
CFG = SimpleNamespace(hc_mult=HC, rms_norm_eps=1e-5, hc_eps=1e-6, hc_sinkhorn_iters=20)
SITE = (HC, CFG.hc_sinkhorn_iters, CFG.rms_norm_eps, CFG.hc_eps, CFG.rms_norm_eps)


@unittest.skipUnless(torch_tpu is not None, "Requires torch_tpu")
class TestGlmMhcKernel(unittest.TestCase):
    def setUp(self):
        register_glm_mhc_ops()
        self.pre, self.seam, self.post = glm_mhc_ops()
        self.gen = torch.Generator().manual_seed(7)
        mix = (2 + HC) * HC
        self.fn = torch.randn(mix, HC * HIDDEN, generator=self.gen) * 0.02
        self.scale = torch.tensor([0.7, 1.3, 2.0])
        self.base = torch.randn(mix, generator=self.gen) * 0.5
        self.vec = torch.zeros(1, VEC_WIDTH)
        self.vec[0, :mix] = self.base
        self.vec[0, mix : mix + 3] = self.scale
        # The next sublayer's input RMSNorm, applied in-kernel to the collapse.
        self.norm_weight = (
            1 + 0.1 * torch.randn(HIDDEN, generator=self.gen)
        ).bfloat16()
        self.norm_w32 = self.norm_weight.float()

    def ref_norm(self, layer_in):
        x = layer_in.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + CFG.rms_norm_eps)
        return (x * self.norm_weight.float()).bfloat16()

    def ref_pre(self, streams):
        return hc_pre(streams, self.fn, self.scale, self.base, CFG)

    def assertClose(self, actual, expected, rtol):
        actual, expected = actual.cpu().double(), expected.cpu().double()
        rel = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        self.assertLess(rel.item(), rtol, f"relative error {rel.item():.3e}")

    def test_pre_seam_post_match_reference(self):
        tpu = lambda t: t.to("tpu")
        # 40 tokens exercises the padded multi-block prefill path.
        for tokens in (1, 16, 40):
            with self.subTest(tokens=tokens):
                streams = torch.randn(tokens, HC, HIDDEN, generator=self.gen).bfloat16()
                out = torch.randn(tokens, HIDDEN, generator=self.gen).bfloat16()
                flat = streams.reshape(tokens, -1)
                ref_in, ref_post, ref_comb = self.ref_pre(streams)
                layer_in, post, comb = self.pre(
                    tpu(flat), tpu(self.fn), tpu(self.vec), tpu(self.norm_w32), *SITE
                )
                # Normalized collapsed inputs are bf16: one rounding step of error.
                self.assertClose(layer_in, self.ref_norm(ref_in), 4e-3)
                self.assertClose(post, ref_post, 1e-5)
                self.assertClose(comb.view(-1, HC, HC), ref_comb, 1e-5)

                gates = (tpu(ref_post), tpu(ref_comb.reshape(tokens, -1)))
                ref_res = hc_post(out, streams, ref_post, ref_comb)
                ref_in2, ref_post2, ref_comb2 = self.ref_pre(ref_res)
                res, layer_in2, post2, comb2 = self.seam(
                    tpu(out),
                    tpu(flat),
                    *gates,
                    tpu(self.fn),
                    tpu(self.vec),
                    tpu(self.norm_w32),
                    *SITE,
                )
                self.assertClose(res.view(-1, HC, HIDDEN), ref_res, 4e-3)
                self.assertClose(layer_in2, self.ref_norm(ref_in2), 4e-3)
                self.assertClose(post2, ref_post2, 1e-4)
                self.assertClose(comb2.view(-1, HC, HC), ref_comb2, 1e-4)

                last = self.post(tpu(out), tpu(flat), *gates, HC)
                self.assertClose(last.view(-1, HC, HIDDEN), ref_res, 4e-3)


if __name__ == "__main__":
    unittest.main()
