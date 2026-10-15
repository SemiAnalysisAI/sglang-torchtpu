"""Which is closer to float64 truth: fused kernel or torch_tpu reference hc_pre?"""
import sys; sys.path.insert(0, "test/registered/tpu/unit")
import torch, torch_tpu  # noqa
from test_glm_mhc_kernel import CFG, SITE, HC, HIDDEN
from sglang.srt.models.glm5_next_tpu import hc_pre
from sglang.srt.hardware_backend.tpu.glm_mhc import glm_mhc_ops, register_glm_mhc_ops
from sglang.srt.hardware_backend.tpu.kernels.glm_mhc import VEC_WIDTH
register_glm_mhc_ops(); pre, seam, post = glm_mhc_ops()
g = torch.Generator().manual_seed(7); mix = (2 + HC) * HC
fn = torch.randn(mix, HC * HIDDEN, generator=g) * 0.02; scale = torch.tensor([0.7, 1.3, 2.0]); base = torch.randn(mix, generator=g) * 0.5
vec = torch.zeros(1, VEC_WIDTH); vec[0, :mix] = base; vec[0, mix:mix + 3] = scale
for T in (1, 16):
    x = torch.randn(T, HC, HIDDEN, generator=g).to(torch.bfloat16)
    r64 = hc_pre(x.double(), fn.double(), scale.double(), base.double(), CFG)
    rt = [t.cpu() for t in hc_pre(x.to("tpu"), fn.to("tpu"), scale.to("tpu"), base.to("tpu"), CFG)]
    rk = [t.cpu() for t in pre(x.reshape(T, -1).to("tpu"), fn.to("tpu"), vec.to("tpu"), *SITE)]
    rk[2] = rk[2].view(-1, HC, HC)
    for name, a, b, c in zip(("layer", "post", "comb"), rk, rt, r64):
        e = lambda u: ((u.double() - c.double()).norm() / c.double().norm()).item()
        print(f"T={T} {name}: kernel vs f64 {e(a):.2e} | torch_tpu ref vs f64 {e(b):.2e}")
