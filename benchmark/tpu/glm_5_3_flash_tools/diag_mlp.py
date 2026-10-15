"""Where does the fused MLP diverge from DeepseekV2MLP's TPU chain?"""
import sys; sys.path.insert(0, "test/registered/tpu/unit")
import torch, torch.nn.functional as F, torch_tpu  # noqa
import jax, jax.numpy as jnp, numpy as np
from sglang.srt.hardware_backend.tpu import fp8_linear
from test_fp8_mlp_small_m import HIDDEN, INTER, LIMIT
g = torch.Generator().manual_seed(5)
fp8 = lambda *s: (torch.randn(*s, generator=g) * 0.5).to(torch.float8_e4m3fn)
w13, w2 = fp8(HIDDEN, 2 * INTER).to("tpu"), fp8(INTER, HIDDEN).to("tpu")
s13 = (torch.rand(2 * INTER, generator=g) * 0.01 + 1e-3).to("tpu"); s2 = (torch.rand(HIDDEN, generator=g) * 0.01 + 1e-3).to("tpu")
x = (torch.randn(1, HIDDEN, generator=g) * 4).bfloat16().to("tpu")
gu = fp8_linear.apply_tpu_fp8_linear(x, w13, s13)
gg, uu = gu.chunk(2, -1); gg = gg.clamp(max=LIMIT); uu = uu.clamp(min=-LIMIT, max=LIMIT)
silu_t = F.silu(gg); h_t = silu_t * uu
print("torch silu dtype", silu_t.dtype)
# candidate silu formulas from the clamped bf16 gate
g32 = gg.float()
cands = {"f32 x*sigmoid, round": (g32 * torch.sigmoid(g32)).bfloat16(),
         "bf16 sigmoid then bf16 mul": (gg * torch.sigmoid(gg)),
         "f32 x/(1+exp(-x)), round": (g32 / (1 + torch.exp(-g32))).bfloat16()}
for k, v in cands.items():
    print(f"{k:28s} silu exact={torch.equal(v.cpu(), silu_t.cpu())} mismatches={(v.cpu()!=silu_t.cpu()).sum().item()}")
h_f = (cands["f32 x*sigmoid, round"].float() * uu.float()).bfloat16()
print("h mismatches (kernel formula):", (h_f.cpu() != h_t.cpu()).sum().item(), "/", h_t.numel())
fp8_linear.register_fp8_mlp_small_m_op()
out = fp8_linear._SMALL_M_MLP_OP(x, w13, s13, w2, s2, LIMIT).float().cpu()
ref = fp8_linear.apply_tpu_fp8_linear(h_t, w2, s2).float().cpu()
ref_h = fp8_linear.apply_tpu_fp8_linear(h_f, w2, s2).float().cpu()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"kernel vs ref {rel(out, ref):.3e} | kernel vs ref-with-kernel-h {rel(out, ref_h):.3e} | ref-h-variants {rel(ref_h, ref):.3e}")
