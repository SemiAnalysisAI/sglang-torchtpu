"""Mega KDA prefill path vs the fused GDN v3 kernel (KDA, bounded gate) at Flash per-rank shapes."""
import sys, time
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn
from sglang_tpu.attention.pallas.gdn.config import AttentionMode
from sglang_tpu.kernels.mega_kda import kda_prefill_mega
H, DK, DV, K = 8, 128, 128, 4
DIM = 2 * H * DK + H * DV
T = 8192; S = 9
lens = [5000, 2500, 13]; prefix = [0, 700, 0]
qsl = jnp.array([0, 5000, 7500, 7513], jnp.int32)
r = iter(jax.random.split(jax.random.PRNGKey(0), 16))
qkv = (jax.random.normal(next(r), (T, DIM)) * 0.5).astype(jnp.bfloat16)
b = jax.random.normal(next(r), (T, H)).astype(jnp.bfloat16)
a = jax.random.normal(next(r), (T, H * DK)).astype(jnp.bfloat16)
conv_state = (jax.random.normal(next(r), (S, K - 1, 1, DIM)) * 0.5).astype(jnp.bfloat16)
rec = (jax.random.normal(next(r), (S, H, DK, DV)) * 0.05).astype(jnp.float32)
w = (jax.random.normal(next(r), (DIM, 1, K)) * 0.3).astype(jnp.bfloat16)
a_log = jax.random.normal(next(r), (H,)) * 0.5
dt_bias = jax.random.normal(next(r), (H, DK)) * 0.1
idx = jnp.array([3, 5, 7], jnp.int32)
seq_lens = jnp.array([l + p for l, p in zip(lens, prefix)], jnp.int32)
dist = jnp.array([0, 3, 3], jnp.int32)
kw = dict(n_kq=H, n_v=H, d_k=DK, d_v=DV, kernel_size=K)
def fused(qkv, b, a, cs, rs):
    (nc, nr), out = fused_conv1d_gdn(qkv, b, a, cs, rs, w, None, a_log, dt_bias, qsl, idx, dist, seq_lens, None,
                                     attention_mode=AttentionMode.KDA, gate_lower_bound=-5.0, prefill_only=True, **kw)
    return nc, nr, out.reshape(-1, H, DV)
def mega(qkv, b, a, cs, rs):
    return kda_prefill_mega(qkv, b, a, cs, rs, w, None, a_log, dt_bias, qsl, idx, seq_lens, None, gate_lower_bound=-5.0, **kw)
res = {}
for name, f in (("fused", fused), ("mega", mega)):
    jf = jax.jit(f)
    o = jax.block_until_ready(jf(qkv, b, a, conv_state, rec))
    n = 20; t0 = time.perf_counter()
    for _ in range(n): o2 = jf(qkv, b, a, conv_state, rec)
    jax.block_until_ready(o2)
    res[name] = o; print(f"{name}: {(time.perf_counter()-t0)/n*1e3:.3f} ms", flush=True)
(fc, fr, fo), (mc, mr, mo) = res["fused"], res["mega"]
def rel(x, y):
    x = x.astype(jnp.float32); y = y.astype(jnp.float32)
    return float(jnp.linalg.norm(x - y) / (jnp.linalg.norm(y) + 1e-12)), float(jnp.max(jnp.abs(x - y)))
print("out rel/maxabs (live rows):", rel(mo[:7513], fo[:7513]), " padding rows max:", float(jnp.max(jnp.abs(mo[7513:].astype(jnp.float32)))), float(jnp.max(jnp.abs(fo[7513:].astype(jnp.float32)))))
for s, i in enumerate([3, 5, 7]):
    print(f"seq {s} slot {i}: rec rel/maxabs {rel(mr[i], fr[i])}  conv maxabs {rel(mc[i], fc[i])[1]}")
untouched = [j for j in range(S) if j not in (3, 5, 7)]
print("untouched slots identical:", bool(jnp.all(mr[jnp.array(untouched)] == rec[jnp.array(untouched)])), bool(jnp.all(mc[jnp.array(untouched)] == conv_state[jnp.array(untouched)])))

# Packed-weight path through the production dispatcher (as TpuKDAAttnBackend calls it).
from sglang_tpu.attention_gdn_pallas import _kda_prefill_with_fallback
rows = -(-(K + H + 1) // 8) * 8; width = -(-max(DIM, DK, H) // 128) * 128
packed = jnp.zeros((rows, width), jnp.float32)
packed = packed.at[:K, :DIM].set(w.astype(jnp.float32).reshape(DIM, K).T)
packed = packed.at[K:K + H, :DK].set(dt_bias.astype(jnp.float32))
packed = packed.at[K + H, :H].set(a_log.astype(jnp.float32))
def via_dispatch(qkv, b, a, cs, rs):
    return _kda_prefill_with_fallback(qkv, b, a, cs, rs, w, None, a_log, dt_bias, qsl, idx, dist, seq_lens, None, packed,
                                      n_kq=H, n_v=H, d_k=DK, d_v=DV, kernel_size=K, gate_lower_bound=-5.0)
dc, dr, do = jax.block_until_ready(jax.jit(via_dispatch)(qkv, b, a, conv_state, rec))
print("dispatch(packed) vs mega(unpacked): out identical", bool(jnp.all(do == mo)), "rec identical", bool(jnp.all(dr == mr)), "conv identical", bool(jnp.all(dc == mc)))
