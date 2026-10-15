"""mHC site kernel at prefill size: device time per post+pre site vs token block size."""
import functools, sys, time
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])  # tree root holding sglang_tpu
import sglang_tpu.kernels.glm_mhc as m
H, D, ITERS = 4, 4096, 20
MIX = (2 + H) * H
T = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
k = iter(jax.random.split(jax.random.PRNGKey(0), 8))
res = jax.random.normal(next(k), (T, H * D)).astype(jnp.bfloat16)
x = jax.random.normal(next(k), (T, D)).astype(jnp.bfloat16)
post = jax.nn.sigmoid(jax.random.normal(next(k), (T, H)))
comb = jax.nn.softmax(jax.random.normal(next(k), (T, H * H)), -1)
fn = jax.random.normal(next(k), (MIX, H * D)) * 0.02
vec = jnp.zeros((1, m.VEC_WIDTH), jnp.float32).at[0, MIX:MIX + 3].set(jnp.array([0.7, 1.3, 2.0]))
norm = jnp.ones((D,), jnp.float32)
kw = dict(hc=H, iters=ITERS, rms_eps=1e-6, hc_eps=1e-6, norm_weight=norm, norm_eps=1e-6)
ref = None
for bt in [int(b) for b in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["32", "64", "128", "256"])]:
    m._MAX_BLOCK_TOKENS = bt
    f = jax.jit(lambda x, res, post, comb: m.mhc_site(x, res, post, comb, fn, vec, **kw))
    try:
        out = jax.block_until_ready(f(x, res, post, comb))
    except Exception as e:
        print(f"bt={bt}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
    n = 30; t0 = time.perf_counter()
    for _ in range(n): o = f(x, res, post, comb)
    jax.block_until_ready(o)
    dt = (time.perf_counter() - t0) / n * 1e3
    if ref is None: ref = out
    err = max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) for a, b in zip(out, ref))
    print(f"T={T} bt={bt}: {dt:.3f} ms per site, max |diff| vs bt=32 {err:.3g}", flush=True)
