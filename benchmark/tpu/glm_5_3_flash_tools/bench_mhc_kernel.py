"""Device time of 90 chained mHC seams (one decode step) : fused kernel vs XLA reference."""
import importlib.util, os, sys, time
import jax, jax.numpy as jnp
_spec = importlib.util.spec_from_file_location("glm_mhc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "python/sglang/srt/hardware_backend/tpu/kernels/glm_mhc.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
from check_mhc_kernel import ref_pre, ref_post, H, D, MIX, ITERS, RMS_EPS, HC_EPS
SITES = 90

def run(T):
    k = iter(jax.random.split(jax.random.PRNGKey(0), 8))
    res = jax.random.normal(next(k), (T, H * D)).astype(jnp.bfloat16)
    out = jax.random.normal(next(k), (T, D)).astype(jnp.bfloat16)
    fns = jax.random.normal(next(k), (SITES, MIX, H * D)) * 0.02
    scale = jnp.array([0.7, 1.3, 2.0], jnp.float32); base = jax.random.normal(next(k), (MIX,)) * 0.5
    vec = _m.pack_hc_vec(scale, base)
    kw = dict(hc=H, iters=ITERS, rms_eps=RMS_EPS, hc_eps=HC_EPS)

    @jax.jit
    def fused(res, out, fns):
        lay, post, comb = _m.mhc_site(None, res, None, None, fns[0], vec, **kw)
        def body(i, c):
            r, l, p, cb = c
            x = (l.astype(jnp.float32) * 0.5 + out.astype(jnp.float32) * 0.5).astype(jnp.bfloat16)
            return _m.mhc_site(x, r, p, cb, fns[i], vec, **kw)
        return jax.lax.fori_loop(1, SITES, body, (res, lay, post, comb))

    @jax.jit
    def ref(res, out, fns):
        lay, post, comb = ref_pre(res.reshape(T, H, D), fns[0], scale, base)
        def body(i, c):
            r, l, p, cb = c
            x = (l.astype(jnp.float32) * 0.5 + out.astype(jnp.float32) * 0.5).astype(jnp.bfloat16)
            nr = ref_post(x, r.reshape(T, H, D), p, cb)
            l2, p2, c2 = ref_pre(nr, fns[i], scale, base)
            return nr.reshape(T, -1), l2, p2, c2
        return jax.lax.fori_loop(1, SITES, body, (res, lay, post, comb))

    for name, f in (("fused", fused), ("xla ref", ref)):
        jax.block_until_ready(f(res, out, fns))
        t0 = time.perf_counter(); n = 20
        for _ in range(n): o = f(res, out, fns)
        jax.block_until_ready(o)
        print(f"T={T} {name}: {(time.perf_counter()-t0)/n*1e3:.3f} ms per {SITES} sites", flush=True)

for t in (sys.argv[1:] or ["1", "16", "8192"]): run(int(t))
