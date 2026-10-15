"""Fused GLM mHC site kernel vs the torch-reference math (re-expressed in JAX, f32 HIGHEST)."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
import importlib.util, os
_spec = importlib.util.spec_from_file_location("glm_mhc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "python/sglang/srt/hardware_backend/tpu/kernels/glm_mhc.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
mhc_site, pack_hc_vec = _m.mhc_site, _m.pack_hc_vec

H, D, ITERS, RMS_EPS, HC_EPS = 4, 4096, 20, 1e-5, 1e-6
MIX = (2 + H) * H
HI = jax.lax.Precision.HIGHEST


def ref_pre(x, fn, scale, base):  # x [T, H, D] bf16, mirrors glm5_next_tpu.hc_pre
    flat = x.reshape(x.shape[0], -1).astype(jnp.float32)
    flat = flat * jax.lax.rsqrt(jnp.mean(flat * flat, -1, keepdims=True) + RMS_EPS)
    m = jnp.dot(flat, fn.T, precision=HI)
    pre, post, comb = m[:, :H], m[:, H:2 * H], m[:, 2 * H:]
    pre = jax.nn.sigmoid(pre * scale[0] + base[:H]) + HC_EPS
    post = 2 * jax.nn.sigmoid(post * scale[1] + base[H:2 * H])
    comb = jax.nn.softmax((comb * scale[2] + base[2 * H:]).reshape(-1, H, H), -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    return (pre[..., None] * x.astype(jnp.float32)).sum(-2).astype(x.dtype), post, comb


def ref_post(out, res, post, comb):
    y = post[..., None] * out.astype(jnp.float32)[:, None, :] + jnp.einsum(
        "tij,tid->tjd", comb, res.astype(jnp.float32), precision=HI)
    return y.astype(res.dtype)


def rel(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)), float(np.abs(a - b).max())


def main(T):
    k = iter(jax.random.split(jax.random.PRNGKey(T), 16))
    res = jax.random.normal(next(k), (T, H, D)).astype(jnp.bfloat16)
    out = jax.random.normal(next(k), (T, D)).astype(jnp.bfloat16)
    fn = jax.random.normal(next(k), (MIX, H * D)) * 0.02
    scale = jnp.array([0.7, 1.3, 2.0], jnp.float32)
    base = jax.random.normal(next(k), (MIX,)) * 0.5
    vec = pack_hc_vec(scale, base)
    kw = dict(hc=H, iters=ITERS, rms_eps=RMS_EPS, hc_eps=HC_EPS)
    worst = 0.0
    # pre only
    lay_r, post_r, comb_r = ref_pre(res, fn, scale, base)
    f_pre = jax.jit(lambda r, f, v: mhc_site(None, r, None, None, f, v, **kw))
    lay, post, comb = f_pre(res.reshape(T, -1), fn, vec)
    for name, a, b in (("pre.layer", lay, lay_r), ("pre.post", post, post_r), ("pre.comb", comb.reshape(T, H, H), comb_r)):
        e = rel(a, b); worst = max(worst, e[0]); print(f"T={T} {name}: rel={e[0]:.2e} max={e[1]:.2e}")
    # seam: post with (post_r, comb_r) then pre
    new_r = ref_post(out, res, post_r, comb_r)
    lay2_r, post2_r, comb2_r = ref_pre(new_r, fn, scale, base)
    f_seam = jax.jit(lambda x, r, p, c, f, v: mhc_site(x, r, p, c, f, v, **kw))
    new, lay2, post2, comb2 = f_seam(out, res.reshape(T, -1), post_r, comb_r.reshape(T, -1), fn, vec)
    for name, a, b in (("seam.res", new.reshape(T, H, D), new_r), ("seam.layer", lay2, lay2_r),
                       ("seam.post", post2, post2_r), ("seam.comb", comb2.reshape(T, H, H), comb2_r)):
        e = rel(a, b); worst = max(worst, e[0]); print(f"T={T} {name}: rel={e[0]:.2e} max={e[1]:.2e}")
    # post only
    f_post = jax.jit(lambda x, r, p, c: mhc_site(x, r, p, c, None, None, **kw))
    (new3,) = f_post(out, res.reshape(T, -1), post_r, comb_r.reshape(T, -1))
    e = rel(new3.reshape(T, H, D), new_r); worst = max(worst, e[0]); print(f"T={T} post.res: rel={e[0]:.2e} max={e[1]:.2e}")
    # timing: seam kernel alone vs reference XLA seam
    args = (out, res.reshape(T, -1), post_r, comb_r.reshape(T, -1), fn, vec)
    f_ref = jax.jit(lambda x, r, p, c, f: ref_pre(ref_post(x, r.reshape(T, H, D), p, c.reshape(T, H, H)), f, scale, base))
    for name, f, a in (("kernel seam", f_seam, args), ("xla ref seam", f_ref, args[:5])):
        jax.block_until_ready(f(*a))
        t0 = time.perf_counter()
        for _ in range(200):
            o = f(*a)
        jax.block_until_ready(o)
        print(f"T={T} {name}: {(time.perf_counter() - t0) / 200 * 1e6:.1f} us/call (incl. dispatch)")
    return worst


if __name__ == "__main__":
    worst = max(main(int(t)) for t in (sys.argv[1:] or ["1", "4", "16", "64", "8192"]))
    print("PASS" if worst < 5e-3 else "FAIL", f"worst rel={worst:.2e}")
