"""Isolate mHC site cost at prefill size: precision, Sinkhorn iterations, post-only."""
import importlib.util, sys, time, types
import jax, jax.numpy as jnp
SRC = open(sys.argv[1] + "/sglang_tpu/kernels/glm_mhc.py").read()
def load(src, name):
    mod = types.ModuleType(name); exec(compile(src, name, "exec"), mod.__dict__); return mod
H, D = 4, 4096; MIX = (2 + H) * H; T = 8192
k = iter(jax.random.split(jax.random.PRNGKey(0), 8))
res = jax.random.normal(next(k), (T, H * D)).astype(jnp.bfloat16)
x = jax.random.normal(next(k), (T, D)).astype(jnp.bfloat16)
post = jax.nn.sigmoid(jax.random.normal(next(k), (T, H)))
comb = jax.nn.softmax(jax.random.normal(next(k), (T, H * H)), -1)
fn = jax.random.normal(next(k), (MIX, H * D)) * 0.02
norm = jnp.ones((D,), jnp.float32)
def timeit(f, *a):
    o = jax.block_until_ready(f(*a)); n = 30; t0 = time.perf_counter()
    for _ in range(n): o = f(*a)
    jax.block_until_ready(o); return (time.perf_counter() - t0) / n * 1e3, o
base = None
for label, src, iters, post_only in (
    ("baseline HIGHEST iters20", SRC, 20, False),
    ("DEFAULT iters20", SRC.replace("Precision.HIGHEST", "Precision.DEFAULT"), 20, False),
    ("HIGHEST iters1", SRC, 1, False),
    ("post only", SRC, 20, True),
):
    m = load(src, "mhc_" + label.replace(" ", "_")); m._MAX_BLOCK_TOKENS = 256
    vec = jnp.zeros((1, m.VEC_WIDTH), jnp.float32).at[0, MIX:MIX + 3].set(jnp.array([0.7, 1.3, 2.0]))
    kw = dict(hc=H, iters=iters, rms_eps=1e-6, hc_eps=1e-6)
    if post_only:
        f = jax.jit(lambda x, res, post, comb: m.mhc_site(x, res, post, comb, None, None, **kw))
    else:
        f = jax.jit(lambda x, res, post, comb: m.mhc_site(x, res, post, comb, fn, vec, norm_weight=norm, norm_eps=1e-6, **kw))
    dt, o = timeit(f, x, res, post, comb)
    note = ""
    if not post_only and iters == 20:
        if base is None: base = o
        note = " max|diff| vs HIGHEST: " + ", ".join(f"{float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))):.3g}" for a, b in zip(o, base))
    print(f"{label:28s} {dt:.3f} ms/site{note}", flush=True)
