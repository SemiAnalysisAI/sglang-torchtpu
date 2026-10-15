"""Old vs new mHC site kernel: device time and max output difference, bf16 fn as in the checkpoint."""
import sys, time, types
import jax, jax.numpy as jnp
def load(path, name):
    mod = types.ModuleType(name); exec(compile(open(path).read(), name, "exec"), mod.__dict__); return mod
old, new = load(sys.argv[1], "old"), load(sys.argv[2], "new")
H, D = 4, 4096; MIX = (2 + H) * H
def timeit(f, *a):
    o = jax.block_until_ready(f(*a)); n = 30; t0 = time.perf_counter()
    for _ in range(n): o = f(*a)
    jax.block_until_ready(o); return (time.perf_counter() - t0) / n * 1e3, o
for T in (1, 6, 8192):
    k = iter(jax.random.split(jax.random.PRNGKey(T), 8))
    res = jax.random.normal(next(k), (T, H * D)).astype(jnp.bfloat16)
    x = jax.random.normal(next(k), (T, D)).astype(jnp.bfloat16)
    post = 2 * jax.nn.sigmoid(jax.random.normal(next(k), (T, H)))
    comb = jax.nn.softmax(jax.random.normal(next(k), (T, H * H)), -1)
    fn = (jax.random.normal(next(k), (MIX, H * D)) * 0.02).astype(jnp.bfloat16)
    norm = (1 + 0.1 * jax.random.normal(next(k), (D,))).astype(jnp.float32)
    vec = jnp.zeros((1, old.VEC_WIDTH), jnp.float32).at[0, :MIX].set(0.3 * jax.random.normal(next(k), (MIX,))).at[0, MIX:MIX + 3].set(jnp.array([0.7, 1.3, 2.0]))
    for has_post in (True, False):
        outs = {}
        for name, m in (("old", old), ("new", new)):
            kw = dict(hc=H, iters=20, rms_eps=1e-6, hc_eps=1e-6, norm_weight=norm, norm_eps=1e-6)
            if has_post:
                f = jax.jit(lambda x, res, post, comb, vec, m=m: m.mhc_site(x, res, post, comb, fn, vec, **kw))
                outs[name] = timeit(f, x, res, post, comb, vec)
            else:
                f = jax.jit(lambda res, vec, m=m: m.mhc_site(None, res, None, None, fn, vec, **kw))
                outs[name] = timeit(f, res, vec)
        diffs = []
        for a, b in zip(outs["old"][1], outs["new"][1]):
            a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
            diffs.append(f"{float(jnp.max(jnp.abs(a32 - b32))):.2g}/{float(jnp.mean(a32 != b32)):.1e}")
        print(f"T={T:5d} {'post+pre' if has_post else 'pre     '} old {outs['old'][0]:.3f} ms  new {outs['new'][0]:.3f} ms  maxdiff/frac-differ per output: {diffs}", flush=True)
