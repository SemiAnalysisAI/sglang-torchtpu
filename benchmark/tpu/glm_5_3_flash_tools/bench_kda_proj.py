"""Device time of 34 KDA projection blocks (one decode step) under different weight layouts."""
import time, jax, jax.numpy as jnp
H, K, R, NB = 4096, 3072, 128, 8   # hidden, qkv out, low rank, beta heads (TP8 per rank)
L = 34
key = iter(jax.random.split(jax.random.PRNGKey(0), 32))
rn = lambda *s: (jax.random.normal(next(key), s) * 0.02).astype(jnp.bfloat16)
x = rn(1, H)
Wqkv, Wb, Wfa, Wga = rn(L, K, H), rn(L, NB, H), rn(L, R, H), rn(L, R, H)
Wfb, Wgb = rn(L, 1024, R), rn(L, 1024, R)
lin = lambda x, w: x @ w.T
def separate(i, x):
    q = lin(x, Wqkv[i]); b = lin(x, Wb[i]); f = lin(lin(x, Wfa[i]), Wfb[i]); g = lin(lin(x, Wga[i]), Wgb[i])
    return q, b, f, g
Wcat = jnp.concatenate([Wqkv, Wb, Wfa, Wga], axis=1)                      # 3336 rows, unaligned splits
Wup = jnp.stack([jnp.swapaxes(Wfb, 1, 2), jnp.swapaxes(Wgb, 1, 2)], axis=1)  # [L, 2, R, 1024]
def fused_now(i, x):
    o = lin(x, Wcat[i]); q, b, fa, ga = jnp.split(o, [K, K + NB, K + NB + R], axis=-1)
    f, g = jnp.matmul(jnp.stack([fa, ga]), Wup[i])
    return q, b, f, g
Wal = jnp.concatenate([Wqkv, Wfa, Wga, jnp.pad(Wb, ((0, 0), (0, 128 - NB), (0, 0)))], axis=1)  # 3456, 128-aligned
def fused_aligned(i, x):
    o = lin(x, Wal[i]); q = o[:, :K]; fa = o[:, K:K + R]; ga = o[:, K + R:K + 2 * R]; b = o[:, K + 2 * R:K + 2 * R + NB]
    return q, b, lin(fa, Wfb[i]), lin(ga, Wgb[i])
def run(f):
    @jax.jit
    def step(x):
        def body(i, acc):
            q, b, fo, g = f(i, x)
            return acc + q[:, :1].astype(jnp.float32) + b[:, :1] + fo[:, :1] + g[:, :1]
        return jax.lax.fori_loop(0, L, body, jnp.zeros((1, 1), jnp.float32))
    return step
for name, f in (("separate", separate), ("fused (current)", fused_now), ("fused aligned", fused_aligned)):
    s = run(f); jax.block_until_ready(s(x)); t0 = time.perf_counter()
    for _ in range(50): o = s(x)
    jax.block_until_ready(o)
    print(f"{name:16s}: {(time.perf_counter()-t0)/50*1e3:.3f} ms per 34 layers", flush=True)
