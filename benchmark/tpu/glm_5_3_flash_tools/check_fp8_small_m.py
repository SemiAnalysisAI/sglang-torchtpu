"""Small-M W8A8 Pallas linear vs the apply_tpu_fp8_linear math (XLA), at GLM-5.3-Flash TP8 shapes."""
import importlib.util, os, sys, time
import jax, jax.numpy as jnp, numpy as np

_spec = importlib.util.spec_from_file_location("fp8sm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "python/sglang/srt/hardware_backend/tpu/kernels/fp8_linear_small_m.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
F8 = jnp.float8_e4m3fn; F8MAX = float(jnp.finfo(F8).max)

def ref(x, w, ws):  # mirrors fp8_linear.py: pad to 16, row quantize, f32 dot, rescale, cast
    m = x.shape[0]; pad = (16 - m % 16) % 16
    xp = jnp.pad(x, ((0, pad), (0, 0)))
    xs = jnp.maximum(jnp.max(jnp.abs(xp), axis=1, keepdims=True).astype(jnp.float32) / F8MAX, 1e-12)
    xq = (xp.astype(jnp.float32) / xs).astype(F8)
    out = jnp.dot(xq, w, preferred_element_type=jnp.float32) * xs * ws.reshape(1, -1)
    return out[:m].astype(x.dtype)

SHAPES = [(4096, 512), (256, 4096), (4096, 2048), (2048, 4096), (1536, 2048), (4096, 3072), (1536, 4096)]
worst = 0.0
for m in (1, 4, 16):
    for k, n in SHAPES:
        key = jax.random.PRNGKey(k * 7 + n + m)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (m, k)).astype(jnp.bfloat16)
        w = (jax.random.normal(k2, (k, n)) * 0.5).astype(F8)
        ws = jax.random.uniform(k3, (n,), minval=0.001, maxval=0.01)
        a = np.asarray(jax.jit(_m.fp8_linear_small_m)(x, w, ws), np.float32)
        b = np.asarray(jax.jit(ref)(x, w, ws), np.float32)
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)); worst = max(worst, rel)
        exact = float(np.mean(a == b))
        print(f"m={m:2d} k={k} n={n}: rel={rel:.2e} bit-exact={exact*100:.1f}%", flush=True)

# Device time of 125 chained decode linears (one Flash decode step's worth).
x = jax.random.normal(jax.random.PRNGKey(1), (1, 4096)).astype(jnp.bfloat16)
w1 = (jax.random.normal(jax.random.PRNGKey(2), (4096, 4096)) * 0.5).astype(F8)
ws1 = jnp.full((4096,), 0.003, jnp.float32)
def chain(f):
    @jax.jit
    def run(x):
        return jax.lax.fori_loop(0, 125, lambda i, h: (f(h, w1, ws1) * 0.01).astype(jnp.bfloat16), x)
    return run
for name, f in (("kernel", _m.fp8_linear_small_m), ("xla ref", ref)):
    r = chain(f); jax.block_until_ready(r(x)); t0 = time.perf_counter()
    for _ in range(20): o = r(x)
    jax.block_until_ready(o)
    print(f"{name}: {(time.perf_counter()-t0)/20*1e3:.3f} ms per 125 linears (4096x4096)", flush=True)
print("PASS" if worst < 1e-2 else "FAIL", f"worst rel={worst:.2e}")
