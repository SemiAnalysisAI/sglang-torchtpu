"""Streaming one-token MoE vs the production fused-MoE route-meta path (EP=1 shapes)."""
import os, time
os.environ.setdefault("SGLANG_TPU_FP8_MOE_FP32_SCALES", "1")
import jax, jax.numpy as jnp, numpy as np
from sglang.srt.layers.moe.pallas.fused_moe_op import _fused_moe_jax
from sglang.srt.hardware_backend.tpu.kernels.moe_stream_bs1 import moe_stream_bs1

H, N, E, K, LIMIT, LAYERS = 4096, 256, 288, 8, 10.0, 43
F8 = jnp.float8_e4m3fn
key = iter(jax.random.split(jax.random.PRNGKey(0), 64))
w1 = (jax.random.normal(next(key), (E, H, 2 * N)) * 0.5).astype(F8)
w2 = (jax.random.normal(next(key), (E, N, H)) * 0.5).astype(F8)
s1 = jax.random.uniform(next(key), (E, 1, 1, 2 * N), minval=1e-3, maxval=1e-2)
s2 = jax.random.uniform(next(key), (E, 1, 1, H), minval=1e-3, maxval=1e-2)

def prod(x, ids, wts):
    return _fused_moe_jax(x, w1, w2, wts, ids, s1, s2, None, None, topk=K,
                          activation_name=f"silu_clamped:{LIMIT}", permute_dense_threshold=512)

def stream(x, ids, wts):
    return moe_stream_bs1(x, ids, wts, w1, s1, w2, s2, LIMIT)

worst, exact = 0.0, []
for t in range(10):
    x = (jax.random.normal(next(key), (1, H)) * 3).astype(jnp.bfloat16)
    ids = jax.random.permutation(next(key), E)[:K].astype(jnp.int32).reshape(1, K)
    wts = jax.nn.softmax(jax.random.normal(next(key), (1, K))).astype(jnp.bfloat16)
    a = np.asarray(jax.jit(stream)(x, ids, wts), np.float32)
    b = np.asarray(jax.jit(prod)(x, ids, wts), np.float32)
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b)); worst = max(worst, rel)
    exact.append(float(np.mean(a == b)))
print(f"vs production: worst rel {worst:.2e}, bit-exact lanes {np.mean(exact)*100:.1f}%")

def chain(f):
    @jax.jit
    def run(x, ids, wts):
        def body(i, h):
            return f(h, (ids + i * 7) % E, wts)
        return jax.lax.fori_loop(0, LAYERS, body, x)
    return run

for name, f in (("production route-meta + gmm", prod), ("streaming kernel", stream)):
    r = chain(f); jax.block_until_ready(r(x, ids, wts)); t0 = time.perf_counter()
    for _ in range(20): o = r(x, ids, wts)
    jax.block_until_ready(o)
    dt = (time.perf_counter() - t0) / 20 * 1e3
    print(f"{name:28s}: {dt/LAYERS*1e3:6.1f} us/layer")
