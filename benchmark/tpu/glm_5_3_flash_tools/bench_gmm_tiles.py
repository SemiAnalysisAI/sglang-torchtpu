"""gmm_v2 tile sweep at GLM-5.3-Flash EP1 prefill shapes (8192 tokens x top-8, 288 experts, 1/8-width FP8)."""
import sys, time
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
from sglang_tpu.moe.pallas.gmm_v2 import TileSizes, gmm_v2
G, M, K, N = 288, 65536, 4096, 256
r = iter(jax.random.split(jax.random.PRNGKey(0), 8))
ids = jax.random.randint(next(r), (M,), 0, G)
gs = jnp.bincount(ids, length=G).astype(jnp.int32)
x1 = (jax.random.normal(next(r), (M, K)) * 0.5).astype(jnp.bfloat16)
w1 = (jax.random.normal(next(r), (G, K, 2 * N)) * 0.05).astype(jnp.float8_e4m3fn)
s1 = jnp.full((G, 1, 1, 2 * N), 0.01, jnp.float32)
x2 = (jax.random.normal(next(r), (M, N)) * 0.5).astype(jnp.bfloat16)
w2 = (jax.random.normal(next(r), (G, N, K)) * 0.05).astype(jnp.float8_e4m3fn)
s2 = jnp.full((G, 1, 1, K), 0.01, jnp.float32)
def bench(name, lhs, rhs, sc, tiles):
    f = jax.jit(lambda l: gmm_v2(l, rhs, gs, sc, preferred_element_type=jnp.bfloat16,
                                 **({} if tiles is None else {"tile_info": TileSizes(*tiles)})))
    try:
        o = jax.block_until_ready(f(lhs))
    except Exception as e:
        print(f"{name} {tiles}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True); return None
    n = 20; t0 = time.perf_counter()
    for _ in range(n): o2 = f(lhs)
    jax.block_until_ready(o2)
    dt = (time.perf_counter() - t0) / n * 1e3
    print(f"{name} tiles={tiles}: {dt:.3f} ms", flush=True); return o
ref1 = bench("gmm1 [K=4096,N=512]", x1, w1, s1, None)
for t in [(256, 4096, 512), (512, 4096, 512), (1024, 4096, 512), (256, 2048, 512), (512, 2048, 512)]:
    o = bench("gmm1 [K=4096,N=512]", x1, w1, s1, t)
    if o is not None and ref1 is not None: print("   same as default:", bool(jnp.all(o == ref1)))
ref2 = bench("gmm2 [K=256,N=4096]", x2, w2, s2, None)
for t in [(256, 256, 4096), (512, 256, 4096), (1024, 256, 4096), (256, 256, 2048), (512, 256, 2048)]:
    o = bench("gmm2 [K=256,N=4096]", x2, w2, s2, t)
    if o is not None and ref2 is not None: print("   same as default:", bool(jnp.all(o == ref2)))
