"""Old vs new fused_moe_op._fused_moe_jax on EP1 prefill-like inputs."""
import importlib.util, sys, time
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
import sglang_tpu.moe.pallas.fused_moe_op as NEW
spec = importlib.util.spec_from_file_location("fused_moe_op_old", sys.argv[2]); OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
G, M, TOPK, K, N = 64, 2048, 8, 1024, 256
r = iter(jax.random.split(jax.random.PRNGKey(1), 8))
x = (jax.random.normal(next(r), (M, K)) * 0.5).astype(jnp.bfloat16)
w1 = (jax.random.normal(next(r), (G, K, 2 * N)) * 0.03).astype(jnp.bfloat16)
w2 = (jax.random.normal(next(r), (G, N, K)) * 0.03).astype(jnp.bfloat16)
tw, ti = jax.lax.top_k(jax.nn.softmax(jax.random.normal(next(r), (M, G))), TOPK)
tw = (tw / tw.sum(-1, keepdims=True)).astype(jnp.float32); ti = ti.astype(jnp.int32)
for sc in (1, 0):
    res = {}
    for name, mod in (("old", OLD), ("new", NEW)):
        f = jax.jit(lambda x, mod=mod, sc=sc: mod._fused_moe_jax(x, w1, w2, tw, ti, None, None, None, None, TOPK, "silu", use_sc_gather=sc))
        o = jax.block_until_ready(f(x)); n = 10; t0 = time.perf_counter()
        for _ in range(n): o2 = f(x)
        jax.block_until_ready(o2); res[name] = (o, (time.perf_counter() - t0) / n * 1e3)
    print(f"sc={sc}: old {res['old'][1]:.3f} ms new {res['new'][1]:.3f} ms identical={bool(jnp.all(res['old'][0] == res['new'][0]))}", flush=True)
