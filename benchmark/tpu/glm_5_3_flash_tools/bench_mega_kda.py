"""sglang-jax mega KDA prefill kernel at GLM-5.3-Flash per-rank shapes (8 heads, K=V=128, bound -5)."""
import sys, time, types
import jax, jax.numpy as jnp
REF = sys.argv[1]
kern = types.ModuleType("mega_kernel"); exec(compile(open(REF + "/kernel.py").read(), "mega_kernel", "exec"), kern.__dict__)
sys.modules["sgl_jax.srt.kernels.kda.mega_kda.kernel"] = kern
init = types.ModuleType("mega_init"); exec(compile(open(REF + "/__init__.py").read(), "mega_init", "exec"), init.__dict__)
H, K, V = 8, 128, 128
for T in [int(t) for t in (sys.argv[2:] or ["8192", "4096", "1024"])]:
    r = iter(jax.random.split(jax.random.PRNGKey(T), 8))
    q = jax.random.normal(next(r), (1, T, H, K)).astype(jnp.bfloat16)
    k = jax.random.normal(next(r), (1, T, H, K)).astype(jnp.bfloat16)
    v = jax.random.normal(next(r), (1, T, H, V)).astype(jnp.bfloat16)
    g = jax.random.normal(next(r), (1, T, H, K)).astype(jnp.bfloat16)
    beta = jax.nn.sigmoid(jax.random.normal(next(r), (1, T, H))).astype(jnp.bfloat16)
    A_log = jax.random.normal(next(r), (H,)) * 0.5
    dt_bias = jax.random.normal(next(r), (H * K,)) * 0.1
    state = jnp.zeros((1, H, K, V), jnp.float32)
    cu = jnp.array([0, T], jnp.int32)
    f = jax.jit(lambda q, k, v, g, beta, s: init.kda_forward_packed(q, k, v, g, beta, cu_seqlens=cu, A_log=A_log, dt_bias=dt_bias, scale=K ** -0.5, initial_state=s, lower_bound=-5.0))
    try:
        o = jax.block_until_ready(f(q, k, v, g, beta, state))
    except Exception as e:
        print(f"T={T}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True); continue
    n = 30; t0 = time.perf_counter()
    for _ in range(n): o = f(q, k, v, g, beta, state)
    jax.block_until_ready(o)
    print(f"T={T} H={H}: mega KDA {(time.perf_counter() - t0) / n * 1e3:.3f} ms per layer, out finite={bool(jnp.all(jnp.isfinite(o[0].astype(jnp.float32))))}", flush=True)
