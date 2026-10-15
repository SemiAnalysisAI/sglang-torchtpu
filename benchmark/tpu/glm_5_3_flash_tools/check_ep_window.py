"""Windowed EP MoE (fixed-capacity local row window) vs the full-row EP body."""
import sys, time
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
from sglang_tpu.moe.pallas import fused_moe_op as F
G, E, OFF, M, TOPK, K, N = 64, 8, 16, 1024, 8, 1024, 256
r = iter(jax.random.split(jax.random.PRNGKey(0), 16))
x = (jax.random.normal(next(r), (M, K)) * 0.5).astype(jnp.bfloat16)
w1 = (jax.random.normal(next(r), (E, K, 2 * N)) * 0.03).astype(jnp.bfloat16)
w2 = (jax.random.normal(next(r), (E, N, K)) * 0.03).astype(jnp.bfloat16)
def route(skew):
    logits = jax.random.normal(next(r), (M, G))
    if skew:  # push most tokens onto this rank's experts to overflow the window
        logits = logits.at[:, OFF:OFF + E].add(4.0)
    w, ids = jax.lax.top_k(jax.nn.softmax(logits), TOPK)
    return (w / w.sum(-1, keepdims=True)).astype(jnp.float32), ids.astype(jnp.int32)
print("capacity", F._ep_window_capacity(M * TOPK, E, G, False))
for skew in (False, True):
    tw, ti = route(skew)
    local = int(jnp.sum((ti >= OFF) & (ti < OFF + E)))
    for sc in (1, 0):
        outs = {}
        for name, win in (("window", 0), ("full", -1)):
            f = jax.jit(lambda x, tw, ti, win=win, sc=sc: F._fused_moe_jax(
                x, w1, w2, tw, ti, None, None, None, None, TOPK, "silu",
                global_num_experts=G, expert_group_offset=OFF, use_sc_gather=sc, _ep_window=win))
            o = jax.block_until_ready(f(x, tw, ti)); n = 10; t0 = time.perf_counter()
            for _ in range(n): o2 = f(x, tw, ti)
            jax.block_until_ready(o2); outs[name] = (o, (time.perf_counter() - t0) / n * 1e3)
        a, b = outs["window"][0].astype(jnp.float32), outs["full"][0].astype(jnp.float32)
        print(f"skew={skew} local_rows={local} sc={sc}: window {outs['window'][1]:.3f} ms full {outs['full'][1]:.3f} ms "
              f"maxdiff {float(jnp.max(jnp.abs(a - b))):.3g} identical={bool(jnp.all(a == b))}", flush=True)
