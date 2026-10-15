"""Router top-k for one token (288 experts, top-8): XLA chain vs Pallas kernels.

Device time over 43 chained layers (one decode step's worth), plus exactness of
the expert set and weights against the XLA chain.
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np
from jax.experimental import pallas as pl

E, K, LAYERS, SCALE = 288, 8, 43, 2.5
LANES = 384


def xla_router(logits, bias):  # biased_grouped_topk_impl, one group, renormalize, scaling
    scores = jax.nn.sigmoid(logits)
    choice = scores + bias[None]
    _, ids = jax.lax.top_k(choice, K)
    w = jnp.take_along_axis(scores, ids, axis=1)
    w = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-20)
    return w * SCALE, ids.astype(jnp.int32)


def _rank_kernel(logits_ref, bias_ref, w_ref, ids_ref):
    x = logits_ref[...]  # [1, LANES], padded with -inf choice below
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
    scores = jax.nn.sigmoid(x)
    choice = jnp.where(lanes < E, scores + bias_ref[...], -jnp.inf)  # [1, L]
    # Column copy of choice via a diagonal reduction (no transpose needed).
    rows = jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 1)
    sq = jnp.broadcast_to(choice, (LANES, LANES))
    choice_col = jnp.sum(jnp.where(rows == cols, sq, 0.0), axis=1, keepdims=True)  # [L,1] = choice[i]
    # rank[i] = #{j : c_j > c_i or (c_j == c_i and j < i)}  (torch.topk / XLA TopK order)
    beats = (sq > choice_col) | ((sq == choice_col) & (cols < rows))
    rank_col = jnp.sum(beats.astype(jnp.int32), axis=1, keepdims=True)  # [L,1]
    scores_col = jnp.sum(jnp.where(rows == cols, jnp.broadcast_to(scores, (LANES, LANES)), 0.0), axis=1, keepdims=True)
    # slot k takes the expert whose rank is k: one-hot [L, K] reductions over sublanes.
    slots = jax.lax.broadcasted_iota(jnp.int32, (LANES, 128), 1)
    pick = (rank_col == slots) & (slots < K)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, (LANES, 128), 0)
    ids = jnp.sum(jnp.where(pick, row_ids, 0), axis=0, keepdims=True)  # [1, 128]
    w = jnp.sum(jnp.where(pick, scores_col, 0.0), axis=0, keepdims=True)
    w = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-20)
    w_ref[...] = w * SCALE
    ids_ref[...] = ids


def pallas_rank_router(logits, bias):
    x = jnp.pad(logits, ((0, 0), (0, LANES - E)))
    b = jnp.pad(bias.reshape(1, E), ((0, 0), (0, LANES - E)))
    w, ids = pl.pallas_call(
        _rank_kernel,
        out_shape=(jax.ShapeDtypeStruct((1, 128), jnp.float32), jax.ShapeDtypeStruct((1, 128), jnp.int32)),
    )(x, b)
    return w[:, :K], ids[:, :K]


def chain(f):
    @jax.jit
    def run(logits, bias):
        def body(i, acc):
            w, ids = f(logits + acc[0, 0] * 1e-9, bias)
            return acc + w[:, :1] + ids[:, :1].astype(jnp.float32) * 1e-9
        return jax.lax.fori_loop(0, LAYERS, body, jnp.zeros((1, 1), jnp.float32))
    return run


key = jax.random.PRNGKey(0)
bad = 0
for t in range(50):
    k1, k2, key = jax.random.split(key, 3)
    logits = jax.random.normal(k1, (1, E)) * 2
    bias = jax.random.normal(k2, (E,)) * 0.1
    w_a, i_a = jax.jit(xla_router)(logits, bias)
    w_b, i_b = jax.jit(pallas_rank_router)(logits, bias)
    oa, ob = np.argsort(np.asarray(i_a)[0]), np.argsort(np.asarray(i_b)[0])
    same = np.array_equal(np.asarray(i_a)[0][oa], np.asarray(i_b)[0][ob])
    werr = np.max(np.abs(np.asarray(w_a)[0][oa] - np.asarray(w_b)[0][ob]))
    bad += (not same) or werr > 1e-6
print(f"mismatching trials: {bad}/50")
logits = jax.random.normal(key, (1, E)); bias = jnp.zeros((E,))
for name, f in (("xla chain", xla_router), ("pallas rank", pallas_rank_router)):
    r = chain(f); jax.block_until_ready(r(logits, bias)); t0 = time.perf_counter()
    for _ in range(50): o = r(logits, bias)
    jax.block_until_ready(o)
    print(f"{name:12s}: {(time.perf_counter()-t0)/50*1e3:.3f} ms per {LAYERS} layers")
