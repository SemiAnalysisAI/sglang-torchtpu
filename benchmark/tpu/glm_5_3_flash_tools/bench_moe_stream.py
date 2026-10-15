"""Prototype: one-token MoE that streams the k selected experts (tensor-parallel layout).

Per rank (TP8 / EP1): all 288 experts at 1/8 width (intermediate 256). Grid over the
k=8 slots with the expert ids scalar-prefetched; each step's weight blocks are
w13[ids[s]] / w2[ids[s]], so the pipeline prefetches the next expert's weights while
the current one computes. Output: the weighted sum over slots, f32 accumulated.

Compares against an XLA reference with the same math and reports device time for
43 chained layers (one decode step) plus the HBM floor.
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

H, N, E, K, LAYERS, LIMIT = 4096, 256, 288, 8, 43, 10.0
F8 = jnp.float8_e4m3fn; F8MAX = float(jnp.finfo(F8).max)


def qdot(x, w, ws):  # per-row FP8 quantize, FP8 dot, rescale (apply_tpu_fp8_linear math)
    xs = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True) / F8MAX, 1e-12)
    acc = jax.lax.dot_general((x / xs).astype(F8), w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    return acc * xs * ws


def expert(x32, w13, s13, w2, s2):
    gu = qdot(x32, w13, s13).astype(jnp.bfloat16)
    g, u = gu[:, :N], gu[:, N:]
    g = jnp.minimum(g, LIMIT); u = jnp.clip(u, -LIMIT, LIMIT)
    g32 = g.astype(jnp.float32)
    h = ((g32 * jax.nn.sigmoid(g32)).astype(jnp.bfloat16).astype(jnp.float32) * u.astype(jnp.float32)).astype(jnp.bfloat16)
    return qdot(h.astype(jnp.float32), w2, s2)  # f32 [1, H]


def _kernel(ids_ref, wts_ref, x_ref, w13_ref, s13_ref, w2_ref, s2_ref, o_ref, acc_ref):
    s = pl.program_id(0)

    @pl.when(s == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    y = expert(x_ref[...].astype(jnp.float32), w13_ref[0], s13_ref[0], w2_ref[0], s2_ref[0])
    acc_ref[...] += wts_ref[s] * y

    @pl.when(s == pl.num_programs(0) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@jax.jit
def stream_moe(x, ids, wts, w13, s13, w2, s2):
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(K,),
        in_specs=[
            pl.BlockSpec((1, H), lambda s, ids, wts: (0, 0)),
            pl.BlockSpec((1, H, 2 * N), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, 1, 2 * N), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, N, H), lambda s, ids, wts: (ids[s], 0, 0)),
            pl.BlockSpec((1, 1, H), lambda s, ids, wts: (ids[s], 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, H), lambda s, ids, wts: (0, 0)),
        scratch_shapes=[pltpu.VMEM((1, H), jnp.float32)],
    )
    return pl.pallas_call(
        _kernel, grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((1, H), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="moe_stream_bs1",
    )(ids, wts, x, w13, s13, w2, s2)


@jax.jit
def ref_moe(x, ids, wts, w13, s13, w2, s2):
    out = jnp.zeros((1, H), jnp.float32)
    for k in range(K):
        e = ids[k]
        out = out + wts[k] * expert(x.astype(jnp.float32), w13[e], s13[e], w2[e], s2[e])
    return out.astype(jnp.bfloat16)


key = iter(jax.random.split(jax.random.PRNGKey(0), 16))
w13 = (jax.random.normal(next(key), (E, H, 2 * N)) * 0.5).astype(F8)
w2 = (jax.random.normal(next(key), (E, N, H)) * 0.5).astype(F8)
s13 = jax.random.uniform(next(key), (E, 1, 2 * N), minval=1e-3, maxval=1e-2)
s2 = jax.random.uniform(next(key), (E, 1, H), minval=1e-3, maxval=1e-2)
x = (jax.random.normal(next(key), (1, H)) * 2).astype(jnp.bfloat16)
ids = jax.random.permutation(next(key), E)[:K].astype(jnp.int32)
wts = jax.nn.softmax(jax.random.normal(next(key), (K,))).astype(jnp.float32)

a = np.asarray(stream_moe(x, ids, wts, w13, s13, w2, s2), np.float32)
b = np.asarray(ref_moe(x, ids, wts, w13, s13, w2, s2), np.float32)
print(f"rel err vs XLA reference: {np.linalg.norm(a-b)/np.linalg.norm(b):.2e}, bit-exact {np.mean(a==b)*100:.1f}%")


def chain(f):
    @jax.jit
    def run(x, ids):
        def body(i, h):
            nid = (ids + i * 7) % E
            return f(h, nid, wts, w13, s13, w2, s2)
        return jax.lax.fori_loop(0, LAYERS, body, x)
    return run


for name, f in (("stream kernel", stream_moe), ("xla per-expert", ref_moe)):
    r = chain(f); jax.block_until_ready(r(x, ids)); t0 = time.perf_counter()
    for _ in range(20): o = r(x, ids)
    jax.block_until_ready(o)
    dt = (time.perf_counter() - t0) / 20 * 1e3
    print(f"{name:15s}: {dt:.3f} ms per {LAYERS} layers = {dt/LAYERS*1e3:.1f} us/layer")
floor = K * (H * 2 * N + N * H) / 3.35e12 * 1e6
print(f"HBM floor (3.35 TB/s): {floor:.1f} us/layer, {floor*LAYERS/1e3:.3f} ms per step")
