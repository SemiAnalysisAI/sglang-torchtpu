"""Where does the mHC seam kernel's ~10.6 us go? Device time per call at decode shapes.

usage: python bench_mhc_seam.py [--tokens 8] [--iters 500]
Times sglang_tpu.kernels.glm_mhc.mhc_site (post + pre + norm, GLM-5.3-Flash shapes)
with 20 Sinkhorn rounds, 1 round, and 0 extra rounds, each chained inside one jitted
loop on a single device, so the difference isolates the Sinkhorn chain.
"""
import argparse
import time

import jax
import jax.numpy as jnp
from jax import lax

from sglang_tpu.kernels.glm_mhc import VEC_WIDTH, mhc_site

HC, D = 4, 4096
MIX = (2 + HC) * HC


def timed(fn, args, iters, reps=5):
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--iters", type=int, default=500)
    a = ap.parse_args()
    keys = iter(jax.random.split(jax.random.PRNGKey(0), 8))
    fn = jax.random.normal(next(keys), (MIX, HC * D), jnp.float32) * 0.01
    vec = jax.random.normal(next(keys), (1, VEC_WIDTH), jnp.float32) * 0.1
    norm_w = jnp.ones((D,), jnp.float32)
    for t in a.tokens:
        x = jax.random.normal(next(keys), (t, D)).astype(jnp.bfloat16)
        res = jax.random.normal(next(keys), (t, HC * D)).astype(jnp.bfloat16)
        post = jax.nn.sigmoid(jax.random.normal(next(keys), (t, HC)))
        comb = jax.nn.softmax(jax.random.normal(next(keys), (t, HC * HC)), axis=-1)

        def run(sinkhorn):
            def body(_, carry):
                xx, rr, pp, cc = carry
                new_res, layer, p2, c2 = mhc_site(
                    xx, rr, pp, cc, fn, vec, hc=HC, iters=sinkhorn,
                    rms_eps=1e-6, hc_eps=1e-6, norm_weight=norm_w, norm_eps=1e-6,
                )
                return layer, new_res, p2, c2

            return jax.jit(
                lambda *c: lax.fori_loop(0, a.iters, body, c)
            )

        args = (x, res, post, comb)
        r20 = timed(run(20), args, a.iters)
        r1 = timed(run(1), args, a.iters)
        print(
            f"T={t}: seam with 20 Sinkhorn rounds {r20:6.2f} us, 1 round {r1:6.2f} us "
            f"-> Sinkhorn ~{r20 - r1:5.2f} us",
            flush=True,
        )


if __name__ == "__main__":
    main()
