"""Microbenchmark: XLA psum vs a one-shot Pallas all-reduce at decode/verify shapes.

usage: python bench_allreduce.py [--tokens 1 6 8 16] [--iters 2000]
Runs in one process over all local TPU devices (TP8 on a 2x2x1 v7x host), the same
mesh the server's TP group uses. Each rank holds a [T, 4096] bf16 activation; both
variants return the sum on every rank. Reports us per all-reduce and the max abs
difference between the two results.
"""
import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

HIDDEN = 4096


def _one_shot_kernel(x_ref, out_ref, send_sems, recv_sems, buf, *, world):
    me = lax.axis_index("model").astype(jnp.int32)
    buf[...] = x_ref[...]
    copies = []
    for i in range(world):
        # Rank `me` writes its rows into slot `me` of rank `peer`'s gather buffer.
        peer = jnp.mod(me + i, world)
        copies.append(
            pltpu.make_async_remote_copy(
                buf,
                out_ref.at[me],
                send_sems.at[i],
                recv_sems.at[i],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
        )
    for c in copies:
        c.start()
    for c in copies:
        c.wait()


def one_shot_all_reduce(x, world):
    tokens, hidden = x.shape
    gathered = pl.pallas_call(
        functools.partial(_one_shot_kernel, world=world),
        out_shape=jax.ShapeDtypeStruct((world, tokens, hidden), x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.SemaphoreType.DMA((world,)),
            pltpu.SemaphoreType.DMA((world,)),
            pltpu.VMEM((tokens, hidden), x.dtype),
        ],
        name=f"one_shot_all_reduce_{world}",
    )(x)
    return jnp.sum(gathered.astype(jnp.float32), axis=0).astype(x.dtype)


def timed(fn, x, iters, reps=5):
    """Device time per all-reduce: `iters` chained calls inside one jitted loop,
    best of `reps`, so host dispatch is amortized away."""
    fn(x).block_until_ready()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(x).block_until_ready()
        best = min(best, time.perf_counter() - t0)
    return best / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 6, 8, 16])
    ap.add_argument("--iters", type=int, default=500)
    a = ap.parse_args()
    devices = jax.devices()
    world = len(devices)
    mesh = Mesh(np.array(devices), ("model",))
    spec = NamedSharding(mesh, P("model", None))
    print(f"{world} devices: {devices[0].device_kind}")

    def psum_local(x):
        return lax.psum(x, "model")

    def one_shot_local(x):
        return one_shot_all_reduce(x, world)

    for t in a.tokens:
        shard = functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=P("model", None),
            out_specs=P("model", None),
            check_vma=False,
        )
        def looped(step):
            # Scale by 1/world so chained iterations stay bounded.
            body = lambda _, y: step(y) * (1.0 / world)
            return jax.jit(shard(lambda y: lax.fori_loop(0, a.iters, body, y)))

        xla_once = jax.jit(shard(psum_local))
        mine_once = jax.jit(shard(one_shot_local))
        xla, mine = looped(psum_local), looped(one_shot_local)
        x = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(t), (world * t, HIDDEN)).astype(
                jnp.bfloat16
            ),
            spec,
        )
        diff = float(
            jnp.max(
                jnp.abs(
                    xla_once(x).astype(jnp.float32) - mine_once(x).astype(jnp.float32)
                )
            )
        )
        us_xla = timed(xla, x, a.iters)
        us_mine = timed(mine, x, a.iters)
        print(
            f"T={t:3d} ({t * HIDDEN * 2 // 1024} KiB/rank): XLA psum {us_xla:7.2f} us, "
            f"one-shot {us_mine:7.2f} us, max |diff| {diff:.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
