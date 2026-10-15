"""All-reduce of [T, 4096] bf16 across the 8 local TPU devices at prefill sizes: psum vs reduce-scatter + all-gather."""
import functools, sys, time
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
devs = jax.devices(); W = len(devs)
mesh = Mesh(devs, ("model",))
print("devices", W, devs[0].device_kind, flush=True)
def timeit(f, x, n=20):
    o = jax.block_until_ready(f(x)); t0 = time.perf_counter()
    for _ in range(n): o = f(x)
    jax.block_until_ready(o); return (time.perf_counter() - t0) / n * 1e3
for T in [int(t) for t in (sys.argv[1:] or ["1024", "2048", "4096", "8192"])]:
    x = jax.device_put(jnp.ones((W * T, 4096), jnp.bfloat16), NamedSharding(mesh, P("model", None)))
    psum = jax.jit(shard_map(lambda a: jax.lax.psum(a, "model"), mesh, in_specs=P("model", None), out_specs=P("model", None)))
    rsag = jax.jit(shard_map(lambda a: jax.lax.all_gather(jax.lax.psum_scatter(a, "model", scatter_dimension=0, tiled=True), "model", axis=0, tiled=True),
                             mesh, in_specs=P("model", None), out_specs=P("model", None)))
    mb = T * 4096 * 2 / 1e6
    for name, f in (("psum", psum), ("rs+ag", rsag)):
        ms = timeit(f, x)
        # ring algorithmic bytes per rank: 2 * (W-1)/W * size
        print(f"T={T:5d} ({mb:.0f} MB) {name:6s}: {ms*1e3:8.1f} us, {2*(W-1)/W*mb/ms:7.1f} GB/s per rank (ring algbw)", flush=True)
