# Local check (CPU, jax 0.10.2): the spike's flat shard_map(decode_local) exports to one TPU custom call.
import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
sys.path.insert(0, "/home/chandrasekhard/tpu-megakernels")
sys.path.insert(0, "/home/chandrasekhard/dagent/merged-t5")
import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

chip = tpu_info.chip_version_from_device_kind("TPU7x")
info = tpu_info.get_tpu_info_for_chip(chip, 1)
pltpu.get_tpu_info = lambda: info
tpu_info.get_tpu_info = lambda: info
tpu_info.get_device_kind = lambda: "TPU7x"
tpu_info.is_tpu_device = lambda: True
tpu_info_for_cpu_export = contextlib.nullcontext
batch, context = int(sys.argv[1]), int(sys.argv[2])
mesh = Mesh(np.array(jax.devices()), ("tp",))
with tpu_info_for_cpu_export():
    from kimi import decode_megakernel as model
    from kimi import load as kimi_load

    weights_abs = kimi_load.abstract_weights(mesh, "/mnt/kimik3-ssd/kimik3", layers=93)
    names = sorted(weights_abs)
    nw = len(names)
    options = {
        "gate_first_kda_projection": True,
        "kernel_options": frozenset({"mla_cache_write_in_attend", "mla_pv_mxu_hilo"}),
    }

    def local(*args):
        weights = {n: a[0] for n, a in zip(names, args[:nw])}
        states = tuple(a[0] for a in args[nw : nw + 4])
        token, pos = args[nw + 4], args[nw + 5]
        logits, new_states = model.decode_local(
            weights, states, token, pos, batch_size=batch, layers=93, **options
        )
        return (logits[None],) + tuple(s[None] for s in new_states)

    in_specs = (P("tp"),) * (nw + 4) + (P(), P())
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"),) * 5, check_vma=False
    )
    sh = NamedSharding(mesh, P("tp"))
    rep = NamedSharding(mesh, P())
    st = [
        jax.ShapeDtypeStruct((32, batch, 69, 3, 3, 3, 128), jnp.bfloat16, sharding=sh),
        jax.ShapeDtypeStruct((32, batch, 69, 3, 128, 128), jnp.float32, sharding=sh),
        jax.ShapeDtypeStruct(
            (32, batch, 24, 3, context, 256), jnp.bfloat16, sharding=sh
        ),
        jax.ShapeDtypeStruct(
            (32, batch, 24, 3, context, 128), jnp.bfloat16, sharding=sh
        ),
    ]
    tok = jax.ShapeDtypeStruct((batch,), jnp.int32, sharding=rep)
    f = jax.jit(sharded, donate_argnums=tuple(range(nw, nw + 4)), keep_unused=True)
    t0 = time.time()
    exp = jax.export.export(f, platforms=["tpu"])(
        *[weights_abs[n] for n in names], *st, tok, tok
    )
    txt = exp.mlir_module()
    print(
        f"exported in {time.time() - t0:.1f}s, custom calls {txt.count('tpu_custom_call')}, in {len(exp.in_avals)} out {len(exp.out_avals)}"
    )
    for a in exp.out_avals:
        print("  out", a)
