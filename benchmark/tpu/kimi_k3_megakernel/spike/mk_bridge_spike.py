# SPDX-License-Identifier: Apache-2.0
"""Bridge spike: run the Inferact Kimi-K3 decode megakernel inside torch_tpu worker processes.

One process per TPU device (torchrun, 4 hosts x 8). The megakernel's shard_map'd decode step is
exported through SGLang's cpu_export_jax_op bridge and called on torch tensors, with synthetic
weights in the megakernel's per-rank layout (random, NaN-free). Prints per-step device time.

Env: MK_SRC (megakernel repo), MK_CKPT (checkpoint dir, for config/shapes), MK_BATCH, MK_CTX,
MK_POS (decode position), MK_STEPS.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '1')}",
)

import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ["MK_SRC"])


def log(rank, *msg):
    if rank == 0:
        print("[spike]", *msg, flush=True)


KERNEL_OPTIONS = frozenset({"mla_cache_write_in_attend", "mla_pv_mxu_hilo"})


def build_decode_op(mesh, batch, context, layers=93):
    """Register the megakernel decode step as a torch custom op through SGLang's bridge.

    The bridge needs explicit jax.Array-annotated parameters, so the flat entry point
    is generated with one named parameter per input (weights, 4 states, token, position).
    """
    import jax
    import torch
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op, tpu_info_for_cpu_export

    with tpu_info_for_cpu_export():
        from kimi import decode_megakernel as model
        from kimi import load as kimi_load

        weights_abs = kimi_load.abstract_weights(
            mesh, os.environ["MK_CKPT"], layers=layers
        )
    names = sorted(weights_abs)
    nw = len(names)
    kda_layers, mla_layers = 69, 24
    state_specs = (
        ((batch, kda_layers, 3, 3, 3, 128), torch.bfloat16),
        ((batch, kda_layers, 3, 128, 128), torch.float32),
        ((batch, mla_layers, 3, context, 256), torch.bfloat16),
        ((batch, mla_layers, 3, context, 128), torch.bfloat16),
    )

    def local(*args):
        weights = {n: a[0] for n, a in zip(names, args[:nw])}
        states = tuple(a[0] for a in args[nw : nw + 4])
        token, pos = args[nw + 4], args[nw + 5]
        logits, new_states = model.decode_local(
            weights,
            states,
            token,
            pos,
            batch_size=batch,
            layers=layers,
            gate_first_kda_projection=True,
            kernel_options=KERNEL_OPTIONS,
        )
        return (logits[None],) + tuple(s[None] for s in new_states)

    in_specs = (P("tp"),) * (nw + 4) + (P(), P())
    out_specs = (P("tp"),) * 5
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )

    params = [f"a{i}" for i in range(nw + 6)]
    src = (
        "def flat(" + ", ".join(f"{p}: jax.Array" for p in params) + ") -> "
        "tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:\n"
        "    return sharded(" + ", ".join(params) + ")\n"
    )
    scope = {"jax": jax, "sharded": sharded}
    exec(src, scope)
    op = cpu_export_jax_op(
        "mkspike::kimi_k3_decode_b%d_c%d" % (batch, context),
        scope["flat"],
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
        donate_argnums=tuple(range(nw, nw + 4)),
    )
    return op, names, weights_abs, state_specs


def main():
    dist.init_process_group(backend="tpu_dist")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 32, world
    batch = int(os.environ.get("MK_BATCH", "1"))
    context = int(os.environ.get("MK_CTX", "1152"))
    position = int(os.environ.get("MK_POS", "128"))
    steps = int(os.environ.get("MK_STEPS", "30"))

    import jax
    from torch_tpu._internal.distributed import tpu_distributed
    from torch_tpu._internal.pallas import get_pallas_mesh

    log(
        rank,
        "jax",
        jax.__version__,
        "world",
        world,
        "global device id",
        tpu_distributed.global_device_id(),
    )
    mesh = get_pallas_mesh(("tp",), (world,))
    log(rank, "mesh devices", [int(d.id) for d in mesh.devices.flat])

    op, names, weights_abs, state_specs = build_decode_op(mesh, batch, context)
    from kimi import (
        decode_megakernel as model,  # imported under the TPU-info context above
    )

    nw = len(names)
    log(
        rank,
        len(names),
        "weight arrays:",
        {n: (tuple(weights_abs[n].shape), str(weights_abs[n].dtype)) for n in names},
    )

    device = torch.device("tpu")
    tdt = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "uint32": torch.uint32,
        "uint8": torch.uint8,
        "int32": torch.int32,
    }
    gen = torch.Generator().manual_seed(1234 + rank)
    weights = []
    t0 = time.time()
    for n in names:
        shape = (1,) + tuple(weights_abs[n].shape[1:])
        dt = tdt[str(weights_abs[n].dtype)]
        t = torch.empty(shape, dtype=dt, device=device)
        # One random host block per weight family, copied into every layer slot: timing only
        # needs realistic magnitudes and routing spread, not distinct layers.
        per_layer = n in model.NAMES
        sub = shape[2:] if per_layer else shape[1:]
        if True:
            if dt in (torch.bfloat16, torch.float32):
                if n.endswith("norm") or n in (
                    "attn_res",
                    "mlp_res",
                    "output_res",
                    "final_norm",
                ):
                    host = torch.ones(sub, dtype=dt)
                elif n == "router_bias":
                    host = torch.zeros(sub, dtype=dt)
                else:
                    host = (torch.randn(sub, generator=gen) * 0.02).to(dt)
            elif dt == torch.float8_e4m3fn:
                host = (
                    (
                        torch.randint(0, 256, sub, generator=gen, dtype=torch.int32)
                        & 0xBF
                    )
                    .to(torch.uint8)
                    .view(torch.float8_e4m3fn)
                )
            elif dt == torch.uint8:
                host = torch.full(sub, 120, dtype=torch.uint8)  # E8M0 scale 2**-7
            elif dt == torch.uint32:
                host = torch.randint(
                    -(2**31), 2**31 - 1, sub, generator=gen, dtype=torch.int32
                ).view(torch.uint32)
            elif dt == torch.int32:
                host = torch.zeros(sub, dtype=torch.int32)
            else:
                raise TypeError(dt)
            block = host.to(device)
            if per_layer:
                for i in range(shape[1]):
                    t[0, i].copy_(block)
            else:
                t[0].copy_(block)
            del block
        weights.append(t)
    log(rank, f"weights on device in {time.time() - t0:.0f}s")
    states = [torch.zeros((1,) + s, dtype=d, device=device) for s, d in state_specs]
    token = torch.randint(
        1000,
        100000,
        (batch,),
        dtype=torch.int32,
        generator=torch.Generator().manual_seed(7),
    ).to(device)
    pos = torch.full((batch,), position, dtype=torch.int32, device=device)

    def step(*args):
        return op(*args)

    compiled = torch.compile(step, fullgraph=True, dynamic=False)
    t0 = time.time()
    out = compiled(*weights, *states, token, pos)
    torch.tpu.synchronize() if hasattr(torch, "tpu") and hasattr(
        torch.tpu, "synchronize"
    ) else out[0].cpu()
    log(
        rank,
        f"first call (compile + run) {time.time() - t0:.1f}s; logits {tuple(out[0].shape)} {out[0].dtype}",
    )
    states = list(out[1:])
    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        out = compiled(*weights, *states, token, pos + i + 1)
        _ = out[0][0, 0, :4].cpu()
        times.append(time.perf_counter() - t0)
        states = list(out[1:])
    times.sort()
    log(
        rank,
        f"batch={batch} context={context} position={position}: median {1e3 * times[len(times) // 2]:.2f} ms, "
        f"min {1e3 * times[0]:.2f} ms, max {1e3 * times[-1]:.2f} ms over {steps} steps",
    )
    log(rank, "logits finite:", bool(torch.isfinite(out[0].float()).all().cpu()))
    print(f"[spike] SPIKE_DONE rank={rank}", flush=True)


if __name__ == "__main__":
    main()
