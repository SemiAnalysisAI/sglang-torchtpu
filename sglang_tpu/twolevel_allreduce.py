# SPDX-License-Identifier: Apache-2.0
"""Two-level one-shot Pallas all-reduce for single-token TP decode tensors.

XLA's all-reduce of a [1, H] bf16 decode activation over a multi-host TP group
is latency bound: on a tpu7x 2x2x4 slice (TP32, 8 TensorCores per host) it took
21 us for H=7168. This kernel does it in two one-shot phases: every rank sends
its whole input to the other ranks of its host and sums what it receives, then
sends that host partial to the rank with the same local index on every other
host and sums again. Each phase sends one DMA per peer: the inputs are packed
as dense (rows, 128) tiles at 16-row aligned offsets of one VMEM buffer, so
several tensors (e.g. Kimi-K3's routed latent and shared-expert output) share
the exchange. The cross-host partial travels as bf16 (half the bytes of f32).
Measured on that slice: 13.0 us at H=7168, 15.3 us at H=10752.

Opt-in via SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE. It applies to the TP group
only, when TP spans the whole world (no DP / PP / extra ranks) so TP ranks are
host-major, for bf16 [1, H] tensors with H a multiple of 128; everything else
keeps the original all-reduce.
"""

import functools
import logging
from threading import Lock

import torch

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

# Largest packed payload (elements) one call reduces.
MAX_ELEMS = 16 * 16384
# Rows of 128 lanes each tensor's block is aligned to in the packed buffer.
_ALIGN_ROWS = 16
# Barrier-semaphore namespace of this kernel. The fused GMM reduce-scatter uses
# collective_id=0; a distinct id keeps the two from sharing a barrier.
_COLLECTIVE_ID = 1
# Tensors per call: 1 (plain all-reduce) or 2 (Kimi-K3 latent + shared).
_ARITIES = (1, 2)

_OPS: dict = {}
_LOCK = Lock()
# Plain values, not dict truthiness: Dynamo traces the eligibility checks.
_READY = False
_WORLD = 0


def _kernel(*refs, n: int, rows: tuple, offs: tuple, world: int, local: int):
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    xs, outs = refs[:n], refs[n : 2 * n]
    packed, gather1, part, gather2, s1, r1, s2, r2 = refs[2 * n :]
    groups = world // local
    me = lax.axis_index("model")
    grp, loc = me // local, lax.rem(me, local)
    local_peers = [grp * local + lax.rem(loc + d, local) for d in range(1, local)]
    remote_peers = [lax.rem(grp + d, groups) * local + loc for d in range(1, groups)]
    mesh_id = pl.DeviceIdType.MESH

    # Every peer must have entered the kernel (its VMEM buffers exist) before
    # anyone writes into them.
    barrier = pltpu.get_barrier_semaphore()
    for p in local_peers + remote_peers:
        pltpu.semaphore_signal(barrier, 1, device_id=(p,), device_id_type=mesh_id)
    if n > 1:
        packed[...] = jnp.zeros_like(packed)  # alignment gaps between tensors
    for x, r, off in zip(xs, rows, offs):
        packed[off : off + r, :] = x[...]
    pltpu.semaphore_wait(barrier, len(local_peers) + len(remote_peers))

    def exchange(src, gather, slot, peers, send_sem, recv_sem):
        copies = []
        for p in peers:
            c = pltpu.make_async_remote_copy(
                src,
                gather.at[slot],
                send_sem,
                recv_sem,
                device_id=(p,),
                device_id_type=mesh_id,
            )
            c.start()
            copies.append(c)
        gather[slot] = src[...]
        for c in copies:
            c.wait_send()
        for c in copies:
            c.wait_recv()

    if local > 1:
        exchange(packed, gather1, loc, local_peers, s1, r1)
        part[...] = jnp.sum(gather1[...].astype(jnp.float32), axis=0).astype(part.dtype)
    else:
        part[...] = packed[...].astype(part.dtype)
    if groups > 1:
        exchange(part, gather2, grp, remote_peers, s2, r2)
        total = jnp.sum(gather2[...].astype(jnp.float32), axis=0)
    else:
        total = part[...].astype(jnp.float32)
    for o, r, off in zip(outs, rows, offs):
        o[...] = total[off : off + r, :].astype(o.dtype)


def twolevel_all_reduce_local(*xs, world: int, local: int):
    """Per-rank body; call inside ``shard_map`` over a 1-D mesh axis "model"
    whose index order is the TP rank order (host-major). Each x is [1, H]."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    n = len(xs)
    groups = world // local
    shapes = tuple(x.shape for x in xs)
    # One token: a [1, H] row as dense (H / 128, 128) tiles; a (1, H) row
    # measured 1-4 us slower to exchange.
    xs = tuple(x.reshape(-1, 128) for x in xs)
    rows = tuple(x.shape[0] for x in xs)
    offs, total = [], 0
    for r in rows:
        offs.append(total)
        total += -(-r // _ALIGN_ROWS) * _ALIGN_ROWS
    total = max(total, _ALIGN_ROWS)
    dtype = xs[0].dtype
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    outs = pl.pallas_call(
        functools.partial(
            _kernel, n=n, rows=rows, offs=tuple(offs), world=world, local=local
        ),
        out_shape=tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in xs),
        in_specs=[vmem] * n,
        out_specs=tuple([vmem] * n),
        scratch_shapes=[
            pltpu.VMEM((total, 128), dtype),
            pltpu.VMEM((local, total, 128), dtype),
            pltpu.VMEM((total, 128), jnp.bfloat16),
            pltpu.VMEM((groups, total, 128), jnp.bfloat16),
        ]
        + [pltpu.SemaphoreType.DMA] * 4,
        compiler_params=pltpu.CompilerParams(collective_id=_COLLECTIVE_ID),
        name=f"twolevel_allreduce{n}_w{world}_l{local}",
    )(*xs)
    outs = tuple(o.reshape(s) for o, s in zip(outs, shapes))
    return outs if n > 1 else outs[0]


def _build(world: int, local: int, n: int):
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("model",), (world,))
    spec = P("model")
    sharded = jax.shard_map(
        functools.partial(twolevel_all_reduce_local, world=world, local=local),
        mesh=mesh,
        in_specs=(spec,) * n,
        out_specs=spec if n == 1 else (spec,) * n,
        check_vma=False,
    )
    if n == 1:

        def _wrapped(x: jax.Array) -> jax.Array:
            return sharded(x)

    else:

        def _wrapped(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
            return sharded(x, y)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::twolevel_allreduce{n}_w{world}_l{local}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(spec,) * n,
        output_partition_specs=(spec,) * n,
    )


# (hosts, ranks per host) the kernel has been measured on and beats XLA's
# all-reduce, bf16 [1, 7168]:
# - tpu7x 2x2x4, TP32 = 4 hosts x 8 ranks: 13.0 vs 21.0 us;
# - one tpu7x host, TP8 = 1 host x 8 ranks (host-local phase only): 10.2 vs
#   15.0 us.
# A one-shot phase with more peers approaches the flat one-shot that measured
# slower than XLA, so other shapes keep XLA's all-reduce until measured.
VALIDATED_TOPOLOGIES = frozenset({(4, 8), (1, 8)})


def local_size(world: int, nnodes: int) -> int:
    """Ranks per host for a host-major TP group of ``world`` ranks, or 0 when
    the two-level split does not apply."""
    local = world // nnodes if nnodes > 0 else 0
    if local <= 0 or world % local:
        return 0
    return local


def _check_ranks_agree(enabled: bool) -> None:
    """Every rank must take the same path: if one host missed the switch, its
    ranks would stay on XLA's all-reduce while the others wait in the Pallas
    barrier, and the first single-token decode would hang. Fail at startup
    instead."""
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return
    seen = [None] * world_group.world_size
    dist.all_gather_object(seen, bool(enabled), group=world_group.cpu_group)
    if len(set(seen)) > 1:
        on = [rank for rank, flag in enumerate(seen) if flag]
        raise RuntimeError(
            "SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE must be set the same on every "
            f"rank; it is on for ranks {on} only."
        )


def register(world: int, local: int) -> None:
    """Build the ops; must run before torch.compile traces the model."""
    global _READY, _WORLD
    with _LOCK:
        for n in _ARITIES:
            if (world, n) not in _OPS:
                _OPS[(world, n)] = _build(world, local, n)
        _WORLD = world
        _READY = True


def register_from_parallel_state() -> None:
    """Called from TpuPlatform.prepare_model_parallel_groups (every rank, after
    the groups exist, before the first collective)."""
    enabled = tpu_envs.SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE.get()
    _check_ranks_agree(enabled)
    if not enabled:
        return
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.runtime_context import get_server_args

    world = get_tensor_model_parallel_world_size()
    if world <= 1 or world != dist.get_world_size():
        logger.info(
            "Two-level Pallas all-reduce off: TP size %d does not span the "
            "world of %d ranks.",
            world,
            dist.get_world_size(),
        )
        return
    server_args = get_server_args()
    local = local_size(world, server_args.nnodes)
    if not local:
        logger.info("Two-level Pallas all-reduce off: no host split for TP %d.", world)
        return
    if (world // local, local) not in VALIDATED_TOPOLOGIES:
        logger.warning(
            "Two-level Pallas all-reduce off: %d hosts x %d ranks is not a "
            "measured topology (%s); XLA's all-reduce stays.",
            world // local,
            local,
            sorted(VALIDATED_TOPOLOGIES),
        )
        return
    register(world, local)
    logger.info(
        "Two-level Pallas all-reduce on: TP %d = %d hosts x %d ranks. It takes "
        "single-token bf16 TP all-reduces (and Kimi-K3's merged MoE reduce); "
        "everything else keeps XLA's all-reduce.",
        world,
        world // local,
        local,
    )
    compile_bs = getattr(server_args, "tpu_compile_bs", None)
    if compile_bs and 1 not in compile_bs:
        logger.warning(
            "Two-level Pallas all-reduce is on but --tpu-compile-bs %s has no "
            "batch size 1, so no compiled decode step takes it.",
            compile_bs,
        )


def eligible(t: torch.Tensor) -> bool:
    # One decode token only: at 4 rows XLA's all-reduce measured faster.
    return (
        t.dim() == 2
        and t.shape[0] == 1
        and t.dtype == torch.bfloat16
        and t.shape[1] % 128 == 0
        and t.numel() <= MAX_ELEMS
    )


def all_reduce_around(original, group, input_: torch.Tensor) -> torch.Tensor:
    """AROUND hook for ``GroupCoordinator.all_reduce``."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    if not (
        _READY
        and group.world_size == _WORLD
        and group is get_tp_group()
        and eligible(input_)
    ):
        return original(group, input_)
    return _OPS[(_WORLD, 1)](input_)


def merged_moe_all_reduce_around(original, latent, shared):
    """AROUND hook for ``kimi_k3.merged_moe_all_reduce``: both TP-partial
    tensors in one packed exchange, with no concat / split copies."""
    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
    )

    if not (
        _READY
        and get_tensor_model_parallel_world_size() == _WORLD
        and eligible(latent)
        and eligible(shared)
        and latent.numel() + shared.numel() <= MAX_ELEMS
    ):
        return original(latent, shared)
    return _OPS[(_WORLD, 2)](latent, shared)
