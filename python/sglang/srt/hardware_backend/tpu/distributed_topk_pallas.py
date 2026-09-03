# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op wrapper for the distributed TPU TopK Pallas kernel."""

import functools
from collections.abc import Callable
from threading import Lock

import torch

from sglang.srt.hardware_backend.tpu.kernels.distributed_topk import (
    SUPPORTED_WORLD_SIZES,
)
from sglang.srt.hardware_backend.tpu.kernels.distributed_topk import (
    distributed_top_k as distributed_top_k_kernel,
)

_OPS: dict[int, Callable] = {}
_OPS_LOCK = Lock()


@functools.cache
def _get_mesh(world_size: int):
    import numpy as np
    from jax.sharding import Mesh

    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_torchtpu_export_devices,
    )

    return Mesh(np.asarray(get_torchtpu_export_devices(world_size)), ("model",))


def _build_op(world_size: int) -> Callable:
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)

    def _local(
        local_logits: jax.Array,
        rank_input: jax.Array,
        org_vocab_size: int,
        global_vocab_size: int,
        org_vocab_padded_count: int,
        k: int,
    ) -> tuple[jax.Array, jax.Array]:
        return distributed_top_k_kernel(
            local_logits,
            rank_input,
            world_size=world_size,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
            k=k,
        )

    sharded = jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, "model"), P("model"), None, None, None, None),
        out_specs=(P(), P()),
        check_vma=False,
    )

    def _wrapped(
        local_logits: jax.Array,
        rank_input: jax.Array,
        org_vocab_size: int,
        global_vocab_size: int,
        org_vocab_padded_count: int,
        k: int,
    ) -> tuple[jax.Array, jax.Array]:
        return sharded(
            local_logits,
            rank_input,
            org_vocab_size,
            global_vocab_size,
            org_vocab_padded_count,
            k,
        )

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::distributed_top_k_tp{world_size}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(
            P(None, "model"),
            P("model"),
            None,
            None,
            None,
            None,
        ),
        output_partition_specs=(P(), P()),
    )


def get_pallas_distributed_top_k_op(world_size: int) -> Callable:
    """Return the lazily registered custom op for one supported topology."""

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Pallas distributed TopK does not support TP{world_size}")
    op = _OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(world_size)
        if op is None:
            op = _build_op(world_size)
            _OPS[world_size] = op
    return op


def register_pallas_distributed_top_k_op(world_size: int) -> None:
    """Register the current TP8/TP16 topology before full-graph tracing."""

    if world_size in SUPPORTED_WORLD_SIZES:
        get_pallas_distributed_top_k_op(world_size)


def pallas_distributed_top_k(
    local_logits: torch.Tensor,
    *,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int,
    world_size: int,
    model_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return replicated exact global TopK values and unpadded token IDs."""

    rank_input = local_logits.new_tensor((model_rank,), dtype=torch.int32)
    values, token_ids = get_pallas_distributed_top_k_op(world_size)(
        local_logits,
        rank_input,
        org_vocab_size,
        global_vocab_size,
        org_vocab_padded_count,
        k,
    )
    return values.float(), token_ids
