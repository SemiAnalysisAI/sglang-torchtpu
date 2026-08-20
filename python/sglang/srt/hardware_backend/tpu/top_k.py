# SPDX-License-Identifier: Apache-2.0
"""Exact TPU TopK primitives for vocab-parallel token sampling.
Sharded reduction communicates only local TopK logits and global token IDs."""

import os
from collections.abc import Callable
from typing import NamedTuple

import torch

from sglang.srt.environ import envs

_TPU_TOP_K_OP: Callable | None = None
_TPU_SHARDED_TOP_K_OP: Callable | None = None
_PALLAS_WORLD_SIZES = frozenset((8, 16))


def supports_distributed_tpu_top_k(lm_head, top_k: int) -> bool:
    """Return whether an LM head has a usable vocab-sharded TPU layout.
    Replicated or incomplete heads must retain the full-logit sampling path."""

    required_layout = (
        "num_embeddings",
        "num_embeddings_per_partition",
        "shard_indices",
    )
    return (
        getattr(lm_head, "enable_tp", False)
        and all(hasattr(lm_head, attr) for attr in required_layout)
        and 0 < top_k <= lm_head.num_embeddings
        and top_k <= lm_head.num_embeddings_per_partition
    )


class TpuVocabShard(NamedTuple):
    """Describe valid base and added tokens inside one padded LM-head shard.
    Device kernels use this metadata to mask padding and reconstruct global IDs.

    A NamedTuple rather than the usual msgspec.Struct: it is constructed
    inside the torch.compile'd forward (fullgraph, so no graph break is
    possible there), and Dynamo cannot instantiate msgspec's C-level
    ``__new__`` in-graph while NamedTuples are traced natively."""

    width: int
    global_vocab_size: int
    org_vocab_size: int
    org_vocab_start: int
    org_vocab_count: int
    org_vocab_padded_count: int
    added_vocab_start: int
    added_vocab_count: int

    @classmethod
    def from_lm_head(cls, lm_head) -> "TpuVocabShard":
        """Build validated host metadata from an LM head's partition indices.
        Invalid padded layouts fail before they can produce device token IDs."""

        if not getattr(lm_head, "enable_tp", False):
            raise ValueError("distributed TPU TopK requires a vocab-sharded LM head")
        indices = lm_head.shard_indices
        shard = cls(
            width=lm_head.num_embeddings_per_partition,
            global_vocab_size=lm_head.num_embeddings,
            org_vocab_size=getattr(lm_head, "org_vocab_size", lm_head.num_embeddings),
            org_vocab_start=indices.org_vocab_start_index,
            org_vocab_count=indices.num_org_elements,
            org_vocab_padded_count=indices.num_org_elements_padded,
            added_vocab_start=indices.added_vocab_start_index,
            added_vocab_count=indices.num_added_elements,
        )
        if shard.org_vocab_padded_count > shard.width:
            raise ValueError(
                "base-vocabulary padded width exceeds LM-head shard width: "
                f"{shard.org_vocab_padded_count} > {shard.width}"
            )
        if shard.org_vocab_padded_count + shard.added_vocab_count > shard.width:
            raise ValueError("added vocabulary exceeds LM-head shard width")
        if not 0 < shard.org_vocab_size <= shard.global_vocab_size:
            raise ValueError("original vocabulary must fit the global vocabulary")
        return shard


def _validate_tpu_sharded_top_k(
    values: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
) -> None:
    if values.device.type != "tpu":
        raise ValueError(
            f"tpu_sharded_top_k requires a TPU tensor, got {values.device}"
        )
    if values.ndim != 2:
        raise ValueError(
            f"tpu_sharded_top_k expects a 2D tensor, got shape {values.shape}"
        )
    if values.shape[-1] != shard.width:
        raise ValueError(
            "vocab shard width does not match local logits: "
            f"{shard.width} != {values.shape[-1]}"
        )
    if not 0 < k <= shard.width or k > shard.global_vocab_size:
        raise ValueError(
            "k must fit both the local padded shard and global vocabulary, "
            f"got k={k}, shard_width={shard.width}, "
            f"global_vocab_size={shard.global_vocab_size}"
        )


def _is_canonical_pallas_shard(shard: TpuVocabShard, rank: int) -> bool:
    """Whether rank-based Pallas ID reconstruction matches this shard."""

    org_start = min(rank * shard.org_vocab_padded_count, shard.org_vocab_size)
    org_end = min(
        (rank + 1) * shard.org_vocab_padded_count,
        shard.org_vocab_size,
    )
    added_width = shard.width - shard.org_vocab_padded_count
    added_start = min(
        shard.org_vocab_size + rank * added_width,
        shard.global_vocab_size,
    )
    added_end = min(
        shard.org_vocab_size + (rank + 1) * added_width,
        shard.global_vocab_size,
    )
    return (
        added_width >= 0
        and shard.org_vocab_start == org_start
        and shard.org_vocab_count == org_end - org_start
        and shard.added_vocab_start == added_start
        and shard.added_vocab_count == added_end - added_start
    )


def supports_pallas_distributed_tpu_top_k(
    shard: TpuVocabShard,
    k: int,
    *,
    group_world_size: int,
    model_rank: int | None,
) -> bool:
    """Return whether static metadata guarantees the direct-HBM path.

    Multi-host callers use this before skipping the full-vocabulary gather.
    That prevents a later Pallas miss from reaching the native collective,
    which is not supported across TPU v7 hosts.
    """

    return (
        supports_pallas_distributed_tpu_top_k_topology(group_world_size)
        and isinstance(model_rank, int)
        and 0 <= model_rank < group_world_size
        and shard.width >= 64
        and 0 < k <= 64
        and _is_canonical_pallas_shard(shard, model_rank)
    )


def supports_pallas_distributed_tpu_top_k_topology(group_world_size: int) -> bool:
    """Return whether a process group can use the full-world Pallas ring."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return (
        not envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.get()
        and group_world_size == world_size
        and world_size in _PALLAS_WORLD_SIZES
    )


def _get_tpu_top_k_op() -> Callable:
    """Create the exported native-XLA TopK custom op once per process.
    Registration remains lazy so non-TPU sampling does not import JAX."""

    global _TPU_TOP_K_OP
    if _TPU_TOP_K_OP is not None:
        return _TPU_TOP_K_OP

    import jax

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(values: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
        # torch_tpu's aten::topk lowering is currently a full StableHLO
        # sort plus slice. lax.top_k exports XLA's native exact TopK op.
        return jax.lax.top_k(values, k)

    op = cpu_export_jax_op("sglang_tpu_pallas::exact_top_k", _wrapped)

    def _fake(values: torch.Tensor, k: int):
        shape = (*values.shape[:-1], k)
        return (
            values.new_empty(shape),
            values.new_empty(shape, dtype=torch.int32),
        )

    op.register_fake(_fake)
    _TPU_TOP_K_OP = op
    return op


def tpu_top_k(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exact descending TopK values from a two-dimensional TPU tensor.
    Indices are int32 positions in the input's final dimension."""

    if values.device.type != "tpu":
        raise ValueError(f"tpu_top_k requires a TPU tensor, got {values.device}")
    if values.ndim != 2:
        raise ValueError(f"tpu_top_k expects a 2D tensor, got shape {values.shape}")
    if not 0 < k <= values.shape[-1]:
        raise ValueError(f"k must be in [1, {values.shape[-1]}], got {k}")
    return _get_tpu_top_k_op()(values, k)


def _get_tpu_sharded_top_k_op() -> Callable:
    """Create the fused padding-mask, local TopK, and ID-mapping custom op.
    Fusion avoids materializing a vocabulary-sized global-ID tensor."""

    global _TPU_SHARDED_TOP_K_OP
    if _TPU_SHARDED_TOP_K_OP is not None:
        return _TPU_SHARDED_TOP_K_OP

    import jax
    import jax.numpy as jnp

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        values: jax.Array,
        k: int,
        org_vocab_start: int,
        org_vocab_count: int,
        org_vocab_padded_count: int,
        added_vocab_start: int,
        added_vocab_count: int,
    ) -> tuple[jax.Array, jax.Array]:
        positions = jnp.arange(values.shape[-1], dtype=jnp.int32)
        added_positions = positions - org_vocab_padded_count
        valid = (positions < org_vocab_count) | (
            (added_positions >= 0) & (added_positions < added_vocab_count)
        )
        masked_values = jnp.where(valid, values, -jnp.inf)
        top_values, top_positions = jax.lax.top_k(masked_values, k)
        token_ids = jnp.where(
            top_positions < org_vocab_count,
            org_vocab_start + top_positions,
            added_vocab_start + top_positions - org_vocab_padded_count,
        )
        return top_values, token_ids.astype(jnp.int32)

    op = cpu_export_jax_op("sglang_tpu_pallas::exact_sharded_top_k", _wrapped)

    def _fake(
        values: torch.Tensor,
        k: int,
        org_vocab_start: int,
        org_vocab_count: int,
        org_vocab_padded_count: int,
        added_vocab_start: int,
        added_vocab_count: int,
    ):
        del (
            org_vocab_start,
            org_vocab_count,
            org_vocab_padded_count,
            added_vocab_start,
            added_vocab_count,
        )
        shape = (*values.shape[:-1], k)
        return (
            values.new_empty(shape),
            values.new_empty(shape, dtype=torch.int32),
        )

    op.register_fake(_fake)
    _TPU_SHARDED_TOP_K_OP = op
    return op


def tpu_sharded_top_k(
    values: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one shard's exact TopK while excluding padded columns.
    Returned int32 IDs already refer to the unpadded global vocabulary."""

    _validate_tpu_sharded_top_k(values, shard, k)
    return _get_tpu_sharded_top_k_op()(
        values,
        k,
        shard.org_vocab_start,
        shard.org_vocab_count,
        shard.org_vocab_padded_count,
        shard.added_vocab_start,
        shard.added_vocab_count,
    )


def register_tpu_top_k_ops() -> None:
    """Register local and global TopK ops before full-graph tracing begins.
    Eager registration keeps custom-op schemas visible to torch.compile."""

    _get_tpu_top_k_op()
    _get_tpu_sharded_top_k_op()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (
        world_size in _PALLAS_WORLD_SIZES
        and not envs.SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K.get()
    ):
        from sglang.srt.hardware_backend.tpu.distributed_topk_pallas import (
            register_pallas_distributed_top_k_op,
        )

        register_pallas_distributed_top_k_op(world_size)


def distributed_tpu_top_k_collective(
    local_logits: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
    *,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge shard-local TopK candidates with Torch collectives."""

    sharded_top_k_logits, sharded_top_k_token_ids = tpu_sharded_top_k(
        local_logits, shard, k
    )

    if group.world_size > 1:
        if shard.global_vocab_size <= 2**24:
            # Token IDs are exact in fp32 for every practical serving
            # vocabulary. Packing values and IDs halves collective latency at
            # the cost of a tiny payload increase over bf16 + int32.
            packed_sharded_top_k = torch.stack(
                (sharded_top_k_logits.float(), sharded_top_k_token_ids.float()),
                dim=-1,
            )
            gathered_sharded_top_k = group.all_gather(packed_sharded_top_k, dim=-2)
            gathered_sharded_top_k_logits = gathered_sharded_top_k[..., 0]
            gathered_sharded_top_k_token_ids = gathered_sharded_top_k[..., 1].to(
                torch.int32
            )
        else:
            # Preserve exact IDs for unusually large vocabularies.
            gathered_sharded_top_k_logits = group.all_gather(
                sharded_top_k_logits, dim=-1
            )
            gathered_sharded_top_k_token_ids = group.all_gather(
                sharded_top_k_token_ids, dim=-1
            )
    else:
        gathered_sharded_top_k_logits = sharded_top_k_logits
        gathered_sharded_top_k_token_ids = sharded_top_k_token_ids

    global_top_k_logits, global_top_k_positions = tpu_top_k(
        gathered_sharded_top_k_logits, k
    )
    global_top_k_token_ids = torch.gather(
        gathered_sharded_top_k_token_ids,
        dim=1,
        index=global_top_k_positions.to(torch.int64),
    )
    return global_top_k_logits.float(), global_top_k_token_ids.to(torch.int32)


def distributed_tpu_top_k(
    local_logits: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
    *,
    group,
    require_pallas: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge exact sharded TopK logits into a replicated global TopK."""

    _validate_tpu_sharded_top_k(local_logits, shard, k)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    model_rank = getattr(group, "rank_in_group", None)
    if (
        supports_pallas_distributed_tpu_top_k(
            shard,
            k,
            group_world_size=group.world_size,
            model_rank=model_rank,
        )
        and local_logits.shape[0] > 0
        and local_logits.dtype in (torch.bfloat16, torch.float32)
    ):
        from sglang.srt.hardware_backend.tpu.distributed_topk_pallas import (
            pallas_distributed_top_k,
        )

        return pallas_distributed_top_k(
            local_logits,
            org_vocab_size=shard.org_vocab_size,
            global_vocab_size=shard.global_vocab_size,
            org_vocab_padded_count=shard.org_vocab_padded_count,
            k=k,
            world_size=world_size,
            model_rank=model_rank,
        )

    if require_pallas:
        raise RuntimeError(
            "multi-host distributed TPU TopK lost Pallas eligibility; "
            "select the full-vocabulary gather before computing local logits"
        )

    return distributed_tpu_top_k_collective(
        local_logits,
        shard,
        k,
        group=group,
    )
