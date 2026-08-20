# SPDX-License-Identifier: Apache-2.0
"""Focused physical-TPU probes for the TP16 distributed TopK core halt.

The gather scan deliberately runs the known-safe-sized candidate tensor first.
If the larger operation core-halts, its last PASS marker identifies the boundary.
The fused pipeline replaces only the final TorchTPU ``aten::gather`` lowering.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '1')}",
)

import torch
import torch.distributed as dist
from torch_tpu._internal.distributed import tpu_distributed  # noqa: F401

_BATCH_SIZE = 128
_TOP_K = 50


def _materialize(stage: str, value: torch.Tensor) -> torch.Tensor:
    """Synchronize and print only after the operation has completed on device."""

    result = value.cpu()
    torch.tpu.synchronize()
    print(
        f"TOPK_PROBE_STAGE_PASS rank={dist.get_rank()} stage={stage} "
        f"shape={tuple(result.shape)} dtype={result.dtype}",
        flush=True,
    )
    return result


def _run_gather_scan(widths: list[int]) -> None:
    """Exercise TorchTPU's int64-index StableHLO gather at candidate widths."""

    rank = dist.get_rank()
    for width in widths:
        source_cpu = torch.arange(_BATCH_SIZE * width, dtype=torch.int32).reshape(
            _BATCH_SIZE, width
        )
        rows = torch.arange(_BATCH_SIZE, dtype=torch.int64).unsqueeze(1)
        slots = torch.arange(_TOP_K, dtype=torch.int64).unsqueeze(0)
        index_cpu = (rows * 17 + slots * 13 + rank) % width
        print(
            f"TOPK_PROBE_STAGE_START rank={rank} stage=torch_gather_i64_w{width}",
            flush=True,
        )
        actual = torch.gather(
            source_cpu.to("tpu"),
            dim=1,
            index=index_cpu.to("tpu"),
        )
        actual_cpu = _materialize(f"torch_gather_i64_w{width}", actual)
        torch.testing.assert_close(
            actual_cpu,
            torch.gather(source_cpu, dim=1, index=index_cpu),
            rtol=0,
            atol=0,
        )


def _build_top_k_with_ids_op():
    """Fuse native TopK and ID selection, retaining int32 indices in XLA."""

    import jax
    import jax.numpy as jnp

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        values: jax.Array,
        token_ids: jax.Array,
        k: int,
    ) -> tuple[jax.Array, jax.Array]:
        top_values, positions = jax.lax.top_k(values, k)
        top_token_ids = jnp.take_along_axis(token_ids, positions, axis=-1)
        return top_values, top_token_ids.astype(jnp.int32)

    return cpu_export_jax_op(
        "sglang_tpu_probe::exact_top_k_with_ids",
        _wrapped,
    )


class _DistGroup:
    """The concatenating all-gather contract used by SGLang's TP group."""

    def __init__(self, world_size: int):
        self.world_size = world_size

    def all_gather(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += value.dim()
        output = torch.empty(
            (self.world_size * value.shape[0], *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        dist.all_gather_into_tensor(output, value)
        output = output.reshape(self.world_size, *value.shape).movedim(0, dim)
        shape = list(value.shape)
        shape[dim] *= self.world_size
        return output.reshape(shape)


def _prepare_candidate_pipeline() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run local TopK and the production packed candidate all-gather."""

    from sglang.srt.hardware_backend.tpu import top_k as top_k_module

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_vocab = 1024
    global_vocab = world_size * local_vocab
    group = _DistGroup(world_size)
    shard = top_k_module.TpuVocabShard(
        width=local_vocab,
        global_vocab_size=global_vocab,
        org_vocab_start=rank * local_vocab,
        org_vocab_count=local_vocab,
        org_vocab_padded_count=local_vocab,
        added_vocab_start=global_vocab,
        added_vocab_count=0,
    )
    row_offsets = torch.arange(_BATCH_SIZE, dtype=torch.float32).unsqueeze(1) / 1024
    local_ids = torch.arange(local_vocab, dtype=torch.float32).unsqueeze(0)
    local_logits_cpu = local_ids + rank * local_vocab + row_offsets
    local_logits = local_logits_cpu.to("tpu")

    print(f"TOPK_PROBE_STAGE_START rank={rank} stage=local_top_k", flush=True)
    local_values, local_token_ids = top_k_module.tpu_sharded_top_k(
        local_logits,
        shard,
        _TOP_K,
    )
    _materialize("local_top_k_values", local_values[:, :1])
    _materialize("local_top_k_ids", local_token_ids[:, :1])

    packed = torch.stack((local_values.float(), local_token_ids.float()), dim=-1)
    print(f"TOPK_PROBE_STAGE_START rank={rank} stage=candidate_all_gather", flush=True)
    gathered = group.all_gather(packed, dim=-2)
    gathered_cpu = _materialize("candidate_all_gather", gathered[:, :1, :])
    if tuple(gathered.shape) != (_BATCH_SIZE, world_size * _TOP_K, 2):
        raise AssertionError(f"unexpected gathered shape: {tuple(gathered.shape)}")
    if gathered_cpu.shape != (_BATCH_SIZE, 1, 2):
        raise AssertionError(
            f"unexpected materialized slice: {tuple(gathered_cpu.shape)}"
        )

    gathered_values = gathered[..., 0]
    gathered_ids = gathered[..., 1].to(torch.int32)
    return gathered_values, gathered_ids, global_vocab


def _expected_global_top_k(global_vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    row_offsets = torch.arange(_BATCH_SIZE, dtype=torch.float32).unsqueeze(1) / 1024
    expected_ids = torch.arange(
        global_vocab - 1,
        global_vocab - _TOP_K - 1,
        -1,
        dtype=torch.int32,
    ).repeat(_BATCH_SIZE, 1)
    return expected_ids.float() + row_offsets, expected_ids


def _run_original_pipeline() -> None:
    """Run the production final TopK followed by TorchTPU aten::gather."""

    from sglang.srt.hardware_backend.tpu import top_k as top_k_module

    gathered_values, gathered_ids, global_vocab = _prepare_candidate_pipeline()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"TOPK_PROBE_STAGE_START rank={rank} stage=global_top_k", flush=True)
    global_values, global_positions = top_k_module.tpu_top_k(
        gathered_values,
        _TOP_K,
    )
    actual_values = _materialize("global_top_k_values", global_values)
    _materialize("global_top_k_positions", global_positions)

    print(
        f"TOPK_PROBE_STAGE_START rank={rank} stage=global_id_torch_gather",
        flush=True,
    )
    global_ids = torch.gather(
        gathered_ids,
        dim=1,
        index=global_positions.to(torch.int64),
    )
    actual_ids = _materialize("global_id_torch_gather", global_ids)
    expected_values, expected_ids = _expected_global_top_k(global_vocab)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    print(
        f"TOPK_ORIGINAL_PIPELINE_PASS rank={rank} "
        f"world_size={world_size} candidate_width={gathered_values.shape[1]}",
        flush=True,
    )


def _run_fused_pipeline() -> None:
    """Run the production candidate pipeline with a fused final ID lookup."""

    gathered_values, gathered_ids, global_vocab = _prepare_candidate_pipeline()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    fused_top_k = _build_top_k_with_ids_op()
    print(f"TOPK_PROBE_STAGE_START rank={rank} stage=fused_global_top_k", flush=True)
    global_values, global_ids = fused_top_k(
        gathered_values,
        gathered_ids,
        _TOP_K,
    )
    actual_values = _materialize("fused_global_top_k_values", global_values)
    actual_ids = _materialize("fused_global_top_k_ids", global_ids)

    expected_values, expected_ids = _expected_global_top_k(global_vocab)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    print(
        f"TOPK_FUSED_PIPELINE_PASS rank={rank} world_size={world_size} "
        f"candidate_width={world_size * _TOP_K}",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: topk_multihost_probe.py gather-scan WIDTH... | original | fused"
        )
    dist.init_process_group(backend="tpu_dist")
    try:
        mode = sys.argv[1]
        if mode == "gather-scan":
            widths = [int(value) for value in sys.argv[2:]]
            if not widths:
                raise ValueError("gather-scan requires at least one width")
            _run_gather_scan(widths)
        elif mode == "original":
            _run_original_pipeline()
        elif mode == "fused":
            _run_fused_pipeline()
        else:
            raise ValueError(f"unknown probe mode: {mode}")
        print(
            f"TOPK_PROBE_PASS rank={dist.get_rank()} mode={mode} "
            f"world_size={dist.get_world_size()}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
