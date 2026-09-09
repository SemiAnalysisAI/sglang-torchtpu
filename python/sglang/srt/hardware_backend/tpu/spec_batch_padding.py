"""Host-side shape admission for compiled TPU speculative helpers."""

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs


def physical_batch_size(logical_bs: int, buckets: tuple[int, ...]) -> int:
    if logical_bs == 0 or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
        return logical_bs
    assert buckets, "Missing TPU helper buckets"
    candidates = [size for size in buckets if size >= logical_bs]
    assert candidates, f"Batch size {logical_bs} exceeds TPU helper buckets {buckets}"
    return min(candidates)


def pad_rows(value: torch.Tensor, rows: int, fill: int = 0) -> torch.Tensor:
    assert value.shape[0] <= rows
    padding = (0, 0) * (value.ndim - 1) + (0, rows - value.shape[0])
    return F.pad(value, padding, value=fill) if rows != value.shape[0] else value
