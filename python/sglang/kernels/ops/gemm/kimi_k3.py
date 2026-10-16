from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.utils import is_hip, is_npu, is_tpu

if TYPE_CHECKING:
    import torch

_is_hip = is_hip()
_is_npu = is_npu()
_is_tpu = is_tpu()

# (n, k) -> the largest num_tokens where the tiny GEMM still beats cuBLAS.
# Doubles as the compile-time max_m: one kernel is built per m up to it.
_K3_TINY_GEMM_MAX_TOKENS = {
    (144, 7168): 16,
    (896, 7168): 8,
    (1536, 128): 12,
}


def kimi_k3_tiny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    import torch

    from .tiny_gemm import tiny_gemm_bf16

    m, k = x.shape
    n, _ = w.shape
    max_num_tokens = _K3_TINY_GEMM_MAX_TOKENS.get((n, k))
    if (
        not _is_npu
        and not _is_tpu
        and not _is_hip
        and x.is_cuda
        and w.is_cuda
        and max_num_tokens is not None
        and 0 < m <= max_num_tokens
    ):
        return tiny_gemm_bf16(x, w, max_m=max_num_tokens)
    return torch.nn.functional.linear(x, w)
