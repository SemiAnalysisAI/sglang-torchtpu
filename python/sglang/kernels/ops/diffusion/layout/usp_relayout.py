# SPDX-License-Identifier: Apache-2.0
"""Device-neutral Ulysses head merge with a lazy CUDA implementation."""

import torch

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def can_use_usp_merge_heads(x: torch.Tensor) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and torch.version.hip is None
        and x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and x.dim() == 5
        and x.numel() > 0
        and x.is_contiguous()
    )


def usp_merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[W, S, B, H, D] -> [B, S, W, H, D], without GPU imports on fallback.

    Keep backend selection outside the JIT module so unsupported devices and
    compiled regions can use the exact PyTorch relayout without CUDA build
    dependencies. The eligible eager CUDA path is unchanged.
    """
    if not torch.compiler.is_compiling() and can_use_usp_merge_heads(x):
        from .usp_relayout_jit import _usp_merge_heads_cuda

        return _usp_merge_heads_cuda(x)
    return x.permute(2, 1, 0, 3, 4).contiguous()
