"""Backend dispatch for the fused scale/shift kernel.

Callers import from here; ``triton/`` stays triton-only. The triton module
is imported lazily so non-triton backends never load it.
"""

import torch


def fuse_scale_shift_kernel(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
    block_l: int = 128,
    block_c: int = 128,
) -> torch.Tensor:
    if (x.is_cuda and scale.is_cuda) or (x.is_xpu and scale.is_xpu):
        from sglang.kernels.ops.diffusion.triton.scale_shift import (
            fuse_scale_shift_kernel as _triton_kernel,
        )

        return _triton_kernel(x, scale, shift, scale_constant, block_l, block_c)

    from sglang.kernels.ops.diffusion.torch_native.scale_shift import (
        fuse_scale_shift_torch,
    )

    return fuse_scale_shift_torch(x, scale, shift, scale_constant)
