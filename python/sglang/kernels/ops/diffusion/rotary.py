"""Backend dispatch for the rotary-embedding kernel.

Callers import from here; ``triton/`` stays triton-only. The triton module
is imported lazily so non-triton backends never load it.
"""

import torch


def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    if x.is_cuda or x.is_xpu or x.device.type == "musa":
        from sglang.kernels.ops.diffusion.triton.rotary import (
            apply_rotary_embedding as _triton_kernel,
        )

        return _triton_kernel(x, cos, sin, interleaved)

    from sglang.kernels.ops.diffusion.torch_native.rotary import (
        apply_rotary_embedding_torch,
    )

    return apply_rotary_embedding_torch(x, cos, sin, interleaved)
