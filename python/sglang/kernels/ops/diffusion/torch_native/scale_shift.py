"""Torch-native scale/shift for backends without triton.

Same math as the fused triton kernel
(``kernels/ops/diffusion/triton/scale_shift.py``), relying on broadcasting;
parity-tested against it in ``tests/test_diffusion_triton_fallbacks.py``.
"""

import torch


def fuse_scale_shift_torch(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
) -> torch.Tensor:
    B, L, C = x.shape

    def _per_token(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:  # [B, F, 1, C] per-frame -> [B, L, C] per-token
            num_frames = t.shape[1]
            return t.expand(B, num_frames, L // num_frames, C).reshape(B, L, C)
        if t.dim() == 2:  # [B, C] -> [B, 1, C]
            return t[:, None, :]
        return t  # 0D/1D scalars and [B, L, C] broadcast natively

    return x * (scale_constant + _per_token(scale)) + _per_token(shift)
