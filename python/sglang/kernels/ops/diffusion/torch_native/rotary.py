"""Torch-native rotary for backends without triton; mirrors the triton
kernel (interleaved pairs, fp32 rotation). Parity-tested in
``tests/test_diffusion_triton_fallbacks.py``."""

import torch


def apply_rotary_embedding_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    head_size = x.shape[-1]
    num_tokens = x.shape[-3]
    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    cos_f = cos.float().reshape(num_tokens, 1, head_size // 2)
    sin_f = sin.float().reshape(num_tokens, 1, head_size // 2)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    out = torch.stack(
        (x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f), dim=-1
    ).flatten(-2)
    return out.to(x.dtype)
