"""Kimi-K3 vision encoder runner + policy for TPU ``torch.compile``.

Kimi-K3's tower is MoonViT-shaped (complex 2-D RoPE, a temporal-pool merger)
but its encoder layers take K3-specific metadata (Python segment bounds,
backend names) that would put a guard on every image layout, and its eager
attention on TPU is per-segment SDPA. So the TPU runner drives the layers'
weights itself, the way ``ViTTpuCompileRunner`` drives Qwen3-VL's blocks:
segment-id Pallas attention (``VisionTpuAttention``) and real-arithmetic RoPE,
over a fixed-shape patch bucket.

Compiled boundary = the encoder **layer loop + ``final_layernorm``**. The patch
embedding, the interpolated position embedding, the RoPE table lookup and the
temporal-pool merger stay eager, as for Kimi-K2.5 (``moonvit_tpu_compile_runner``),
whose per-image bucketing and padding this reuses.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang_tpu.graph_runner.moonvit_tpu_compile_runner import MoonVitTpuCompilePolicy
from sglang_tpu.graph_runner.vit_tpu_compile_runner import (
    VisionPaddingContext,
    _compute_patch_counts,
    _cu_seqlens_to_segment_ids,
    _validate_patch_buckets,
)

logger = logging.getLogger(__name__)

# The merger pools 2x2 windows, so every image covers a multiple of 4 patches;
# square buckets with an even side keep that.
_MERGE_SIZE = 2


def rotate_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Real-arithmetic form of Kimi-K3's complex ``apply_rope``.

    Adjacent pairs (x[2i], x[2i+1]) rotate by angle i, exactly as
    ``view_as_real(view_as_complex(x) * (cos + i*sin))``.

    Args:
        x: ``[S, num_heads, head_dim]``.
        cos, sin: ``[S, head_dim/2]``.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    pairs = x.float().view(*x.shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
    return rotated.flatten(-2).type_as(x)


class KimiK3VitTpuCompileRunner:
    """Runs Kimi-K3's encoder layers + ``final_layernorm`` on TPU.

    Eager until ``KimiK3VisionTpuCompilePolicy.install`` compiles ``forward``
    and routes ``run`` through the padded, bucketed policy. Eager and compiled
    use the same math, so a compiled server serves what an eager one does.
    """

    def __init__(self, encoder: Any, attention_cls: Any = None) -> None:
        self.encoder = encoder
        self.owner = encoder
        if attention_cls is None:
            from sglang.srt.layers.attention.vision import VisionTpuAttention

            attention_cls = VisionTpuAttention
        # The layers hold no attention module of their own on TPU (their eager
        # path calls SDPA per segment), and the segment-id attention has no
        # weights, so one instance per layer lives here.
        self.attention = [
            attention_cls(
                head_dim=block.hidden_size_per_attention_head,
                num_heads=block.num_heads,
                num_kv_heads=block.num_heads,
            )
            for block in encoder.blocks
        ]

    @property
    def supports_segment_ids(self) -> bool:
        return all(
            getattr(attention, "supports_segment_ids", False)
            for attention in self.attention
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encoder over one patch bucket.

        Args:
            x: ``[S, D]`` patch features, position embedding added.
            cos, sin: ``[S, head_dim/2]`` RoPE tables.
            segment_ids: ``[1, S]`` int32 image id per patch; padding carries
                its own trailing id, so it never attends to real patches.
        Returns:
            ``[S, D]``.
        """
        seq_len = x.shape[0]
        for block, attention in zip(self.encoder.blocks, self.attention):
            residual = x
            hidden = block.norm0(x)
            qkv = block.wqkv(hidden).view(
                seq_len, 3, block.num_heads, block.hidden_size_per_attention_head
            )
            q, k, v = torch.unbind(qkv, dim=1)
            q = rotate_interleaved(q, cos, sin)
            k = rotate_interleaved(k, cos, sin)
            attn = attention(q, k, v, 1, segment_ids=segment_ids)
            x = residual + block.wo(attn.flatten(start_dim=-2))
            x = x + block.mlp(block.norm1(x))
        return self.encoder.final_layernorm(x)

    def run(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Eager entry point, used until the policy installs the compiled fn."""
        segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, x.shape[0], x.device)
        return self.forward(x, cos, sin, segment_ids)


class KimiK3VisionTpuCompilePolicy(MoonVitTpuCompilePolicy):
    """Compile policy for Kimi-K3's encoder layer loop + ``final_layernorm``.

    MoonViT's per-image bucketing and padding, with K3's runner output
    (``[S, D]``, no batch axis) and buckets aligned to the 2x2 merger.
    """

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        name: str = "kimi_k3_vit_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, name=name, **kwargs)

    def compute_patch_buckets(self) -> list[int]:
        if envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.is_set():
            # Set but empty => () => warms nothing, i.e. precompile disabled.
            buckets = sorted(
                {int(x) for x in envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()}
            )
            return _validate_patch_buckets(buckets, _MERGE_SIZE)
        return _compute_patch_counts(
            _MERGE_SIZE, max_patches=envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get()
        )

    def unpad_outputs(
        self, result: torch.Tensor, ctx: VisionPaddingContext
    ) -> torch.Tensor:
        # result is [S_bucket, D]; padding is along the patch axis.
        if ctx.padded:
            result = result[: ctx.real_seq_len]
        return result
