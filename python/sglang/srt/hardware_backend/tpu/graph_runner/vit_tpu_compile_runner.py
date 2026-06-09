"""ViT TPU compile runner — wraps ViT blocks + merger with torch.compile."""

from __future__ import annotations

import inspect
import logging

import torch

from sglang.srt.hardware_backend.tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)


def _cu_seqlens_to_segment_ids(
    cu_seqlens: torch.Tensor,
    total_patches: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert cumulative sequence lengths to per-patch segment IDs.

    Returns an *unpadded* [1, total_patches] int32 tensor.  Padding to the
    flash-attention block size is handled downstream by VisionTpuAttention.
    """
    cu = cu_seqlens.cpu().tolist()
    assert cu[-1] == total_patches, (
        f"cu_seqlens[-1]={cu[-1]} != total_patches={total_patches}"
    )
    seg = torch.zeros(1, total_patches, dtype=torch.int32, device=device)
    for i in range(1, len(cu)):
        seg[0, cu[i - 1] : cu[i]] = i - 1
    return seg


class ViTTpuCompileRunner:
    """Vision-encoder runner for TPU using ``torch.compile``.

    If the vision blocks do not accept ``segment_ids`` (e.g. Qwen2-VL /
    Qwen2.5-VL style), the runner falls back to eager execution using
    ``position_embeddings`` instead.
    """

    def __init__(self, vit):
        self.vit = vit
        self._compiled_fn = None

        self._deepstack_visual_indexes = set(
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)

        # Detect whether blocks accept segment_ids; fall back to eager if not.
        first_block = vit.blocks[0] if len(vit.blocks) > 0 else None
        if first_block is not None:
            sig = inspect.signature(first_block.forward)
            self._supports_segment_ids = "segment_ids" in sig.parameters
        else:
            self._supports_segment_ids = False
        if not self._supports_segment_ids:
            logger.warning(
                "ViTTpuCompileRunner: blocks do not accept segment_ids — "
                "falling back to eager ViT execution."
            )

    def _blocks_and_merger(
        self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, segment_ids
    ):
        vit = self.vit
        deepstack_features = []
        num_ds = 0

        for layer_num, blk in enumerate(vit.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                segment_ids=segment_ids,
            )
            if layer_num in self._deepstack_visual_indexes:
                ds_out = self._deepstack_merger_list[num_ds](x)
                deepstack_features.append(ds_out)
                num_ds += 1

        x = vit.merger(x)
        if deepstack_features:
            return torch.cat([x] + deepstack_features, dim=1)
        return x

    def _blocks_and_merger_fallback(self, x, cu_seqlens, position_embeddings):
        """Eager fallback for blocks that do not accept segment_ids."""
        vit = self.vit
        deepstack_features = []
        num_ds = 0

        for layer_num, blk in enumerate(vit.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self._deepstack_visual_indexes:
                ds_out = self._deepstack_merger_list[num_ds](x)
                deepstack_features.append(ds_out)
                num_ds += 1

        x = vit.merger(x)
        if deepstack_features:
            return torch.cat([x] + deepstack_features, dim=1)
        return x

    def run(
        self,
        x,
        cu_seqlens,
        position_embeddings=None,
        rotary_pos_emb_cos=None,
        rotary_pos_emb_sin=None,
        cu_window_seqlens=None,
        output_indices=None,
    ):
        x_3d = x.unsqueeze(1)
        seq_len = x_3d.shape[0]

        if not self._supports_segment_ids:
            result = self._blocks_and_merger_fallback(
                x_3d, cu_seqlens, position_embeddings
            )
            if output_indices is not None:
                result = result.index_select(0, output_indices)
            return result

        segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)

        if self._compiled_fn is None:
            _register_tpu_compile_backend()
            self._compiled_fn = torch.compile(
                torch.no_grad()(self._blocks_and_merger),
                backend="tpu",
                fullgraph=True,
                dynamic=False,
            )
            logger.info(
                "ViTTpuCompileRunner: compiled blocks+merger with "
                "torch.compile(backend='tpu')."
            )

        result = self._compiled_fn(
            x_3d, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, segment_ids
        )

        if output_indices is not None:
            result = result.index_select(0, output_indices)

        return result
