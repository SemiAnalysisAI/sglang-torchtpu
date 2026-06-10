"""ViT TPU compile runner — wraps ViT blocks + merger with torch.compile."""

from __future__ import annotations

import inspect
import logging
import os

import torch
from torch_tpu._internal.sync.sync import synchronize as tpu_sync

from sglang.srt.hardware_backend.tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)


def _compute_patch_counts(patch_size, merge_size, max_patches=1600):
    """Compute valid square-image patch counts from model config.

    Image dims are snapped to multiples of (patch_size * merge_size) by the
    preprocessor, so grid dims are always multiples of merge_size.  This
    enumerates the resulting square patch counts up to max_patches.
    """
    counts = []
    grid_dim = merge_size
    while True:
        pc = grid_dim * grid_dim
        if pc > max_patches:
            break
        counts.append(pc)
        grid_dim += merge_size
    return counts


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

    def _ensure_compiled(self):
        """Lazily create the compiled function if it does not exist yet."""
        if self._compiled_fn is not None:
            return
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

    def precompile(self, patch_counts=None):
        """Warm up the torch.compile cache for common patch counts.

        Each unique patch count triggers a fresh XLA compilation on first
        encounter (~9-13 s).  By pre-compiling for common sizes at startup,
        real requests are more likely to hit a cached graph.

        Args:
            patch_counts: list of int patch counts to precompile for.
                Defaults to patch counts computed from the model config
                (square images up to SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES
                patches, default 1600).  Set the env var
                SGLANG_TPU_VIT_PRECOMPILE_PATCHES to a comma-separated
                list to override, or to an empty string to disable.
        """
        if not self._supports_segment_ids:
            logger.info(
                "ViTTpuCompileRunner.precompile: skipped (segment_ids not "
                "supported, using eager fallback)."
            )
            return

        vit = self.vit

        # Resolve patch counts from env var, argument, or default.
        env_val = os.environ.get("SGLANG_TPU_VIT_PRECOMPILE_PATCHES", None)
        if env_val is not None:
            if env_val.strip() == "":
                logger.info(
                    "ViTTpuCompileRunner.precompile: disabled via "
                    "SGLANG_TPU_VIT_PRECOMPILE_PATCHES=''."
                )
                return
            patch_counts = [int(x.strip()) for x in env_val.split(",") if x.strip()]
        elif patch_counts is None:
            patch_size = getattr(vit, "patch_size", 16)
            merge_size = getattr(vit, "spatial_merge_size", 2)
            max_patches = int(os.environ.get(
                "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES", 1600
            ))
            patch_counts = _compute_patch_counts(
                patch_size, merge_size, max_patches
            )

        if not patch_counts:
            return

        self._ensure_compiled()

        device = vit.patch_embed.proj.weight.device
        dtype = vit.patch_embed.proj.weight.dtype

        # Derive tensor dimensions from the model.
        embed_dim = vit.blocks[0].norm1.normalized_shape[0]
        head_dim = vit.blocks[0].attn.head_size
        # rot_pos_emb flattens [N, 2, head_dim//2] -> [N, head_dim]
        rotary_dim = head_dim

        logger.info(
            "ViTTpuCompileRunner.precompile: warming %d patch counts %s "
            "(embed_dim=%d, rotary_dim=%d).",
            len(patch_counts),
            patch_counts,
            embed_dim,
            rotary_dim,
        )

        for i, pc in enumerate(patch_counts):
            logger.info(
                "ViTTpuCompileRunner.precompile: [%d/%d] patch_count=%d ...",
                i + 1,
                len(patch_counts),
                pc,
            )
            x = torch.zeros(pc, 1, embed_dim, device=device, dtype=dtype)
            cu_seqlens = torch.tensor([0, pc], device=device, dtype=torch.int32)
            cos = torch.zeros(pc, rotary_dim, device=device, dtype=dtype)
            sin = torch.zeros(pc, rotary_dim, device=device, dtype=dtype)
            segment_ids = torch.zeros(1, pc, device=device, dtype=torch.int32)

            result = self._compiled_fn(x, cu_seqlens, cos, sin, segment_ids)
            # Force compilation to finish by synchronizing.
            tpu_sync([result], wait=True)

        logger.info("ViTTpuCompileRunner.precompile: done.")

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

        self._ensure_compiled()

        result = self._compiled_fn(
            x_3d, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, segment_ids
        )

        if output_indices is not None:
            result = result.index_select(0, output_indices)

        return result
