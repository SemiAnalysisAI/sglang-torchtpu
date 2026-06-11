"""ViT TPU compile runner — wraps ViT blocks + merger with torch.compile."""

from __future__ import annotations

import inspect
import logging
import math
import os

import torch
import torch._dynamo
import torch.nn.functional as F
from torch_tpu._internal.sync.sync import synchronize as tpu_sync

from sglang.srt.hardware_backend.tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)

# Default upper bound for precompiled square patch counts. Real (often
# non-square) images are padded up to the nearest bucket at runtime, so this
# should exceed the largest patch count typical images produce; anything beyond
# falls back to a one-time recompile (see run()). MMMU reached 3400 patches, so
# 4900 = 70^2 leaves comfortable headroom (~29 buckets, ~5min precompile).
# Override with SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.
_DEFAULT_MAX_PATCHES = 4900


def _compute_patch_counts(merge_size, min_patches=196, max_patches=_DEFAULT_MAX_PATCHES):
    """Compute valid square-image patch counts from model config.

    Image dims are snapped to multiples of (patch_size * merge_size) by the
    preprocessor, so grid dims are always multiples of merge_size.  This
    enumerates the resulting square patch counts from min_patches to
    max_patches.  These double as the padding buckets used by ``run()``: each
    is a multiple of merge_size**2, so a padded sequence still splits cleanly
    into merger windows.
    """
    counts = []
    min_grid = max(merge_size, math.ceil(math.sqrt(min_patches) / merge_size) * merge_size)
    grid_dim = min_grid
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
        self._patch_buckets = None
        self._warned_oversize = set()

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

        # Match the cache limit used by tpu_graph_runner / cuda_graph_runner
        # so the ViT's many per-patch-count variants don't trip
        # FailOnRecompileLimitHit before the LLM runner gets a chance to
        # raise the limit itself.
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024

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

    def _resolve_patch_buckets(self):
        """Sorted list of patch-count buckets used for precompile + padding.

        Resolution order: SGLANG_TPU_VIT_PRECOMPILE_SIZES (comma list, or empty
        to disable) > square counts derived from the model's merge_size up to
        SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.  Cached after first call.
        """
        if self._patch_buckets is not None:
            return self._patch_buckets

        env_val = os.environ.get("SGLANG_TPU_VIT_PRECOMPILE_SIZES", None)
        if env_val is not None:
            if env_val.strip() == "":
                self._patch_buckets = []
                return self._patch_buckets
            buckets = sorted({int(x.strip()) for x in env_val.split(",") if x.strip()})
        else:
            merge_size = getattr(self.vit, "spatial_merge_size", 2)
            max_patches = int(
                os.environ.get(
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES", str(_DEFAULT_MAX_PATCHES)
                )
            )
            buckets = _compute_patch_counts(merge_size, max_patches=max_patches)

        self._patch_buckets = buckets
        return buckets

    def _bucket_for(self, seq_len):
        """Smallest precompiled bucket >= seq_len, or None if it exceeds the max."""
        for b in self._resolve_patch_buckets():
            if b >= seq_len:
                return b
        return None

    def precompile(self, patch_counts=None):
        """Warm up the torch.compile cache for common patch counts.

        Each unique patch count triggers a fresh XLA compilation on first
        encounter (~9-13 s).  ``run()`` pads every image up to the nearest of
        these buckets, so warming them here means real requests reuse a cached
        graph instead of recompiling per (often non-square) image size.

        Args:
            patch_counts: list of int patch counts to precompile for.  Defaults
                to the model-derived square buckets (see _resolve_patch_buckets).
        """
        if not self._supports_segment_ids:
            logger.info(
                "ViTTpuCompileRunner.precompile: skipped (segment_ids not "
                "supported, using eager fallback)."
            )
            return

        vit = self.vit

        if patch_counts is None:
            patch_counts = self._resolve_patch_buckets()

        if not patch_counts:
            logger.info(
                "ViTTpuCompileRunner.precompile: no patch counts to warm "
                "(disabled via SGLANG_TPU_VIT_PRECOMPILE_SIZES='')."
            )
            return

        self._ensure_compiled()

        device = vit.patch_embed.proj.weight.device
        dtype = vit.patch_embed.proj.weight.dtype

        # Derive tensor dimensions from the model.
        embed_dim = vit.blocks[0].norm1.normalized_shape[0]
        merge_sq = getattr(vit, "spatial_merge_size", 2) ** 2

        logger.info(
            "ViTTpuCompileRunner.precompile: warming %d patch counts %s "
            "(embed_dim=%d).",
            len(patch_counts),
            patch_counts,
            embed_dim,
        )

        # Match the runtime global state so dynamo GLOBAL_STATE guard
        # does not invalidate every precompiled graph.  Ensure
        # default_dtype=float32 and no_grad to match inference.
        init_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.no_grad():
                for i, pc in enumerate(patch_counts):
                    logger.info(
                        "ViTTpuCompileRunner.precompile: [%d/%d] patch_count=%d ...",
                        i + 1,
                        len(patch_counts),
                        pc,
                    )
                    x = torch.zeros(pc, 1, embed_dim, device=device, dtype=dtype)
                    # Use the model's own rot_pos_emb so cos/sin match the
                    # runtime width and dtype exactly. It returns head_dim//2
                    # width; VisionAttention doubles it to head_dim *inside* the
                    # compiled graph, so the graph input must be the rot_pos_emb
                    # output, NOT head_dim. pc is a perfect-square bucket, so a
                    # g x g grid yields exactly pc patches.
                    g = int(math.isqrt(pc))
                    cos, sin = vit.rot_pos_emb([[1, g, g]])
                    # cu_seqlens is kept at a stable length-2 shape across
                    # runtime requests; TPU vision attention uses segment_ids.
                    real = max(merge_sq, pc - merge_sq)
                    cu_seqlens = torch.tensor([0, pc], device=device, dtype=torch.int32)
                    segment_ids = torch.zeros(1, pc, device=device, dtype=torch.int32)
                    if pc > real:
                        segment_ids[0, real:] = 1

                    result = self._compiled_fn(x, cu_seqlens, cos, sin, segment_ids)
                    # Force compilation to finish by synchronizing.
                    tpu_sync([result], wait=True)

        finally:
            torch.set_default_dtype(init_dtype)

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

        self._ensure_compiled()

        merge_sq = getattr(self.vit, "spatial_merge_size", 2) ** 2
        cos = rotary_pos_emb_cos
        sin = rotary_pos_emb_sin

        # Pad the patch sequence up to the nearest precompiled bucket so the
        # compiled graph is reused instead of recompiling (~10-13 s) for every
        # unique (often non-square) image patch count. Padding patches go into
        # their own attention segment (segment id = num real segments) so they
        # never attend to real patches; the extra merged rows are sliced off.
        bucket_len = self._bucket_for(seq_len)
        padded = bucket_len is not None and bucket_len > seq_len

        if padded:
            pad = bucket_len - seq_len
            x_3d = F.pad(x_3d, (0, 0, 0, 0, 0, pad))
            cos = F.pad(cos, (0, 0, 0, pad))
            sin = F.pad(sin, (0, 0, 0, pad))
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)
            num_real_segs = int(cu_seqlens.shape[0]) - 1
            pad_seg = torch.full(
                (1, pad), num_real_segs, dtype=torch.int32, device=x.device
            )
            segment_ids = torch.cat([segment_ids, pad_seg], dim=1)
        else:
            if bucket_len is None and seq_len not in self._warned_oversize:
                self._warned_oversize.add(seq_len)
                buckets = self._resolve_patch_buckets()
                logger.warning(
                    "ViTTpuCompileRunner: patch count %d exceeds the largest "
                    "precompiled bucket %s — this image triggers a one-time ViT "
                    "recompile. Raise SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    seq_len,
                    buckets[-1] if buckets else None,
                )
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)

        if not getattr(self, "_logged_shapes", False):
            self._logged_shapes = True
            logger.info(
                "ViTTpuCompileRunner.run: seq_len=%d bucket=%s padded=%s "
                "x=%s cos=%s sin=%s seg=%s cu=%s",
                seq_len,
                bucket_len,
                padded,
                tuple(x_3d.shape),
                tuple(cos.shape),
                tuple(sin.shape),
                tuple(segment_ids.shape),
                tuple(cu_seqlens.shape),
            )

        # Keep cu_seqlens shape stable across different numbers of images/frames.
        # The TPU vision attention backend consumes segment_ids for masking; passing
        # runtime cu_seqlens through the compiled boundary would otherwise guard on
        # its length and cold-compile per segment count.
        graph_cu_seqlens = torch.tensor(
            [0, x_3d.shape[0]], device=x.device, dtype=torch.int32
        )
        result = self._compiled_fn(x_3d, graph_cu_seqlens, cos, sin, segment_ids)

        if padded:
            # Drop the merged rows that came from padding patches.
            result = result[: seq_len // merge_sq]

        if output_indices is not None:
            result = result.index_select(0, output_indices)

        return result
