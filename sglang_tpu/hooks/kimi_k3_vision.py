# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 vision encoder on TPU: segment-id attention, compiled per bucket.

attach_k3_vision_runner, encode_with_k3_vision_runner
  Why: Kimi-K3's tower runs its encoder eagerly on TPU, attending per image
       with SDPA, so every new image grid compiles the whole encoder's XLA
       programs while serving (the TODO in KimiK3ForConditionalGeneration.
       get_image_feature). The TPU runner and its compile policy
       (sglang_tpu.graph_runner.kimi_k3_vit_tpu_compile_runner) must stay out
       of core, which cannot import sglang_tpu.
  How: AFTER kimi_k3_vl.MoonViT3dEncoder.__init__ attaches the runner as the
       encoder's graph_runners, where KimiK3VisionTpuCompilePolicy (declared in
       core_compile_annotations) finds and compiles it. AROUND
       KimiK3VisionTower._forward_eager keeps the eager patch embedding,
       position embedding and temporal-pool merger, and sends the encoder
       through the runner: compiled per padded bucket under
       --enable-torch-compile, the same math eagerly otherwise.
  Upstream PR or issue: none; the CUDA path has its own graph runner
       (KimiK3ViTCudaGraphRunner) selected inside the tower.
  Remove when: the tower selects its accelerator runner through a platform
       seam instead of an is_cuda branch.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch


def attach_k3_vision_runner(result: None, encoder: Any, *args, **kwargs) -> None:
    """AFTER MoonViT3dEncoder.__init__ (Kimi-K3's): give the encoder its TPU runner."""
    from sglang_tpu.graph_runner.kimi_k3_vit_tpu_compile_runner import (
        KimiK3VitTpuCompileRunner,
    )

    encoder.graph_runners = KimiK3VitTpuCompileRunner(encoder)


def encode_with_k3_vision_runner(
    original_fn: Callable,
    tower: Any,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    *,
    grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
    forward_metadata: Any = None,
) -> Any:
    """AROUND KimiK3VisionTower._forward_eager: the encoder runs on the runner."""
    from sglang.srt.models.kimi_k3_vl import _resolve_grid_thw_list
    from sglang.srt.models.kimi_vl_moonvit import tpool_patch_merger

    runner = getattr(tower.encoder, "graph_runners", None)
    if runner is None or not runner.supports_segment_ids:
        return original_fn(
            tower,
            pixel_values,
            grid_thws,
            grid_thw_list=grid_thw_list,
            forward_metadata=forward_metadata,
        )
    if forward_metadata is not None:
        shapes = forward_metadata.grid_thw_list
        position_embeddings = forward_metadata.position_embeddings
    else:
        shapes = _resolve_grid_thw_list(grid_thws, grid_thw_list)
        position_embeddings = None
    hidden_states = tower.patch_embed(
        pixel_values,
        grid_thws,
        grid_thw_list=shapes,
        position_embeddings=position_embeddings,
    )
    freqs_cis = tower.encoder.rope_2d.get_freqs_cis(
        grid_thws, hidden_states.device, grid_thw_list=shapes
    )
    bounds = [0]
    for t, h, w in shapes:
        bounds.append(bounds[-1] + t * h * w)
    # Host-resident: the runner builds segment ids from it without a device read.
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
    hidden_states = runner.run(
        hidden_states,
        freqs_cis.real.contiguous(),
        freqs_cis.imag.contiguous(),
        cu_seqlens,
    )
    return tpool_patch_merger(
        hidden_states,
        grid_thws,
        merge_kernel_size=tower.merge_kernel_size,
        grid_thw_list=shapes,
    )
