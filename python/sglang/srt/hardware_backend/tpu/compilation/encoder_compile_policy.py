"""Compile-policy installation for TPU vision encoders.

FIXME: only the installer lives here. VisionTpuCompilePolicy,
Qwen3VLVisionTpuCompilePolicy and MoonVitTpuCompilePolicy still sit next to
their runners under tpu/graph_runner/; move them here so this module mirrors
decoder_compile_policy.py and graph_runner/ holds runners only.
"""

from __future__ import annotations

import torch

from sglang.srt.compilation.support_torch_compile import get_torch_compile_policy
from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    VisionTpuCompilePolicy,
)


def install_vision_compile_policies(
    model: torch.nn.Module,
) -> list[VisionTpuCompilePolicy]:
    """Install and warm every TPU vision compile policy attached to model.

    The ModelRunner-free counterpart of TpuGraphRunner's policy loop, for a
    process that owns a vision encoder but no ModelRunner -- the EPD encode
    server. Decoder policies are left alone: that process has no language model.

    Args:
        model: Module tree to scan for support_torch_compile annotations.

    Returns:
        The installed policies, in module-traversal order.

    Raises:
        RuntimeError: No vision policy installed. Silently falling back to eager
            would leave precompiled_max_patch_count reporting the env ceiling
            while the runner pads nothing, so every image recompiles.
    """
    installed: list[VisionTpuCompilePolicy] = []
    annotated = 0
    for module in model.modules():
        policy_cls = get_torch_compile_policy(module)
        if policy_cls is None or not issubclass(policy_cls, VisionTpuCompilePolicy):
            continue
        annotated += 1
        policy = policy_cls(
            owner=module,
            method_name="forward",
            method=module.forward,
            label=type(module).__name__,
        )
        if not policy.install():
            continue
        installed.append(policy)
        policy.precompile()

    if not installed:
        raise RuntimeError(
            f"{type(model).__name__} installed no TPU vision compile policy "
            f"({annotated} annotated module(s) found). The encoder would run "
            "eager and recompile per image shape. Check that the ViT attention "
            "backend supports segment IDs, and that the weights are on TPU "
            "(the runner is selected from the patch-embedding device)."
        )
    return installed
