# SPDX-License-Identifier: Apache-2.0
"""TPU implementations of model architectures that core serves with GPU-only kernels."""


def register_tpu_models() -> None:
    from sglang.srt.models.registry import ModelRegistry

    # replaces core's entries of the same architecture names
    ModelRegistry.register("sglang_tpu.models", overwrite=True)
