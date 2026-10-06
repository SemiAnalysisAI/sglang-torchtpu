# SPDX-License-Identifier: Apache-2.0
"""Explicit model state for GLM's opt-in direct TPU tracer."""

import torch

from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)


class GlmMoeDsaTpuCompilePolicy(DecoderTpuCompilePolicy):
    def get_model_extra_tensor_attrs(self) -> tuple[str, ...]:
        model = self.model_runner.model
        registered = {
            name
            for name, _ in (
                *model.named_parameters(remove_duplicate=False),
                *model.named_buffers(remove_duplicate=False),
            )
        }
        paths = list(super().get_model_extra_tensor_attrs())

        def include(path, value):
            # Keep helper aliases even when their tensor is also a parameter:
            # direct_compile must replace both references while tracing.
            if isinstance(value, torch.Tensor) and path not in registered:
                if path not in paths:
                    paths.append(path)

        # make_layers retains global indices and uses PPMissingLayer outside
        # this rank's interval. Discover actual children instead of assuming a
        # layer count, dense-layer prefix, or pipeline-parallel partition.
        for layer_name, layer in model.model.layers.named_children():
            prefix = f"model.layers.{layer_name}"
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                for name in (
                    "w_kc",
                    "w_vc",
                    "w_scale",
                    "w_scale_k",
                    "w_scale_v",
                    "w_kc_qrep",
                    "q_b_proj_qrep_weight",
                ):
                    include(
                        f"{prefix}.self_attn.{name}",
                        getattr(attention, name, None),
                    )

            topk = getattr(getattr(layer, "mlp", None), "topk", None)
            config = getattr(topk, "topk_config", None)
            if config is not None:
                include(
                    f"{prefix}.mlp.topk.topk_config.correction_bias",
                    getattr(config, "correction_bias", None),
                )

        return tuple(paths)
