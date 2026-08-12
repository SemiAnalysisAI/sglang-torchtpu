# SPDX-License-Identifier: Apache-2.0
"""On TPU, text_encoder defaults to torch_sdpa (pallas is non-causal only)."""

from unittest.mock import patch

from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs


def _make_args(**kwargs):
    return ServerArgs(model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers", **kwargs)


class TestTpuTextEncoderBackendDefault:
    def test_defaults_to_torch_sdpa_on_tpu(self):
        with patch(
            "sglang.multimodal_gen.runtime.server_args.server_args."
            "current_platform.is_tpu",
            return_value=True,
        ):
            args = _make_args()
        assert args.component_attention_backends.get("text_encoder") == "torch_sdpa"

    def test_explicit_user_choice_wins(self):
        with patch(
            "sglang.multimodal_gen.runtime.server_args.server_args."
            "current_platform.is_tpu",
            return_value=True,
        ):
            args = _make_args(
                component_attention_backends={"text_encoder": "tpu_pallas_flash"}
            )
        assert (
            args.component_attention_backends.get("text_encoder")
            == "tpu_pallas_flash"
        )

    def test_non_tpu_platforms_unchanged(self):
        with patch(
            "sglang.multimodal_gen.runtime.server_args.server_args."
            "current_platform.is_tpu",
            return_value=False,
        ):
            args = _make_args()
        assert args.component_attention_backends.get("text_encoder") is None
