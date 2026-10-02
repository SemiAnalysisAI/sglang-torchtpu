from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)

_CONFIG = "sglang.multimodal_gen.configs.pipeline_configs.ltx_2"


def _shard(fn, latents, *, rank, world_size=4, **batch_kwargs):
    batch = SimpleNamespace(**batch_kwargs)
    with (
        patch(f"{_CONFIG}.get_sp_world_size", return_value=world_size),
        patch(f"{_CONFIG}.get_sp_parallel_rank", return_value=rank),
    ):
        fn(batch, latents)
    return batch


@pytest.mark.parametrize(("frames", "padded"), [(16, False), (17, True)])
def test_video_sharding_records_global_padding_on_every_rank(frames, padded):
    config = LTX2PipelineConfig()
    tokens_per_frame = 4 * 4
    height = width = 4 * config.vae_scale_factor
    latents = torch.zeros(1, frames * tokens_per_frame, 8)
    for rank in range(4):
        batch = _shard(
            config.shard_latents_for_sp,
            latents,
            rank=rank,
            height=height,
            width=width,
        )
        assert batch.sp_video_padded is padded


@pytest.mark.parametrize(("frames", "padded"), [(128, False), (126, True)])
def test_audio_sharding_records_global_padding_on_every_rank(frames, padded):
    config = LTX2PipelineConfig()
    for rank in range(4):
        batch = _shard(
            config.shard_audio_latents_for_sp,
            torch.zeros(1, frames, 8),
            rank=rank,
        )
        assert batch.sp_audio_padded is padded


def test_unpadded_sp_shards_skip_the_self_attention_mask():
    batch = SimpleNamespace(sp_video_valid_token_count=8, sp_video_padded=False)
    mask = LTX2DenoisingStage._build_ltx2_sp_self_attention_mask(
        batch, modality="video", seq_len=8, batch_size=2, device=torch.device("cpu")
    )
    assert mask is None


def test_padded_sp_shards_keep_the_mask_on_full_ranks():
    # A full rank still builds its all-true mask when another rank is padded,
    # so every rank takes the masked path.
    full = SimpleNamespace(sp_video_valid_token_count=8, sp_video_padded=True)
    tail = SimpleNamespace(sp_video_valid_token_count=5, sp_video_padded=True)
    masks = [
        LTX2DenoisingStage._build_ltx2_sp_self_attention_mask(
            batch,
            modality="video",
            seq_len=8,
            batch_size=1,
            device=torch.device("cpu"),
        )
        for batch in (full, tail)
    ]
    assert masks[0].all()
    assert masks[1].tolist() == [[True] * 5 + [False] * 3]
