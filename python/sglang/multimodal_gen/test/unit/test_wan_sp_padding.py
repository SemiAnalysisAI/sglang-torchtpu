"""Wan sequence-parallel tail padding reaches self-attention as tail metadata."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.models.dits import wanvideo
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_SP_SIZE = 4
# (frames, height, width) of the latents; patch size (1, 2, 2).
_DIVISIBLE = (2, 4, 8)  # 16 tokens: 4 per rank
_PADDED = (3, 6, 6)  # 27 tokens: 7 per rank, one pad token on the last rank


class _Stop(Exception):
    """Ends the forward once block 0's self-attention is reached."""


class _RecordingSelfAttention(USPAttention):
    def __init__(self, backend=AttentionBackendEnum.TORCH_SDPA):
        nn.Module.__init__(self)
        self.backend = backend
        self.calls = []

    def forward(self, q, k, v, **kwargs):
        self.calls.append(kwargs)
        raise _Stop


class _NonMaskingSelfAttention(nn.Module):
    """Stands in for SLA's MinimalA2AAttnOp, which takes no mask."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, q, k, v):
        self.calls += 1
        raise _Stop


class _PassThroughNorm(nn.Module):
    def forward(self, x, shift, scale):
        return x


@contextlib.contextmanager
def _sp_rank(rank):
    tp_group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=-1: x
    )
    sp_group = SimpleNamespace(
        world_size=_SP_SIZE,
        rank_in_group=rank,
        ulysses_world_size=_SP_SIZE,
        ring_world_size=1,
    )
    with (
        patch.object(parallel_state, "_TP", tp_group),
        patch.object(parallel_state, "_SP", sp_group),
    ):
        yield


def _tiny_wan():
    arch = WanVideoArchConfig(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=8,
        ffn_dim=16,
        num_layers=1,
    )
    return wanvideo.WanTransformer3DModel(WanVideoConfig(arch_config=arch), {})


def _run_until_self_attention(model, latent_shape, self_attention):
    block = model.blocks[0]
    # Only the path to self-attention matters; skip the fused norm kernels.
    block.norm1 = _PassThroughNorm()
    block.norm_q = block.norm_k = None
    block.attn1 = self_attention
    batch = SimpleNamespace(
        enable_sequence_shard=True, enable_teacache=False, enable_spectrum=False
    )
    context = SimpleNamespace(forward_batch=batch, attn_metadata=None)
    with patch.object(wanvideo, "get_forward_context", return_value=context):
        model(
            torch.randn(1, 4, *latent_shape),
            torch.randn(1, 5, 8),
            torch.tensor([500.0]),
        )


class TestWanSequenceShardPadding(unittest.TestCase):
    def setUp(self):
        previous = server_args_module._global_server_args
        server_args_module.set_global_server_args(
            SimpleNamespace(
                attention_backend=None,
                attention_backend_config=None,
                kv_gather_degree=1,
                sp_split_auto=False,
            )
        )
        self.addCleanup(server_args_module.set_global_server_args, previous)

    def _self_attention_calls(self, latent_shape, rank, backend=None):
        self_attention = (
            _RecordingSelfAttention(backend) if backend else _RecordingSelfAttention()
        )
        with _sp_rank(rank), self.assertRaises(_Stop):
            _run_until_self_attention(_tiny_wan(), latent_shape, self_attention)
        return self_attention.calls

    def test_divisible_token_count_passes_no_mask_meta(self):
        for rank in range(_SP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(self._self_attention_calls(_DIVISIBLE, rank), [{}])

    def test_padded_shards_pass_tail_meta_to_self_attention(self):
        for rank in range(_SP_SIZE):
            with self.subTest(rank=rank):
                (call,) = self._self_attention_calls(_PADDED, rank)
                meta = call["attn_mask_meta"]
                self.assertEqual((meta["pad_start"], meta["pad_end"]), (27, 28))
                self.assertEqual(meta["local_pad"], int(rank == _SP_SIZE - 1))

    def test_sparse_attention_keeps_attending_the_pad_keys(self):
        # Sparse kernels cannot drop the pad keys without a dense fallback, so
        # they get no metadata and run as before the padding was masked.
        (call,) = self._self_attention_calls(
            _PADDED, 1, backend=AttentionBackendEnum.SLIDING_TILE_ATTN
        )
        self.assertEqual(call, {})
        non_masking = _NonMaskingSelfAttention()
        with _sp_rank(1), self.assertRaises(_Stop):
            _run_until_self_attention(_tiny_wan(), _PADDED, non_masking)
        self.assertEqual(non_masking.calls, 1)


if __name__ == "__main__":
    unittest.main()
